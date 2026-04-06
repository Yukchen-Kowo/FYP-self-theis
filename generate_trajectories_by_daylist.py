"""
generate_trajectories_by_daylist.py
===================================
基于指定的 day_id 列表（来自 low_stress.csv / middle_stress.csv / high_stress.csv /
random_50_from_remaining.csv 等文件）生成 **恰好对应这些天** 的 optimal 轨迹。

核心改动（相对于原 generate_trajectories.py）：
  1. 新增 --day_list_csv 参数，指定一个包含 day_id 列的 CSV 文件
  2. 从该 CSV 提取去重、排序后的 day_id 列表，自动覆盖 n_trajectories
  3. 强制将 config['random_day'] 设为 False
  4. 在每条轨迹生成前，根据 day_id 计算对应日期，
     将 temp_env.sim_starting_date 设为该日期，
     使得 temp_env.reset() 精确使用这一天的环境数据
  5. 因此：50 个 day_id → 恰好 50 条轨迹，一一对应，不重复不遗漏

用法示例：
  python generate_trajectories_by_daylist.py \
      --config_file ./config_files/PST_V2G_ProfixMax_25.yaml \
      --dataset optimal \
      --day_list_csv ./stress_splits/low_stress.csv \
      --use_generated True
"""

import os
import time
import datetime
import numpy as np
import pickle
import yaml
from tqdm import tqdm
import shutil
import gzip

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.rl_agent.reward import (SquaredTrackingErrorReward,
                                     ProfitMax_TrPenalty_UserIncentives,
                                     profit_maximization, SimpleReward)
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.baselines.heuristics import RandomAgent, RoundRobin_GF, ChargeAsFastAsPossible
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMax_state

import pandas as pd


# ============================================================
# 工具函数：day_id → datetime
# ============================================================
BASE_DATE = datetime.datetime(2022, 1, 1)   # 数据集起始日


def day_id_to_date(day_id: int, hour: int = 0, minute: int = 0) -> datetime.datetime:
    """将 day_id（0-based，距 2022-01-01 的天数）转换为 datetime。"""
    return BASE_DATE.replace(hour=hour, minute=minute) + datetime.timedelta(days=int(day_id))


def load_day_ids_from_csv(csv_path: str):
    """从 split CSV 中读取去重、排序后的 day_id 列表。"""
    df = pd.read_csv(csv_path)
    if "day_id" not in df.columns:
        raise ValueError(f"CSV 文件 {csv_path} 中没有 'day_id' 列")
    day_ids = sorted(df["day_id"].unique().tolist())
    return day_ids


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":

    # ---------- 解析参数 ----------
    # 在原有 arg_parser 基础上手动添加 --day_list_csv
    import sys
    # 先用 argparse 解析原有参数，然后提取 day_list_csv
    import argparse

    # 复制一份 arg_parser 的逻辑并增加 day_list_csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of the experiment")
    parser.add_argument("--render_train", default=False, type=bool)
    parser.add_argument("--render_eval", default=True, type=bool)
    parser.add_argument("--load_model", default=False, type=bool)
    parser.add_argument("--save_dir", default="./saved_models/")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--timesteps", default=10*1e6, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--replay_size", default=1e5, type=int)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.001)
    parser.add_argument("--noise_stddev", default=0.3, type=int)
    parser.add_argument("--hidden_size", nargs=2, default=[256, 256], type=tuple)
    parser.add_argument("--wandb", default=False, type=bool)
    parser.add_argument("--config_file",
                        default="./config_files/PST_V2G_ProfixMax_25.yaml")
    parser.add_argument("--n_test_cycles", default=50, type=int)
    parser.add_argument("--n_trajectories", "--n", default=2, type=int)
    parser.add_argument("--save_eval_replays", "--s", default=False,
                        action="store_true")
    parser.add_argument("--dataset", default="random", type=str)
    parser.add_argument("--use_generated", default=False, type=bool)

    # ★ 新增参数
    parser.add_argument("--day_list_csv", type=str, default=None,
                        help="包含 day_id 列的 CSV 文件路径。"
                             "指定后将严格按其中的 day_id 逐天生成轨迹，"
                             "自动覆盖 n_trajectories 和 random_day。")

    args = parser.parse_args()

    # ---------- 读取 day_id 列表（如果指定了 --day_list_csv） ----------
    use_day_list = args.day_list_csv is not None
    if use_day_list:
        day_ids = load_day_ids_from_csv(args.day_list_csv)
        n_trajectories = len(day_ids)
        print(f"[day_list mode] 从 {args.day_list_csv} 读取到 {n_trajectories} 个 day_id")
        print(f"  day_id 范围: {day_ids[0]} ~ {day_ids[-1]}")
    else:
        day_ids = None
        n_trajectories = args.n_trajectories

    # ---------- 环境与配置 ----------
    SAVE_EVAL_REPLAYS = args.save_eval_replays
    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state
    problem = args.config_file.split("/")[-1].split(".")[0]

    # ★ 读取 config 并强制关闭 random_day（如果使用 day_list）
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    if use_day_list:
        config['random_day'] = False
        # 用 day_ids[0] 的日期初始化，后续在循环里会逐条覆盖
        init_date = day_id_to_date(day_ids[0])
        config['year'] = init_date.year
        config['month'] = init_date.month
        config['day'] = init_date.day

        # 将修改后的 config 写入临时 yaml，供 EV2Gym 读取
        import tempfile
        tmp_config_path = os.path.join(tempfile.gettempdir(), f"_tmp_config_{os.getpid()}.yaml")
        with open(tmp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        config_file_to_use = tmp_config_path
    else:
        config_file_to_use = args.config_file

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]

    env = EV2Gym(config_file=config_file_to_use,
                 state_function=state_function,
                 reward_function=reward_function,
                 save_replay=SAVE_EVAL_REPLAYS,
                 use_generated=args.use_generated)

    temp_env = EV2Gym(config_file=config_file_to_use,
                      save_replay=True,
                      reward_function=reward_function,
                      state_function=state_function,
                      use_generated=args.use_generated)

    trajectories = []
    if args.dataset not in ["random", "optimal", "bau",
                            "mixed_bau_50", "mixed_bau_25", "mixed_bau_75"]:
        raise ValueError(f"Trajectories type {args.dataset} not supported")

    trajecotries_type = args.dataset

    file_name = (f"{problem}_{trajecotries_type}"
                 f"_{number_of_charging_stations}_{n_trajectories}.pkl")
    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # make eval replay folder
    if SAVE_EVAL_REPLAYS:
        if not os.path.exists("eval_replays"):
            os.makedirs("eval_replays")
        file_name = (f"{problem}_{trajecotries_type}"
                     f"_{number_of_charging_stations}_{n_trajectories}")
        save_folder_path = f"./eval_replays/" + file_name
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        print(f"Saving evaluation replays to {save_folder_path}")

    epoch = 0

    for i in tqdm(range(n_trajectories)):

        # ★★★ 核心改动：在生成每条轨迹前，锁定当天的日期 ★★★
        if use_day_list:
            target_day_id = day_ids[i]
            target_date = day_id_to_date(
                target_day_id,
                hour=config.get('hour', 0),
                minute=config.get('minute', 0),
            )
            # 更新 temp_env 的日期（影响下一次 reset）
            temp_env.sim_starting_date = target_date
            temp_env.config['random_day'] = False
            # 同步更新 env 的日期（用于非-optimal 类型）
            env.sim_starting_date = target_date
            env.config['random_day'] = False

        trajectory_i = {"observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                        "action_mask": [],
                        }

        epoch_return = 0

        if trajecotries_type == "random":
            agent = RandomAgent(env)
        elif trajecotries_type == "bau":
            agent = RoundRobin_GF(env)
        elif trajecotries_type == "mixed_bau_50":
            if i % 2 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)
        elif trajecotries_type == "mixed_bau_25":
            if i % 4 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)
        elif trajecotries_type == "mixed_bau_75":
            if i % 4 == 0:
                agent = RandomAgent(env)
            else:
                agent = RoundRobin_GF(env)
        elif trajecotries_type == "optimal":
            from ev2gym.baselines.gurobi_models.PST_V2G_profit_max_mo \
                import mo_PST_V2GProfitMaxOracleGB

            _, _ = temp_env.reset()

            # ★ 验证日期是否正确（调试用，可注释掉）
            if use_day_list:
                actual_date = temp_env.sim_date
                expected_date = target_date
                assert actual_date.date() == expected_date.date(), \
                    (f"日期不匹配！ day_id={target_day_id}, "
                     f"expected={expected_date.date()}, actual={actual_date.date()}")
                print(f"  [轨迹 {i}] day_id={target_day_id}, "
                      f"sim_date={actual_date.strftime('%Y-%m-%d')}")

            agent = ChargeAsFastAsPossible()
            for _ in range(temp_env.simulation_length):
                actions = agent.get_action(temp_env)
                new_state, reward, done, truncated, stats = temp_env.step(actions)
                if done:
                    break

            new_replay_path = f"./replay/replay_{temp_env.sim_name}.pkl"
            timelimit = 180
            agent = mo_PST_V2GProfitMaxOracleGB(new_replay_path,
                                                 timelimit=timelimit,
                                                 MIPGap=None)

        elif trajecotries_type == "mpc":
            from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2
            agent = eMPC_V2G_v2(env,
                                control_horizon=10,
                                MIPGap=0.1,
                                time_limit=30,
                                verbose=False)
        else:
            raise ValueError(
                f"Trajectories type {trajecotries_type} not supported")

        if trajecotries_type == "optimal":
            env = EV2Gym(config_file=config_file_to_use,
                         load_from_replay_path=new_replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         save_replay=SAVE_EVAL_REPLAYS,
                         )
            os.remove(new_replay_path)

        state, _ = env.reset()

        if SAVE_EVAL_REPLAYS:
            env.eval_mode = "optimal"

        while True:
            actions = agent.get_action(env)
            new_state, reward, done, truncated, stats = env.step(actions)

            trajectory_i["observations"].append(state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)
            trajectory_i["action_mask"].append(stats['action_mask'])

            state = new_state
            if done:
                if SAVE_EVAL_REPLAYS:
                    replay_path = (env.replay_path + 'replay_'
                                   + env.sim_name + '.pkl')
                    new_replay_path = (f"./eval_replays/{file_name}"
                                       f"/replay_{env.sim_name}_{i}.pkl")
                    shutil.move(replay_path, new_replay_path)
                break

        print(f'Stats: {env.stats["total_reward"]}')
        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])
        trajectory_i["action_mask"] = np.array(trajectory_i["action_mask"])

        trajectories.append(trajectory_i)

        if trajecotries_type == "optimal":
            divident = 100
        else:
            divident = 1000

        if i % divident == 0 and not SAVE_EVAL_REPLAYS and i > 0:
            print(f'Saving trajectories to {save_folder_path + file_name}')
            with gzip.open(save_folder_path + file_name + ".gz", 'wb') as f:
                pickle.dump(trajectories, f)

    env.close()

    # 清理临时 config
    if use_day_list and os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    if SAVE_EVAL_REPLAYS:
        print(f'Generated {n_trajectories} trajectories and saved '
              f'them in {save_folder_path}')
    else:
        print(f'Saving trajectories to {save_folder_path + file_name}')
        with gzip.open(save_folder_path + file_name + ".gz", 'wb') as f:
            pickle.dump(trajectories, f)

        with gzip.open(save_folder_path + file_name + ".gz", 'rb') as f:
            loaded_data = pickle.load(f)

        print(loaded_data[0]["observations"].shape)
        print(loaded_data[0]["actions"].shape)
        print(loaded_data[0]["rewards"].shape)
        print(loaded_data[0]["dones"].shape)
        print(loaded_data[0]["action_mask"].shape)

    # ★ 打印 day_id 与轨迹的对应关系，方便核对
    if use_day_list:
        print(f"\n===== day_id → trajectory 映射 =====")
        for idx, did in enumerate(day_ids):
            d = day_id_to_date(did)
            print(f"  trajectory[{idx}] ← day_id={did} ({d.strftime('%Y-%m-%d')})")