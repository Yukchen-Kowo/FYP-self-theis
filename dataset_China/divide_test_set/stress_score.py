# -*- coding: utf-8 -*-
# ============================================================
# 文件名：stress_score.py
# 功能：独立计算 daily.csv 中每一天的压力分数 stress_score
# 适用场景：把原 timedit_stress_project 里的压力打分功能单独拿出来使用，
#          不依赖扩散模型，也不需要 timedit_stress_project 的其他文件。
#
# 需要安装的库：
#   pip install pandas numpy
#
# 默认用法：
#   python stress_score.py
#
# 默认输入 / 输出：
#   输入：当前目录下的 daily.csv
#   输出：daily_stress_scores.csv
#        scored_daily.csv
#
# 自定义输入输出文件：
#   python stress_score.py --input daily.csv --output daily_stress_scores.csv
#
# PowerShell 示例：
#   cd I:\code\final0325\0325\dataset_China
#   python .\stress_score.py
#
# 输入 CSV 必须包含以下列：
#   day_id, day_of_week, is_weekend, price, load, lambda, t
#
# 说明：
#   1. 每天默认有 96 个时间步，即 15 分钟粒度。
#   2. 脚本会按 day_id 和 t 排序，并在每个 day_id 内生成 step_in_day=0~95。
#   3. 输出 daily_stress_scores.csv 只包含 day_id 和 stress_score。
#   4. 输出 scored_daily.csv 会保留原始逐时间步数据，并给每一行附加对应当天的 stress_score。
# ============================================================

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# 参与压力打分的三个核心变量。
# 注意：lambda 是列名字符串，不是 Python 关键字本身，所以可以这样使用。
TARGET_COLS = ["price", "load", "lambda"]


def percentile_from_sorted(sorted_values: np.ndarray, value: float) -> float:
    """根据已经排序的数组，计算某个值的经验分位数，返回范围为 [0, 1]。"""
    sorted_values = np.asarray(sorted_values, dtype=np.float32)
    if sorted_values.size == 0:
        return 0.5
    idx = np.searchsorted(sorted_values, value, side="right")
    return float(idx) / float(sorted_values.size)


def topk_mean(values: np.ndarray, frac: float = 0.1) -> float:
    """计算数组中最高 frac 比例数值的均值，例如 frac=0.1 表示最高 10% 的均值。"""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * frac)))
    return float(np.mean(np.sort(values)[-k:]))


@dataclass
class StressConfig:
    """压力打分器配置。"""

    steps_per_day: int = 96
    min_group_size: int = 6
    active_threshold: float = 1.5
    deviation_clip: float = 8.0
    variable_weights: Optional[Dict[str, float]] = None
    variable_component_weights: Optional[Dict[str, float]] = None
    joint_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        # 不同变量在最终 stress_score 中的权重。
        # price/load/lambda 分别衡量电价、负荷和到达强度压力；joint 衡量多变量同时异常的联合压力。
        if self.variable_weights is None:
            self.variable_weights = {
                "price": 0.35,
                "load": 0.35,
                "lambda": 0.15,
                "joint": 0.15,
            }

        # 单个变量内部：水平异常、爬坡变化、持续时间三部分的权重。
        if self.variable_component_weights is None:
            self.variable_component_weights = {
                "level": 0.5,
                "ramp": 0.25,
                "duration": 0.25,
            }

        # 联合压力：两个及以上变量同时活跃、三个变量同时活跃的权重。
        if self.joint_weights is None:
            self.joint_weights = {
                "two_or_more": 0.7,
                "three": 0.3,
            }


class StressScorer:
    """
    独立版每日压力打分器。

    它从历史 daily.csv 中学习基线分布，然后给每个 day_id 计算一个 stress_score。
    该类已经把原 timedit_stress_project 中 stress.py 需要的辅助函数合并进来了，
    因此不再依赖 timedit_stress_project 的其他文件。
    """

    def __init__(self, config: Optional[StressConfig] = None) -> None:
        self.config = config or StressConfig()
        self.target_cols = TARGET_COLS.copy()

        # 基线匹配优先级：
        # 1. 同 day_of_week + 同 step_in_day
        # 2. 同 is_weekend + 同 step_in_day
        # 3. 只看同 step_in_day
        self.group_strategies: List[Tuple[str, ...]] = [
            ("day_of_week", "step_in_day"),
            ("is_weekend", "step_in_day"),
            ("step_in_day",),
        ]

        self.baseline_tables: Dict[str, Dict[Tuple[str, ...], pd.DataFrame]] = {}
        self.diff_scales: Dict[str, float] = {}
        self.sorted_feature_values: Dict[str, np.ndarray] = {}
        self.sorted_total_values: Optional[np.ndarray] = None
        self.training_daily_scores_: Optional[pd.DataFrame] = None
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "StressScorer":
        """在历史数据上拟合基线，并计算训练集内每天的 stress_score。"""
        self._validate_input(df)
        self.baseline_tables = {var: {} for var in self.target_cols}

        for var in self.target_cols:
            for strategy in self.group_strategies:
                agg = (
                    df.groupby(list(strategy))[var]
                    .agg(
                        count="count",
                        median="median",
                        q25=lambda s: s.quantile(0.25),
                        q75=lambda s: s.quantile(0.75),
                    )
                    .reset_index()
                )
                self.baseline_tables[var][strategy] = agg

            # 用 day_id 内相邻时间步的绝对差，估计该变量的日内变化尺度。
            diffs = df.groupby("day_id")[var].diff().abs().dropna().to_numpy(dtype=np.float32)
            if diffs.size == 0:
                scale = 1.0
            else:
                scale = float(np.quantile(diffs, 0.75))
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = float(np.mean(diffs) + 1e-6) if diffs.size > 0 else 1.0
            self.diff_scales[var] = max(scale, 1e-6)

        enriched = self._compute_row_deviations(df.copy())
        daily_raw = self._compute_daily_raw_features(enriched)
        calibrated = self._calibrate_daily_features(daily_raw)
        self.training_daily_scores_ = calibrated.copy()
        self.fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合打分器，并把每一天的 stress_score 合并回逐时间步数据。"""
        self.fit(df)
        if self.training_daily_scores_ is None:
            raise RuntimeError("StressScorer 拟合失败，未生成每日压力分数。")
        merged = df.merge(
            self.training_daily_scores_[["day_id", "stress_score"]],
            on="day_id",
            how="left",
        )
        if merged["stress_score"].isna().any():
            raise RuntimeError("有部分行未能分配到 stress_score，请检查 day_id 是否异常。")
        return merged

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已经拟合好的打分器，对新的 DataFrame 计算每日 stress_score。"""
        self._require_fitted()
        self._validate_input(df)
        enriched = self._compute_row_deviations(df.copy())
        daily_raw = self._compute_daily_raw_features(enriched)
        return self._apply_calibration_to_raw(daily_raw)

    def score_generated_windows(
        self,
        windows: np.ndarray,
        day_of_week: Sequence[int],
        is_weekend: Sequence[int],
    ) -> pd.DataFrame:
        """
        对形状为 [N, 96, 3] 的生成场景窗口打分。

        这里 3 个通道默认按 price, load, lambda 排列。
        这个函数不是生成 daily_stress_scores.csv 必需的，只是保留给以后评估生成样本使用。
        """
        self._require_fitted()
        if windows.ndim != 3 or windows.shape[-1] != len(self.target_cols):
            raise ValueError("windows 必须是形状 [N, steps_per_day, 3] 的数组。")
        if len(day_of_week) != windows.shape[0] or len(is_weekend) != windows.shape[0]:
            raise ValueError("day_of_week 和 is_weekend 的长度必须等于窗口数量 N。")

        rows: List[Dict[str, Union[float, int]]] = []
        for day_idx in range(windows.shape[0]):
            for step in range(windows.shape[1]):
                record: Dict[str, Union[float, int]] = {
                    "day_id": int(day_idx),
                    "step_in_day": int(step),
                    "day_of_week": int(day_of_week[day_idx]),
                    "is_weekend": int(is_weekend[day_idx]),
                }
                for ch_idx, col in enumerate(self.target_cols):
                    record[col] = float(windows[day_idx, step, ch_idx])
                rows.append(record)
        return self.score_dataframe(pd.DataFrame(rows))

    def _validate_input(self, df: pd.DataFrame) -> None:
        """检查打分所需字段是否齐全。"""
        required = {"day_id", "step_in_day", "day_of_week", "is_weekend"}
        required.update(self.target_cols)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"压力打分缺少必要列: {sorted(missing)}")

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("StressScorer 尚未拟合，请先调用 fit 或 fit_transform。")

    def _lookup_baseline(self, row: pd.Series, var: str) -> Tuple[float, float]:
        """按优先级查找某一行、某一变量对应的历史基线中位数和尺度。"""
        for strategy in self.group_strategies:
            table = self.baseline_tables[var][strategy]
            mask = np.ones(len(table), dtype=bool)
            for key in strategy:
                mask &= table[key].to_numpy() == row[key]
            matched = table.loc[mask]
            if not matched.empty and int(matched["count"].iloc[0]) >= self.config.min_group_size:
                median = float(matched["median"].iloc[0])
                scale = float((matched["q75"].iloc[0] - matched["q25"].iloc[0]) / 1.349)
                return median, max(scale, 1e-6)

        # 如果分组样本不足，就退回到全局中位数基线。
        global_series = pd.concat(
            [self.baseline_tables[var][strategy]["median"] for strategy in self.group_strategies],
            axis=0,
            ignore_index=True,
        )
        fallback_median = float(global_series.median()) if not global_series.empty else 0.0
        return fallback_median, 1.0

    def _compute_row_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """逐时间步计算相对基线的正向偏离和日内变化强度。"""
        for var in self.target_cols:
            medians = []
            scales = []
            for _, row in df.iterrows():
                median, scale = self._lookup_baseline(row, var)
                medians.append(median)
                scales.append(scale)

            medians_arr = np.asarray(medians, dtype=np.float32)
            scales_arr = np.asarray(scales, dtype=np.float32)
            raw = df[var].to_numpy(dtype=np.float32)

            # 只关注高于基线的正向压力，低于基线的部分截断为 0。
            z = (raw - medians_arr) / np.maximum(scales_arr, 1e-6)
            z_pos = np.clip(z, 0.0, self.config.deviation_clip)

            df[f"{var}_baseline"] = medians_arr
            df[f"{var}_scale"] = scales_arr
            df[f"{var}_dev_pos"] = z_pos

            # 归一化日内变化幅度，用于衡量 ramp pressure。
            diffs = df.groupby("day_id")[var].diff().fillna(0.0).abs().to_numpy(dtype=np.float32)
            df[f"{var}_abs_diff_norm"] = diffs / max(self.diff_scales[var], 1e-6)
        return df

    def _compute_daily_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """把逐时间步压力特征汇总成每日 raw stress features。"""
        records: List[Dict[str, Union[float, int]]] = []
        for day_id, g in df.groupby("day_id", sort=True):
            row: Dict[str, Union[float, int]] = {
                "day_id": int(day_id),
                "day_of_week": int(g["day_of_week"].iloc[0]),
                "is_weekend": int(g["is_weekend"].iloc[0]),
            }
            active_cols = []

            for var in self.target_cols:
                dev = g[f"{var}_dev_pos"].to_numpy(dtype=np.float32)
                diffs = g[f"{var}_abs_diff_norm"].to_numpy(dtype=np.float32)

                # level_raw：这一天的高分位压力水平。
                level_raw = 0.6 * float(np.quantile(dev, 0.95)) + 0.4 * topk_mean(dev, frac=0.10)

                # duration_raw：压力超过阈值的时间比例。
                duration_raw = float(np.mean(dev > self.config.active_threshold))

                # ramp_raw：日内变化强度的高分位值。
                ramp_raw = float(np.quantile(diffs, 0.95)) if diffs.size > 0 else 0.0

                row[f"{var}_level_raw"] = level_raw
                row[f"{var}_duration_raw"] = duration_raw
                row[f"{var}_ramp_raw"] = ramp_raw
                active_cols.append(dev > self.config.active_threshold)

            # joint_raw：多个变量同时处于高压力状态的比例。
            active_mat = np.column_stack(active_cols)
            num_active = active_mat.sum(axis=1)
            row["joint_raw"] = (
                self.config.joint_weights["two_or_more"] * float(np.mean(num_active >= 2))
                + self.config.joint_weights["three"] * float(np.mean(num_active == 3))
            )
            records.append(row)
        return pd.DataFrame(records)

    def _calibrate_daily_features(self, daily_raw: pd.DataFrame) -> pd.DataFrame:
        """把 raw features 转换成分位数特征，并进一步得到 stress_score。"""
        feature_cols = [
            *(f"{var}_{metric}_raw" for var in self.target_cols for metric in ["level", "ramp", "duration"]),
            "joint_raw",
        ]
        for col in feature_cols:
            self.sorted_feature_values[col] = np.sort(daily_raw[col].to_numpy(dtype=np.float32))

        calibrated = self._apply_calibration_to_raw(daily_raw)
        self.sorted_total_values = np.sort(calibrated["stress_unscaled"].to_numpy(dtype=np.float32))
        calibrated["stress_score"] = calibrated["stress_unscaled"].apply(
            lambda v: percentile_from_sorted(self.sorted_total_values, float(v))
        )
        return calibrated

    def _apply_calibration_to_raw(self, daily_raw: pd.DataFrame) -> pd.DataFrame:
        """应用已经学到的分位数标定，把每日 raw features 转成最终分数。"""
        out = daily_raw.copy()

        for var in self.target_cols:
            for metric in ["level", "ramp", "duration"]:
                raw_col = f"{var}_{metric}_raw"
                pct_col = f"{var}_{metric}_pct"
                sorted_vals = self.sorted_feature_values.get(raw_col)
                if sorted_vals is None:
                    raise RuntimeError(f"缺少 {raw_col} 的标定值。")
                out[pct_col] = out[raw_col].apply(lambda v: percentile_from_sorted(sorted_vals, float(v)))

        joint_sorted = self.sorted_feature_values.get("joint_raw")
        if joint_sorted is None:
            raise RuntimeError("缺少 joint_raw 的标定值。")
        out["joint_pct"] = out["joint_raw"].apply(lambda v: percentile_from_sorted(joint_sorted, float(v)))

        weights = self.config.variable_component_weights
        for var in self.target_cols:
            out[f"{var}_stress"] = (
                weights["level"] * out[f"{var}_level_pct"]
                + weights["ramp"] * out[f"{var}_ramp_pct"]
                + weights["duration"] * out[f"{var}_duration_pct"]
            )

        var_weights = self.config.variable_weights
        out["stress_unscaled"] = (
            var_weights["price"] * out["price_stress"]
            + var_weights["load"] * out["load_stress"]
            + var_weights["lambda"] * out["lambda_stress"]
            + var_weights["joint"] * out["joint_pct"]
        )

        if self.sorted_total_values is not None and len(self.sorted_total_values) > 0:
            out["stress_score"] = out["stress_unscaled"].apply(
                lambda v: percentile_from_sorted(self.sorted_total_values, float(v))
            )
        return out


def prepare_daily_dataframe(input_file: Union[str, Path]) -> pd.DataFrame:
    """读取 daily.csv，检查必要列，并生成 step_in_day。"""
    input_file = Path(input_file)
    print(f"正在读取数据: {input_file}")
    df = pd.read_csv(input_file)

    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    required_columns = {"day_id", "day_of_week", "is_weekend", "price", "load", "lambda", "t"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"数据中缺少必要列: {sorted(missing_columns)}。"
            "请确保 daily.csv 包含: day_id, day_of_week, is_weekend, price, load, lambda, t"
        )

    print("正在计算 step_in_day...")
    df = df.sort_values(["day_id", "t"]).reset_index(drop=True)

    # 比 t % 96 更稳：按每个 day_id 内部排序后的顺序生成 0~95。
    # 这样即使 t 是真实时间戳或不连续整数，也不会影响 step_in_day。
    df["step_in_day"] = df.groupby("day_id").cumcount().astype(int)

    print(f"step_in_day 范围: {df['step_in_day'].min()} 到 {df['step_in_day'].max()}")
    if df["step_in_day"].max() > 95 or df["step_in_day"].min() < 0:
        print("警告: step_in_day 的范围超出 0~95，请检查是否每一天正好 96 条记录。")

    print("数据验证:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 总天数: {df['day_id'].nunique()}")
    print(f"  - 平均每天时间步数: {df.groupby('day_id').size().mean():.1f}")

    day_sizes = df.groupby("day_id").size()
    bad_days = day_sizes[day_sizes != 96]
    if len(bad_days) > 0:
        print(f"警告: 有 {len(bad_days)} 天不是 96 个时间步。脚本仍会运行，但建议先检查数据完整性。")

    return df


def analyze_daily_data(input_file: Union[str, Path] = "daily.csv") -> None:
    """在正式打分前打印 daily.csv 的基本结构，方便排查数据问题。"""
    try:
        df = pd.read_csv(input_file)
        print("=" * 60)
        print("数据结构分析")
        print("=" * 60)
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前 5 行数据:")
        print(df.head())
        print("\n基本统计信息:")
        print(df.describe())

        if {"day_id", "price", "load", "lambda"}.issubset(df.columns):
            print("\n按 day_id 分组的统计:")
            daily_stats = df.groupby("day_id").agg({
                "price": ["count", "mean", "std"],
                "load": ["count", "mean", "std"],
                "lambda": ["count", "mean", "std"],
            }).round(4)
            print(daily_stats.head(10))
            print(f"\n总天数: {df['day_id'].nunique()}")
            print(f"每天的平均时间步数: {df.groupby('day_id').size().mean():.1f}")

        if {"day_id", "t"}.issubset(df.columns):
            df = df.sort_values(["day_id", "t"]).reset_index(drop=True)
            df["step_in_day"] = df.groupby("day_id").cumcount().astype(int)
            print("\nstep_in_day 统计:")
            print(f"  范围: {df['step_in_day'].min()} 到 {df['step_in_day'].max()}")
            print(f"  唯一值数量: {df['step_in_day'].nunique()}")
            steps_per_day = df.groupby("day_id")["step_in_day"].nunique()
            print("\n每天的唯一时间步数:")
            print(f"  平均: {steps_per_day.mean():.1f}")
            print(f"  最小: {steps_per_day.min()}")
            print(f"  最大: {steps_per_day.max()}")
            if steps_per_day.min() < 96:
                print("  警告: 某些天的时间步数不足 96 个。")
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"分析数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_daily_stress_scores(
    input_file: Union[str, Path] = "daily.csv",
    output_file: Union[str, Path] = "daily_stress_scores.csv",
    scored_output_file: Optional[Union[str, Path]] = None,
) -> bool:
    """从逐时间步 daily.csv 生成每日 daily_stress_scores.csv。"""
    try:
        input_file = Path(input_file)
        output_file = Path(output_file)

        if scored_output_file is None:
            scored_output_file = input_file.with_name("scored_" + input_file.name)
        else:
            scored_output_file = Path(scored_output_file)

        df = prepare_daily_dataframe(input_file)

        print("\n正在配置 StressScorer...")
        config = StressConfig(
            steps_per_day=96,
            min_group_size=6,
            active_threshold=1.5,
            deviation_clip=8.0,
        )

        print("正在训练 StressScorer 并计算压力分数...")
        scorer = StressScorer(config=config)
        scored_df = scorer.fit_transform(df)

        print("正在提取每日 stress_score...")
        daily_scores = scored_df.groupby("day_id").first()[["stress_score"]].reset_index()

        print(f"正在保存每日压力分数到: {output_file}")
        daily_scores.to_csv(output_file, index=False)

        print(f"正在保存带 stress_score 的逐时间步数据到: {scored_output_file}")
        scored_df.to_csv(scored_output_file, index=False)

        print("\n生成的 daily_stress_scores.csv 统计信息:")
        print(f"总天数: {len(daily_scores)}")
        print(f"stress_score 范围: [{daily_scores['stress_score'].min():.4f}, {daily_scores['stress_score'].max():.4f}]")
        print(f"平均 stress_score: {daily_scores['stress_score'].mean():.4f}")
        print("\n前 5 天的 stress_score:")
        print(daily_scores.head())

        print(f"\n成功生成: {output_file}")
        print(f"同时生成: {scored_output_file}")
        return True

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_with_custom_config(
    input_file: Union[str, Path] = "daily.csv",
    output_file: Union[str, Path] = "daily_stress_scores_custom.csv",
) -> Optional[pd.DataFrame]:
    """使用自定义权重和阈值生成 stress_score；如不需要可忽略该函数。"""
    try:
        df = prepare_daily_dataframe(input_file)

        custom_config = StressConfig(
            steps_per_day=96,
            min_group_size=4,
            active_threshold=1.2,
            deviation_clip=6.0,
            variable_weights={"price": 0.4, "load": 0.3, "lambda": 0.2, "joint": 0.1},
        )

        scorer = StressScorer(config=custom_config)
        scored_df = scorer.fit_transform(df)
        daily_scores = scored_df.groupby("day_id").first()[["stress_score"]].reset_index()
        daily_scores.to_csv(output_file, index=False)
        print(f"使用自定义配置生成 {output_file} 成功。")
        return daily_scores

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="独立计算 daily.csv 的每日 stress_score，不依赖 timedit_stress_project。"
    )
    parser.add_argument(
        "--input",
        default="daily.csv",
        help="输入 CSV 文件路径，默认是当前目录下的 daily.csv。",
    )
    parser.add_argument(
        "--output",
        default="daily_stress_scores.csv",
        help="输出每日压力分数 CSV，默认是 daily_stress_scores.csv。",
    )
    parser.add_argument(
        "--scored-output",
        default=None,
        help="输出带 stress_score 的逐时间步 CSV。默认会保存为 scored_输入文件名，例如 scored_daily.csv。",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="跳过正式打分前的数据结构分析。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_analyze:
        print("正在分析数据...")
        analyze_daily_data(args.input)

    print("\n" + "=" * 60)
    print("开始生成 stress scores...")
    print("=" * 60 + "\n")

    success = generate_daily_stress_scores(
        input_file=args.input,
        output_file=args.output,
        scored_output_file=args.scored_output,
    )

    if success:
        print("\n" + "=" * 50)
        print("生成完成。")
    else:
        print("\n生成失败，请检查上面的错误信息和 daily.csv 数据格式。")
