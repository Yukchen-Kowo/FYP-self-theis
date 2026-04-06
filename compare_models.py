"""
compare_models.py
=================
汇总多个模型在多个测试集上的评估结果，生成论文级别的对比表和图。

用法：
  python compare_models.py --eval_dirs eval_results/baseline_low eval_results/baseline_mid eval_results/baseline_high ... --labels "Baseline|Low" "Baseline|Mid" ... --output_dir ./comparison_results/

更简单的用法（自动扫描目录结构）：
  python compare_models.py --root_dir ./eval_results/ --output_dir ./comparison_results/

  预期 root_dir 结构：
  eval_results/
  ├── baseline/
  │   ├── low/summary.json
  │   ├── middle/summary.json
  │   ├── high/summary.json
  │   └── random50/summary.json
  ├── cvae/
  │   ├── low/summary.json
  │   ...
  ├── gan/
  │   ...
  └── diffusion/
      ...

输出：
  output_dir/
  ├── comparison_table.csv          # 论文 Table 格式（所有模型 × 所有指标）
  ├── comparison_by_stress.csv      # 按 stress level 的 reward 对比
  ├── reward_bar_chart.png          # 分组柱状图
  ├── metrics_radar.png             # 雷达图（多指标对比）
  ├── reward_boxplot.png            # 箱线图
  └── full_comparison.json          # 机器可读
"""

import os
import sys
import json
import argparse
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# 论文风格设置
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# 调色板（对4个模型区分度好）
MODEL_COLORS = {
    'Baseline':  '#2196F3',
    '+CVAE':     '#4CAF50',
    '+GAN':      '#FF9800',
    '+Diffusion':'#E91E63',
}
STRESS_HATCHES = {
    'low':      '',
    'middle':   '//',
    'high':     'xx',
    'random50': '..',
}


def load_summary(path):
    """加载 summary.json"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_per_episode(path):
    """加载 per_episode_stats.csv"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def scan_root_dir(root_dir):
    """
    自动扫描 root_dir 下的目录结构，返回:
      results[model_name][stress_level] = summary_dict
      per_ep[model_name][stress_level] = [per_episode rows]
    """
    results = {}
    per_ep = {}
    for model_name in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        results[model_name] = {}
        per_ep[model_name] = {}
        for stress in sorted(os.listdir(model_dir)):
            stress_dir = os.path.join(model_dir, stress)
            summary_path = os.path.join(stress_dir, 'summary.json')
            per_ep_path = os.path.join(stress_dir, 'per_episode_stats.csv')
            if os.path.exists(summary_path):
                results[model_name][stress] = load_summary(summary_path)
            if os.path.exists(per_ep_path):
                per_ep[model_name][stress] = load_per_episode(per_ep_path)
    return results, per_ep


# ============================================================
# Table 1: 论文级综合对比表（类似论文 Table 8）
# ============================================================
def generate_comparison_table(results, output_dir):
    """
    生成全模型 × 全指标的对比表。
    对每个模型，将所有 stress level 的结果平均。
    """
    display_metrics = [
        ("Reward [-]",              "total_reward"),
        ("Energy Charged [kWh]",    "total_energy_charged"),
        ("Energy Discharged [kWh]", "total_energy_discharged"),
        ("User Satisfaction [%]",   "average_user_satisfaction"),
        ("Power Violation [kW]",    "power_tracker_violation"),
        ("Costs [€]",             "total_profits"),
        ("Tracking Error",          "tracking_error"),
        ("Exec. Time [sec/step]",   "exec_time_per_step"),
    ]

    model_names = list(results.keys())
    rows = []

    for model in model_names:
        row = {'Model': model}
        stress_summaries = list(results[model].values())
        for display_name, key in display_metrics:
            means = [s[key]['mean'] for s in stress_summaries if key in s]
            stds = [s[key]['std'] for s in stress_summaries if key in s]
            if means:
                overall_mean = np.mean(means)
                overall_std = np.mean(stds)
                row[display_name] = f"{overall_mean:.4f} ± {overall_std:.4f}"
            else:
                row[display_name] = "N/A"

        # optimality gap
        gaps = [s.get('optimality_gap_pct', {}).get('mean', None)
                for s in stress_summaries]
        gaps = [g for g in gaps if g is not None]
        if gaps:
            row['Optimality Gap [%]'] = f"{np.mean(gaps):.2f}"
        else:
            row['Optimality Gap [%]'] = "N/A"

        rows.append(row)

    # 保存
    header = ['Model'] + [d[0] for d in display_metrics] + ['Optimality Gap [%]']
    path = os.path.join(output_dir, 'comparison_table.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison table to {path}")

    # 也打印出来
    print("\n" + "=" * 120)
    print("MODEL COMPARISON TABLE")
    print("=" * 120)
    print(f"{'Model':15s}", end="")
    for d, _ in display_metrics:
        print(f"  {d:>25s}", end="")
    print()
    for row in rows:
        print(f"{row['Model']:15s}", end="")
        for d, _ in display_metrics:
            print(f"  {row.get(d, 'N/A'):>25s}", end="")
        print()
    print("=" * 120)


# ============================================================
# Table 2: 按 stress level 的 reward 对比
# ============================================================
def generate_stress_table(results, output_dir):
    """每个模型在每个 stress level 上的 reward。"""
    model_names = list(results.keys())
    all_stress = sorted(set(s for m in results.values() for s in m.keys()))

    path = os.path.join(output_dir, 'comparison_by_stress.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + all_stress)
        for model in model_names:
            row = [model]
            for stress in all_stress:
                if stress in results[model] and 'total_reward' in results[model][stress]:
                    s = results[model][stress]['total_reward']
                    row.append(f"{s['mean']:.4f} ± {s['std']:.4f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"Saved stress comparison to {path}")


# ============================================================
# Figure 1: 分组柱状图（Reward by Model × Stress Level）
# ============================================================
def plot_reward_bar_chart(results, output_dir):
    model_names = list(results.keys())
    all_stress = sorted(set(s for m in results.values() for s in m.keys()))

    n_models = len(model_names)
    n_stress = len(all_stress)
    x = np.arange(n_stress)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = list(MODEL_COLORS.values())[:n_models]
    if len(colors) < n_models:
        colors += plt.cm.Set2(np.linspace(0, 1, n_models - len(colors))).tolist()

    for i, model in enumerate(model_names):
        means, stds = [], []
        for stress in all_stress:
            if stress in results[model] and 'total_reward' in results[model][stress]:
                means.append(results[model][stress]['total_reward']['mean'])
                stds.append(results[model][stress]['total_reward']['std'])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=model, color=colors[i], capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Stress Level')
    ax.set_ylabel('Total Reward')
    ax.set_title('Model Performance Across Stress Levels')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in all_stress])
    ax.legend(framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    path = os.path.join(output_dir, 'reward_bar_chart.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved bar chart to {path}")


# ============================================================
# Figure 2: 雷达图（多指标对比）
# ============================================================
def plot_radar_chart(results, output_dir):
    """对每个模型取所有 stress level 的平均，画雷达图。"""
    # 选取雷达图的指标（数值越大越好的方向统一）
    radar_metrics = [
        ("Reward",           "total_reward",             1),   # 越大越好
        ("User Satisf.",     "average_user_satisfaction", 1),   # 越大越好
        ("V2G Discharge",    "total_energy_discharged",   1),   # 越大越好
        ("-Power Violation",  "power_tracker_violation",  -1),  # 越小越好，取负
        ("-Cost",            "total_profits",             -1),  # cost 是负值，取负变正
        ("-Tracking Error",   "tracking_error",           -1),  # 越小越好
    ]

    model_names = list(results.keys())
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = list(MODEL_COLORS.values())[:len(model_names)]
    if len(colors) < len(model_names):
        colors += plt.cm.Set2(np.linspace(0, 1, len(model_names) - len(colors))).tolist()

    # 先收集所有值以便归一化
    all_raw = {m: [] for m in model_names}
    for model in model_names:
        stress_summaries = list(results[model].values())
        for _, key, sign in radar_metrics:
            vals = [s[key]['mean'] for s in stress_summaries if key in s]
            all_raw[model].append(sign * np.mean(vals) if vals else 0)

    # Min-max 归一化到 [0, 1]
    all_vals = np.array([all_raw[m] for m in model_names])  # (n_models, n_metrics)
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    for i, model in enumerate(model_names):
        normed = (np.array(all_raw[model]) - mins) / ranges
        vals = normed.tolist() + [normed[0]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in radar_metrics])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    ax.set_title('Multi-Metric Comparison (Normalized)', y=1.08)

    path = os.path.join(output_dir, 'metrics_radar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved radar chart to {path}")


# ============================================================
# Figure 3: 箱线图（Reward 分布）
# ============================================================
def plot_reward_boxplot(per_ep, output_dir):
    """每个模型（合并所有 stress level）的 reward 分布箱线图。"""
    model_names = list(per_ep.keys())
    data = []
    labels = []
    for model in model_names:
        rewards = []
        for stress, rows in per_ep[model].items():
            for r in rows:
                try:
                    rewards.append(float(r.get('total_reward', 0)))
                except (ValueError, TypeError):
                    pass
        data.append(rewards)
        labels.append(model)

    if not any(data):
        print("No per-episode data for boxplot, skipping.")
        return

    colors = list(MODEL_COLORS.values())[:len(model_names)]
    if len(colors) < len(model_names):
        colors += ['#999999'] * (len(model_names) - len(colors))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markersize=6))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Distribution Across All Test Scenarios')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    path = os.path.join(output_dir, 'reward_boxplot.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved boxplot to {path}")


# ============================================================
# Figure 4: Stress Level × Model 热力图
# ============================================================
def plot_heatmap(results, output_dir):
    model_names = list(results.keys())
    all_stress = sorted(set(s for m in results.values() for s in m.keys()))

    matrix = np.zeros((len(model_names), len(all_stress)))
    for i, model in enumerate(model_names):
        for j, stress in enumerate(all_stress):
            if stress in results[model] and 'total_reward' in results[model][stress]:
                matrix[i, j] = results[model][stress]['total_reward']['mean']

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(np.arange(len(all_stress)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([s.replace('_', ' ').title() for s in all_stress])
    ax.set_yticklabels(model_names)

    # 在格子里标数值
    for i in range(len(model_names)):
        for j in range(len(all_stress)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                    fontsize=9, color='black')

    ax.set_title('Reward by Model × Stress Level')
    fig.colorbar(im, ax=ax, label='Total Reward')

    path = os.path.join(output_dir, 'reward_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved heatmap to {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compare multiple model evaluations")
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Root directory with model/stress subdirs containing summary.json')
    parser.add_argument('--eval_dirs', nargs='+', default=None,
                        help='Explicit list of eval result directories')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each eval_dir (model|stress format)')
    parser.add_argument('--output_dir', type=str, default='./comparison_results/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.root_dir:
        results, per_ep = scan_root_dir(args.root_dir)
    elif args.eval_dirs:
        # 从显式目录列表构建
        results = {}
        per_ep = {}
        for i, d in enumerate(args.eval_dirs):
            label = args.labels[i] if args.labels and i < len(args.labels) else f"model_{i}"
            parts = label.split('|')
            model = parts[0] if len(parts) >= 1 else f"model_{i}"
            stress = parts[1] if len(parts) >= 2 else "all"
            if model not in results:
                results[model] = {}
                per_ep[model] = {}
            sp = os.path.join(d, 'summary.json')
            if os.path.exists(sp):
                results[model][stress] = load_summary(sp)
            ep = os.path.join(d, 'per_episode_stats.csv')
            if os.path.exists(ep):
                per_ep[model][stress] = load_per_episode(ep)
    else:
        print("Please provide --root_dir or --eval_dirs")
        sys.exit(1)

    print(f"Loaded results for {len(results)} models:")
    for m, stresses in results.items():
        print(f"  {m}: {list(stresses.keys())}")

    # 生成所有输出
    generate_comparison_table(results, args.output_dir)
    generate_stress_table(results, args.output_dir)
    plot_reward_bar_chart(results, args.output_dir)
    plot_radar_chart(results, args.output_dir)
    if per_ep and any(per_ep.values()):
        plot_reward_boxplot(per_ep, args.output_dir)
    plot_heatmap(results, args.output_dir)

    # 保存完整 JSON
    full_path = os.path.join(args.output_dir, 'full_comparison.json')
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
