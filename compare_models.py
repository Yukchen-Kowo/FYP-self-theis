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
  └── gan/
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
  └── gan/
      ...
  

输出：
  output_dir/
  ├── comparison_table.csv          # 论文 Table 格式（所有模型 × 所有指标）
  ├── comparison_by_stress.csv      # 按 stress level 的 reward 对比
  ├── reward_bar_chart.png          # 分组柱状图
  ├── metrics_radar.png             # 雷达图（多指标对比）
  ├── reward_boxplot.png            # 箱线图
  ├── reward_heatmap.png            # 热力图
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
    'Baseline':   '#2196F3',
    '+CVAE':      '#4CAF50',
    '+GAN':       '#FF9800',
    '+Diffusion': '#E91E63',
}

STRESS_HATCHES = {
    'low': '',
    'middle': '//',
    'high': 'xx',
    'random50': '..',
}

# 固定 stress 顺序
STRESS_ORDER = {
    'low': 0,
    'middle': 1,
    'high': 2,
    'random50': 3,
}

# 常见模型名别名，方便 root_dir 模式下自动显示得更规范
MODEL_NAME_ALIASES = {
    'baseline': 'Baseline',
    'baseline300': 'Baseline',
    'baseline400': 'Baseline',
    'baseline450': 'Baseline',
    'baseline500': 'Baseline',
    'cvae': '+CVAE',
    'cvae_model': '+CVAE',
    'gan': '+GAN',
    'gan_model': '+GAN',
    'df_model': '+Diffusion',
    'diffusion': '+Diffusion',
    'diffusion_model': '+Diffusion',
}

MODEL_ORDER = {
    'Baseline': 0,
    '+CVAE': 1,
    '+GAN': 2,
    '+Diffusion': 3,
}


def normalize_stress_name(stress_name):
    """
    将 stress 名称归一化：
    - remaining50 -> random50
    - Random50 / random50 / remaining50 统一为 random50
    """
    if stress_name is None:
        return 'all'

    s = str(stress_name).strip()
    s_lower = s.lower().replace(' ', '').replace('_', '')

    if s_lower in ('remaining50', 'random50'):
        return 'random50'
    if s_lower == 'low':
        return 'low'
    if s_lower == 'middle':
        return 'middle'
    if s_lower == 'high':
        return 'high'

    return str(stress_name).strip().lower()


def display_stress_name(stress_name):
    """
    图上/表头显示用名称
    """
    s = normalize_stress_name(stress_name)
    if s == 'random50':
        return 'Random50'
    if s in ('low', 'middle', 'high'):
        return s
    return s.replace('_', ' ').title()


def sort_stress_levels(stresses):
    """
    固定顺序：low -> middle -> high -> random50
    其他未知 stress 放在后面，按字母排序
    """
    stresses = list(stresses)
    return sorted(
        stresses,
        key=lambda x: (STRESS_ORDER.get(normalize_stress_name(x), 99), normalize_stress_name(x))
    )


def normalize_model_name(model_name):
    """
    将常见模型目录名归一化成更适合展示的名字
    """
    if model_name is None:
        return 'UnknownModel'
    s = str(model_name).strip()
    return MODEL_NAME_ALIASES.get(s.lower(), s)


def sort_model_names(model_names):
    """
    固定模型顺序：Baseline -> +CVAE -> +GAN -> +Diffusion
    其他模型放后面
    """
    model_names = list(model_names)
    return sorted(model_names, key=lambda x: (MODEL_ORDER.get(x, 99), x.lower()))


def get_model_colors(model_names):
    """
    按模型顺序返回颜色。已知模型用预设色，未知模型自动补色。
    """
    colors = []
    fallback_needed = []

    for i, m in enumerate(model_names):
        if m in MODEL_COLORS:
            colors.append(MODEL_COLORS[m])
        else:
            colors.append(None)
            fallback_needed.append(i)

    if fallback_needed:
        fallback_colors = plt.cm.Set2(np.linspace(0, 1, len(fallback_needed))).tolist()
        for idx, c in zip(fallback_needed, fallback_colors):
            colors[idx] = c

    return colors


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

    for raw_model_name in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, raw_model_name)
        if not os.path.isdir(model_dir):
            continue

        model_name = normalize_model_name(raw_model_name)

        if model_name not in results:
            results[model_name] = {}
            per_ep[model_name] = {}

        for raw_stress in sorted(os.listdir(model_dir)):
            stress_dir = os.path.join(model_dir, raw_stress)
            if not os.path.isdir(stress_dir):
                continue

            stress = normalize_stress_name(raw_stress)

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
        ("Costs [€]",               "total_profits"),
        ("Tracking Error",          "tracking_error"),
        ("Exec. Time [sec/step]",   "exec_time_per_step"),
    ]

    model_names = sort_model_names(results.keys())
    rows = []

    for model in model_names:
        row = {'Model': model}
        stress_summaries = [results[model][s] for s in sort_stress_levels(results[model].keys())]

        for display_name, key in display_metrics:
            means = [s[key]['mean'] for s in stress_summaries if key in s]
            stds = [s[key]['std'] for s in stress_summaries if key in s]

            if means:
                overall_mean = np.mean(means)
                overall_std = np.mean(stds) if stds else 0.0
                row[display_name] = f"{overall_mean:.4f} ± {overall_std:.4f}"
            else:
                row[display_name] = "N/A"

        # optimality gap
        gaps = [s.get('optimality_gap_pct', {}).get('mean', None) for s in stress_summaries]
        gaps = [g for g in gaps if g is not None]

        if gaps:
            row['Optimality Gap [%]'] = f"{np.mean(gaps):.2f}"
        else:
            row['Optimality Gap [%]'] = "N/A"

        rows.append(row)

    header = ['Model'] + [d[0] for d in display_metrics] + ['Optimality Gap [%]']
    path = os.path.join(output_dir, 'comparison_table.csv')

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison table to {path}")

    # 控制台打印
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
    model_names = sort_model_names(results.keys())
    all_stress = sort_stress_levels(set(s for m in results.values() for s in m.keys()))

    path = os.path.join(output_dir, 'comparison_by_stress.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + [display_stress_name(s) for s in all_stress])

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
    model_names = sort_model_names(results.keys())
    all_stress = sort_stress_levels(set(s for m in results.values() for s in m.keys()))

    n_models = len(model_names)
    n_stress = len(all_stress)

    if n_models == 0 or n_stress == 0:
        print("No data for reward bar chart, skipping.")
        return

    x = np.arange(n_stress)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_model_colors(model_names)

    for i, model in enumerate(model_names):
        means, stds = [], []
        for stress in all_stress:
            if stress in results[model] and 'total_reward' in results[model][stress]:
                means.append(results[model][stress]['total_reward']['mean'])
                stds.append(results[model][stress]['total_reward']['std'])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=model,
            color=colors[i],
            capsize=3,
            edgecolor='white',
            linewidth=0.5
        )

    ax.set_xlabel('Stress Level')
    ax.set_ylabel('Total Reward')
    ax.set_title('Model Performance Across Stress Levels')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([display_stress_name(s) for s in all_stress])
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
    radar_metrics = [
        ("Reward",            "total_reward",               1),
        ("User Satisf.",      "average_user_satisfaction",  1),
        ("V2G Discharge",     "total_energy_discharged",    1),
        ("-Power Violation",  "power_tracker_violation",   -1),
        ("-Cost",             "total_profits",             -1),
        ("-Tracking Error",   "tracking_error",            -1),
    ]

    model_names = sort_model_names(results.keys())
    if not model_names:
        print("No data for radar chart, skipping.")
        return

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = get_model_colors(model_names)

    # 先收集所有值以便归一化
    all_raw = {m: [] for m in model_names}

    for model in model_names:
        stress_ordered = sort_stress_levels(results[model].keys())
        stress_summaries = [results[model][s] for s in stress_ordered]

        for _, key, sign in radar_metrics:
            vals = [s[key]['mean'] for s in stress_summaries if key in s]
            all_raw[model].append(sign * np.mean(vals) if vals else 0)

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
    model_names = sort_model_names(per_ep.keys())
    data = []
    labels = []

    for model in model_names:
        rewards = []
        for stress in sort_stress_levels(per_ep[model].keys()):
            rows = per_ep[model][stress]
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

    colors = get_model_colors(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markersize=6)
    )

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
    model_names = sort_model_names(results.keys())
    all_stress = sort_stress_levels(set(s for m in results.values() for s in m.keys()))

    if len(model_names) == 0 or len(all_stress) == 0:
        print("No data for heatmap, skipping.")
        return

    matrix = np.zeros((len(model_names), len(all_stress)))

    for i, model in enumerate(model_names):
        for j, stress in enumerate(all_stress):
            if stress in results[model] and 'total_reward' in results[model][stress]:
                matrix[i, j] = results[model][stress]['total_reward']['mean']

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(np.arange(len(all_stress)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([display_stress_name(s) for s in all_stress])
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(all_stress)):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha='center', va='center',
                fontsize=9, color='black'
            )

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
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help='Root directory with model/stress subdirs containing summary.json'
    )
    parser.add_argument(
        '--eval_dirs',
        nargs='+',
        default=None,
        help='Explicit list of eval result directories'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Labels for each eval_dir (model|stress format)'
    )
    parser.add_argument('--output_dir', type=str, default='./comparison_results/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.root_dir:
        results, per_ep = scan_root_dir(args.root_dir)

    elif args.eval_dirs:
        results = {}
        per_ep = {}

        for i, d in enumerate(args.eval_dirs):
            label = args.labels[i] if args.labels and i < len(args.labels) else f"model_{i}|all"
            parts = label.split('|')

            raw_model = parts[0] if len(parts) >= 1 else f"model_{i}"
            raw_stress = parts[1] if len(parts) >= 2 else "all"

            model = normalize_model_name(raw_model)
            stress = normalize_stress_name(raw_stress)

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
    for m in sort_model_names(results.keys()):
        stress_list = sort_stress_levels(results[m].keys())
        stress_display = [display_stress_name(s) for s in stress_list]
        print(f"  {m}: {stress_display}")

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
