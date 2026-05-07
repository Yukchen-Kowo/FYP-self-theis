
import argparse
from pathlib import Path
import pandas as pd
import json


def get_rank_slice(df_day, start_pct, end_pct):
    """
    Select days by rank after sorting by stress_score ascending, then day_id ascending.
    Range semantics:
      [start_pct, end_pct) for non-terminal ranges
      [start_pct, 1.0] for the final range where end_pct == 1.0
    """
    n = len(df_day)
    start_idx = int(n * start_pct)
    end_idx = n if end_pct >= 1.0 else int(n * end_pct)
    return df_day.iloc[start_idx:end_idx].copy()


def sample_day_ids(df_days, n_sample, seed):
    return (
        df_days.sample(n=n_sample, random_state=seed)["day_id"]
        .sort_values()
        .tolist()
    )


def build_subset(df_full, day_ids):
    # Keep all 96 steps for selected days; preserve original full-data order.
    return df_full[df_full["day_id"].isin(day_ids)].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset_China_all_with_stress.csv into four test sets and one remaining set."
    )
    parser.add_argument(
        "--input",
        default="dataset_China_all_with_stress.csv",
        help="Input CSV file containing at least day_id and stress_score columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="stress_splits",
        help="Directory to save the split CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible day sampling.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = {"day_id", "stress_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Basic integrity checks
    steps_per_day = df.groupby("day_id").size()
    if steps_per_day.nunique() != 1:
        raise ValueError(
            "Not all days have the same number of rows. "
            f"Observed row counts per day: {steps_per_day.value_counts().to_dict()}"
        )

    stress_per_day = df.groupby("day_id")["stress_score"].nunique()
    if stress_per_day.max() != 1:
        bad_days = stress_per_day[stress_per_day > 1].index.tolist()[:10]
        raise ValueError(
            "Some day_id values have more than one stress_score. "
            f"Examples: {bad_days}"
        )

    # Day-level table, sorted by stress_score then day_id to make ranking deterministic.
    df_day = (
        df.groupby("day_id", as_index=False)["stress_score"]
        .first()
        .sort_values(["stress_score", "day_id"], ascending=[True, True])
        .reset_index(drop=True)
    )

    n_days = len(df_day)
    if n_days != 731:
        print(f"[Warning] Expected 731 days based on your description, but found {n_days} days.")

    # Rank-based pools:
    # low    = 0%-20%
    # middle = 40%-60%
    # high   = 80%-100%
    low_pool = get_rank_slice(df_day, 0.00, 0.20)
    middle_pool = get_rank_slice(df_day, 0.40, 0.60)
    high_pool = get_rank_slice(df_day, 0.80, 1.00)

    for name, pool in [("low", low_pool), ("middle", middle_pool), ("high", high_pool)]:
        if len(pool) < 50:
            raise ValueError(f"{name} pool has only {len(pool)} days, fewer than 50.")

    # Sample 50 days from each pool
    low_days = sample_day_ids(low_pool, 50, args.seed + 11)
    middle_days = sample_day_ids(middle_pool, 50, args.seed + 22)
    high_days = sample_day_ids(high_pool, 50, args.seed + 33)

    selected_first_150 = sorted(set(low_days) | set(middle_days) | set(high_days))
    if len(selected_first_150) != 150:
        raise ValueError(
            f"Expected 150 distinct days from low/middle/high, but got {len(selected_first_150)}."
        )

    remaining_after_150 = sorted(set(df_day["day_id"]) - set(selected_first_150))
    if len(remaining_after_150) != n_days - 150:
        raise ValueError("Remaining day count after first 150-day selection is inconsistent.")

    remaining_after_150_df = df_day[df_day["day_id"].isin(remaining_after_150)].copy()
    random_50_days = (
        remaining_after_150_df.sample(n=50, random_state=args.seed + 44)["day_id"]
        .sort_values()
        .tolist()
    )

    final_remaining_days = sorted(set(remaining_after_150) - set(random_50_days))
    if len(final_remaining_days) != n_days - 200:
        raise ValueError("Final remaining day count is inconsistent after selecting random 50 days.")

    # Build row-level datasets
    low_stress_df = build_subset(df, low_days)
    middle_stress_df = build_subset(df, middle_days)
    high_stress_df = build_subset(df, high_days)
    random_remaining_50_df = build_subset(df, random_50_days)
    remaining_531_df = build_subset(df, final_remaining_days)

    # Save CSVs
    low_path = output_dir / "low_stress.csv"
    middle_path = output_dir / "middle_stress.csv"
    high_path = output_dir / "high_stress.csv"
    random50_path = output_dir / "random_50_from_remaining.csv"
    remaining531_path = output_dir / "remaining_531_days.csv"

    low_stress_df.to_csv(low_path, index=False)
    middle_stress_df.to_csv(middle_path, index=False)
    high_stress_df.to_csv(high_path, index=False)
    random_remaining_50_df.to_csv(random50_path, index=False)
    remaining_531_df.to_csv(remaining531_path, index=False)

    # Save metadata for inspection / reproducibility
    summary = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "total_days": int(n_days),
        "rows_per_day": int(steps_per_day.iloc[0]),
        "low_pool_days": int(len(low_pool)),
        "middle_pool_days": int(len(middle_pool)),
        "high_pool_days": int(len(high_pool)),
        "low_stress_selected_days": low_days,
        "middle_stress_selected_days": middle_days,
        "high_stress_selected_days": high_days,
        "random_50_selected_days": random_50_days,
        "remaining_531_days": final_remaining_days,
        "files": {
            "low_stress": str(low_path),
            "middle_stress": str(middle_path),
            "high_stress": str(high_path),
            "random_50_from_remaining": str(random50_path),
            "remaining_531_days": str(remaining531_path),
        },
        "row_counts": {
            "low_stress": int(len(low_stress_df)),
            "middle_stress": int(len(middle_stress_df)),
            "high_stress": int(len(high_stress_df)),
            "random_50_from_remaining": int(len(random_remaining_50_df)),
            "remaining_531_days": int(len(remaining_531_df)),
        },
    }

    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also save selected day lists as CSV for convenience
    pd.DataFrame({"day_id": low_days}).to_csv(output_dir / "low_stress_day_ids.csv", index=False)
    pd.DataFrame({"day_id": middle_days}).to_csv(output_dir / "middle_stress_day_ids.csv", index=False)
    pd.DataFrame({"day_id": high_days}).to_csv(output_dir / "high_stress_day_ids.csv", index=False)
    pd.DataFrame({"day_id": random_50_days}).to_csv(output_dir / "random_50_day_ids.csv", index=False)
    pd.DataFrame({"day_id": final_remaining_days}).to_csv(output_dir / "remaining_531_day_ids.csv", index=False)

    print("Done.")
    print(json.dumps(summary["row_counts"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
