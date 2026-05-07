import argparse
from pathlib import Path
import pandas as pd


def merge_by_day_id(dataset_path: str, stress_path: str, output_path: str) -> None:
    dataset_path = Path(dataset_path)
    stress_path = Path(stress_path)
    output_path = Path(output_path)

    df_main = pd.read_csv(dataset_path)
    df_stress = pd.read_csv(stress_path)

    # Basic checks
    required_main = {'day_id'}
    required_stress = {'day_id', 'stress_score'}

    if not required_main.issubset(df_main.columns):
        missing = required_main - set(df_main.columns)
        raise ValueError(f"Main dataset is missing required columns: {missing}")

    if not required_stress.issubset(df_stress.columns):
        missing = required_stress - set(df_stress.columns)
        raise ValueError(f"Stress dataset is missing required columns: {missing}")

    # Ensure one stress score per day_id
    duplicated_day_ids = df_stress[df_stress.duplicated(subset=['day_id'], keep=False)]
    if not duplicated_day_ids.empty:
        sample_ids = duplicated_day_ids['day_id'].drop_duplicates().tolist()[:10]
        raise ValueError(
            "daily_stress_scores.csv contains duplicate day_id values. "
            f"Example duplicated day_id(s): {sample_ids}"
        )

    # Merge: copy stress_score to every row with the same day_id
    merged = df_main.merge(
        df_stress[['day_id', 'stress_score']],
        on='day_id',
        how='left',
        validate='many_to_one'
    )

    # Report unmatched day_id if any
    missing_count = merged['stress_score'].isna().sum()
    if missing_count > 0:
        missing_day_ids = sorted(set(merged.loc[merged['stress_score'].isna(), 'day_id'].tolist()))[:20]
        print(
            f"Warning: {missing_count} row(s) could not match a stress_score. "
            f"Example missing day_id(s): {missing_day_ids}"
        )
    else:
        print("All rows matched successfully by day_id.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Merged file saved to: {output_path}")
    print(f"Merged shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge stress_score into dataset_China_all.csv by day_id.'
    )
    parser.add_argument(
        '--dataset',
        default='dataset_China_all.csv',
        help='Path to the main dataset CSV file.'
    )
    parser.add_argument(
        '--stress',
        default='daily_stress_scores.csv',
        help='Path to the daily stress score CSV file.'
    )
    parser.add_argument(
        '--output',
        default='dataset_China_all_with_stress.csv',
        help='Path to save the merged CSV file.'
    )

    args = parser.parse_args()
    merge_by_day_id(args.dataset, args.stress, args.output)
