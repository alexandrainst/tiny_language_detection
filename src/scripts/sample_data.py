"""Sample Common Voice data for training and testing.

Creates balanced train/test splits with speaker-independent separation.
Samples data to target hours per language and assigns duration groups.

Usage:
    uv run src/scripts/sample_data.py --target-hours 1.0 --seed 42
"""

import argparse
import logging
import random
import warnings
from pathlib import Path

import pandas as pd

from tiny_language_detection.data.common_voice import load_common_voice
from tiny_language_detection.data.preprocessing import assign_duration_group

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sample Common Voice data for training and testing."
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=1.0,
        help="Target hours of audio per language (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sampled",
        help="Path to output directory for sampled data (default: data/sampled)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for sampling script.

    Raises:
        FileNotFoundError:
            If Danish data is not found.
    """
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Danish data (required)
    logger.info("Loading Danish Common Voice data...")
    try:
        da_records = load_common_voice(data_dir, "da")
        logger.info(f"Loaded {len(da_records)} Danish records")
    except FileNotFoundError as e:
        logger.error(f"Danish data required but not found: {e}")
        raise

    # Load English data (optional)
    en_records = []
    try:
        logger.info("Loading English Common Voice data...")
        en_records = load_common_voice(data_dir, "en")
        logger.info(f"Loaded {len(en_records)} English records")
    except FileNotFoundError:
        warnings.warn(
            "English Common Voice data not found. Proceeding with Danish data only.",
            UserWarning,
            stacklevel=2,
        )

    # Combine all records
    all_records = da_records + en_records
    logger.info(f"Total records: {len(all_records)}")

    # Filter to only include clips that actually exist on disk
    logger.info("Filtering to clips that exist on disk...")
    filtered_records = []
    for rec in all_records:
        lang = rec["language"]
        audio_dir = data_dir / f"cv26-{lang}"
        if lang == "en":
            audio_dir = audio_dir / "clips"
        clip_path = audio_dir / rec["path"]
        if clip_path.exists():
            filtered_records.append(rec)
        else:
            # Clip not extracted yet (e.g., partial download)
            pass

    logger.info(
        f"Clips available on disk: {len(filtered_records)} "
        f"({len(all_records) - len(filtered_records)} clips not found)"
    )

    # Create DataFrame
    df = pd.DataFrame(filtered_records)

    # Speaker-independent split: group by speaker, split speakers 80/20
    train_df, test_df = speaker_independent_split(df, train_ratio=0.8, seed=args.seed)

    logger.info(f"Train speakers: {train_df['client_id'].nunique()}")
    logger.info(f"Test speakers: {test_df['client_id'].nunique()}")

    # Sample to target hours per language for each split
    target_seconds = args.target_hours * 3600

    train_sampled = sample_to_target_hours(train_df, target_seconds, args.seed)
    test_sampled = sample_to_target_hours(test_df, target_seconds, args.seed)

    # Add label column (0 for Danish, 1 for English)
    train_sampled = train_sampled.copy()
    test_sampled = test_sampled.copy()
    train_sampled["label"] = train_sampled["language"].apply(
        lambda x: 0 if x == "da" else 1
    )
    test_sampled["label"] = test_sampled["language"].apply(
        lambda x: 0 if x == "da" else 1
    )

    # Add duration_group column
    train_sampled["duration_group"] = train_sampled["duration"].apply(
        assign_duration_group
    )
    test_sampled["duration_group"] = test_sampled["duration"].apply(
        assign_duration_group
    )

    # Add split column
    train_sampled["split"] = "train"
    test_sampled["split"] = "test"

    # Reorder columns: path, language, label, speaker_id, duration,
    # duration_group, split

    # Rename client_id to speaker_id for output
    train_sampled = train_sampled.rename(columns={"client_id": "speaker_id"})
    test_sampled = test_sampled.rename(columns={"client_id": "speaker_id"})

    final_column_order = [
        "path",
        "language",
        "label",
        "speaker_id",
        "duration",
        "duration_group",
        "split",
    ]

    train_sampled = train_sampled[final_column_order]
    test_sampled = test_sampled[final_column_order]

    # Write to CSV
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_sampled.to_csv(train_path, index=False)
    test_sampled.to_csv(test_path, index=False)

    # Log summary
    logger.info(f"Train samples written to: {train_path}")
    logger.info(f"Test samples written to: {test_path}")
    logger.info(
        f"Train total duration: {train_sampled['duration'].sum() / 3600:.2f} hours"
    )
    logger.info(
        f"Test total duration: {test_sampled['duration'].sum() / 3600:.2f} hours"
    )

    # Log duration group distribution
    logger.info("\nTrain duration group distribution:")
    logger.info(train_sampled["duration_group"].value_counts().to_string())
    logger.info("\nTest duration group distribution:")
    logger.info(test_sampled["duration_group"].value_counts().to_string())


def speaker_independent_split(
    df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data ensuring no speaker appears in both train and test.

    Args:
        df:
            DataFrame with client_id column.
        train_ratio (optional):
            Ratio of speakers to assign to training set. Defaults to 0.8.
        seed (optional):
            Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Get unique speakers
    speakers = df["client_id"].unique()

    # Shuffle speakers
    random.seed(seed)
    shuffled_speakers = list(speakers)
    random.shuffle(shuffled_speakers)

    # Split speakers
    n_train_speakers = int(len(shuffled_speakers) * train_ratio)
    train_speakers = set(shuffled_speakers[:n_train_speakers])
    test_speakers = set(shuffled_speakers[n_train_speakers:])

    # Split DataFrame
    train_df = df[df["client_id"].isin(train_speakers)].copy()
    test_df = df[df["client_id"].isin(test_speakers)].copy()

    return train_df, test_df


def sample_to_target_hours(
    df: pd.DataFrame, target_seconds: float, seed: int = 42
) -> pd.DataFrame:
    """Sample data to reach target hours per language.

    Strategically samples to balance duration groups while reaching target hours.

    Args:
        df:
            DataFrame with language and duration columns.
        target_seconds:
            Target total seconds per language.
        seed (optional):
            Random seed for reproducibility. Defaults to 42.

    Returns:
        Sampled DataFrame.
    """
    if df.empty:
        return df

    # Group by language
    languages = df["language"].unique()
    sampled_dfs = []

    random.seed(seed)

    for lang in languages:
        lang_df = df[df["language"] == lang].copy()

        # Group by duration_group for stratified sampling
        lang_df["duration_group"] = lang_df["duration"].apply(assign_duration_group)

        # Calculate current total duration per language
        current_duration = lang_df["duration"].sum()

        if current_duration <= target_seconds:
            # Use all data if under target
            sampled_dfs.append(lang_df)
            continue

        # Stratified sampling: try to maintain duration group distribution
        duration_groups = ["0-2s", "2-4s", "4-6s", "6+s"]
        group_samples = []

        # Calculate group weights based on current distribution
        group_durations = {}
        for group in duration_groups:
            group_data = lang_df[lang_df["duration_group"] == group]
            group_durations[group] = group_data["duration"].sum()

        total_duration = sum(group_durations.values())
        if total_duration == 0:
            continue

        # Sample proportionally from each group
        for group in duration_groups:
            group_data = lang_df[lang_df["duration_group"] == group]
            if group_data.empty:
                continue

            # Calculate target seconds for this group
            group_weight = group_durations[group] / total_duration
            group_target = target_seconds * group_weight

            # Shuffle and accumulate until we reach target
            group_data = group_data.sample(frac=1, random_state=seed).reset_index(
                drop=True
            )

            accumulated = 0.0
            indices = []
            for idx, row in group_data.iterrows():
                accumulated += row["duration"]
                indices.append(idx)
                if accumulated >= group_target:
                    break

            group_samples.append(group_data.loc[indices])

        if group_samples:
            lang_sampled = pd.concat(group_samples, ignore_index=True)
            sampled_dfs.append(lang_sampled)

    if sampled_dfs:
        result = pd.concat(sampled_dfs, ignore_index=True)
    else:
        result = pd.DataFrame()

    return result


if __name__ == "__main__":
    main()
