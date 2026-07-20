"""Common Voice 26 data loader.

Handles loading and parsing of Common Voice 26 dataset for Danish and English.
Supports both tar.gz archives and extracted directories.
"""

import tarfile
from pathlib import Path

import pandas as pd


def load_common_voice(data_dir: str | Path, language: str) -> list[dict]:
    """Load Common Voice 26 metadata for a given language.

    Args:
        data_dir:
            Path to data directory containing CV26 data. Can contain either
            extracted language folders (e.g., `da/validated.tsv`) or tar.gz
            archives (e.g., `cv26-da.tar.gz`).
        language:
            Language code (e.g., 'da' for Danish, 'en' for English).

    Returns:
        List of dicts with keys: client_id, path, language, duration.

    Raises:
        FileNotFoundError:
            If neither tar.gz nor extracted format is found for the language.
    """
    data_path = Path(data_dir)

    # Try tar.gz format first
    tar_path = data_path / f"cv26-{language}.tar.gz"
    if tar_path.exists():
        return _load_from_tar(tar_path, language)

    # Try extracted format (cv26-{language} naming)
    extracted_path = data_path / f"cv26-{language}" / "validated.tsv"
    if extracted_path.exists():
        return _load_from_extracted(extracted_path, language)

    # Fallback: try simple language code directory
    simple_path = data_path / language / "validated.tsv"
    if simple_path.exists():
        return _load_from_extracted(simple_path, language)

    raise FileNotFoundError(
        f"Common Voice data for '{language}' not found. "
        f"Expected '{tar_path}' or '{extracted_path}'."
    )


def _load_from_tar(tar_path: Path, language: str) -> list[dict]:
    """Load Common Voice metadata from a tar.gz archive.

    Args:
        tar_path:
            Path to the tar.gz archive.
        language:
            Language code for the data.

    Returns:
        List of dicts with metadata.

    Raises:
        ValueError:
            If validated.tsv is not found in the archive.
    """
    records = []

    with tarfile.open(tar_path, "r:gz") as tar:
        # Find validated.tsv within the archive
        # CV26 structure: {language}/validated.tsv
        validated_member = None
        language_prefix = language

        for member in tar.getmembers():
            if member.name.endswith(f"{language_prefix}/validated.tsv"):
                validated_member = member
                break

        if validated_member is None:
            # Fallback: look for any validated.tsv
            for member in tar.getmembers():
                if member.name.endswith("validated.tsv"):
                    validated_member = member
                    break

        if validated_member is None:
            raise ValueError(f"No validated.tsv found in {tar_path}")

        # Extract and read the TSV
        file_obj = tar.extractfile(validated_member)
        if file_obj is None:
            raise ValueError(f"Could not extract {validated_member.name}")

        df = pd.read_csv(file_obj, sep="\t")

        # Try to merge with clip_durations.tsv from the same archive
        for member in tar.getmembers():
            if member.name.endswith(f"{language_prefix}/clip_durations.tsv"):
                durations_file = tar.extractfile(member)
                if durations_file:
                    durations_df = pd.read_csv(durations_file, sep="\t")
                    durations_df = durations_df.rename(columns={"clip": "path"})
                    durations_df["duration_seconds"] = (
                        durations_df["duration[ms]"] / 1000.0
                    )
                    df = df.merge(
                        durations_df[["path", "duration_seconds"]],
                        on="path",
                        how="left",
                    )
                break

        records = _parse_dataframe(df, language)

    return records


def _load_from_extracted(tsv_path: Path, language: str) -> list[dict]:
    """Load Common Voice metadata from an extracted TSV file.

    Also loads clip durations from clip_durations.tsv and merges.

    Args:
        tsv_path:
            Path to the validated.tsv file.
        language:
            Language code for the data.

    Returns:
        List of dicts with metadata including duration.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    # Join with clip_durations.tsv if available
    durations_path = tsv_path.parent / "clip_durations.tsv"
    if durations_path.exists():
        durations_df = pd.read_csv(durations_path, sep="\t")
        # Rename for merging: clip -> path
        durations_df = durations_df.rename(columns={"clip": "path"})
        # Convert duration[ms] to seconds
        durations_df["duration_seconds"] = durations_df["duration[ms]"] / 1000.0
        df = df.merge(durations_df[["path", "duration_seconds"]], on="path", how="left")

    return _parse_dataframe(df, language)


def _parse_dataframe(df: pd.DataFrame, language: str) -> list[dict]:
    """Parse Common Voice DataFrame into list of records.

    Args:
        df:
            DataFrame with Common Voice metadata.
        language:
            Language code to assign to all records.

    Returns:
        List of dicts with client_id, path, language, duration.
    """
    records = []

    for _, row in df.iterrows():
        record = {
            "client_id": str(row.get("client_id", "")),
            "path": str(row.get("path", "")),
            "language": language,
            "duration": float(row.get("duration_seconds", row.get("duration", 0))),
        }
        records.append(record)

    return records
