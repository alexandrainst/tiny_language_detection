"""Experiment result tracking with JSONL serialization."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentResult:
    """Results from a single model experiment.

    Attributes:
        model_id: Unique identifier (e.g., "phase4b-small-kd")
        model_name: Human-readable name (e.g., "Phase 4b Small (KD)")
        architecture: Model architecture (e.g., "CompactCNN", "CNN-RNN")
        params: Number of parameters
        ram_kb: Runtime RAM usage in KB (weights + activations + buffers)
        storage_kb: Storage size in KB (may differ from RAM if quantised)
        accuracy: Overall accuracy (0-100)
        accuracy_da: Danish accuracy (0-100, optional)
        accuracy_en: English accuracy (0-100, optional)
        dataset: Dataset name (e.g., "CommonVoice26", "YODAS-Granary")
        training_mode: Training approach (e.g., "Direct", "Knowledge Distillation")
        precision: Weight precision (e.g., "FP32", "BF16", "INT8")
        epochs: Number of training epochs
        notes: Additional notes or observations
    """

    model_id: str
    model_name: str
    architecture: str
    params: int
    ram_kb: float
    storage_kb: float
    accuracy: float
    dataset: str = "YODAS-Granary"
    accuracy_da: Optional[float] = None
    accuracy_en: Optional[float] = None
    training_mode: str = "Direct"
    precision: str = "FP32"
    epochs: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Create from dictionary.

        Args:
            data: Dictionary with ExperimentResult fields

        Returns:
            ExperimentResult instance
        """
        return cls(**data)


def save_result(result: ExperimentResult, filepath: Path) -> None:
    """Append a result to a JSONL file.

    Args:
        result: Experiment result to save
        filepath: Path to JSONL file (created if doesn't exist)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def load_results(filepath: Path) -> List[ExperimentResult]:
    """Load all results from a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of ExperimentResult objects, sorted by accuracy descending

    Raises:
        FileNotFoundError: If filepath doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results.append(ExperimentResult.from_dict(data))

    return sorted(results, key=lambda r: r.accuracy, reverse=True)
