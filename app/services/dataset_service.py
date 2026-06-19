from __future__ import annotations

import csv
import random
import sys
from functools import lru_cache
from pathlib import Path

from app.schemas import UnstructuredArticleSample

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "training.csv"


class DatasetUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load_dataset_rows(dataset_path: str) -> tuple[tuple[str, str, str], ...]:
    path = Path(dataset_path)
    if not path.exists():
        raise DatasetUnavailableError(f"Dataset not found: {path}")

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(
            (
                (row.get("title") or "").strip(),
                (row.get("text") or "").strip(),
                (row.get("label") or "").strip(),
            )
            for row in reader
            if (row.get("title") or "").strip()
            and (row.get("text") or "").strip()
            and (row.get("label") or "").strip()
        )

    if not rows:
        raise DatasetUnavailableError(f"Dataset is empty or missing required columns: {path}")

    return rows


def get_random_unstructured_sample(
    dataset_path: Path | None = None,
) -> UnstructuredArticleSample:
    path = dataset_path or DEFAULT_DATASET_PATH
    title, text, label = random.choice(_load_dataset_rows(str(path.resolve())))
    return UnstructuredArticleSample(title=title, text=text, label=label)
