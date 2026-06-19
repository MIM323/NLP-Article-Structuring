from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "article_template_classifier.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "outputs" / "classification_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved article template classifier.")
    parser.add_argument("--data", required=True, help="CSV dataset with title,text,label columns.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Saved joblib model path.")
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS_PATH), help="Metrics JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(Path(args.model))
    metadata = artifact["metadata"]

    dataframe = pd.read_csv(args.data).dropna(subset=["title", "text", "label"]).copy()
    dataframe["label"] = dataframe["label"].astype(str)
    keep_labels = set(metadata["labels"])
    dataframe = dataframe[dataframe["label"].isin(keep_labels)].copy()
    dataframe["combined_text"] = dataframe["title"].astype(str) + "\n\n" + dataframe["text"].astype(str)

    _, x_test, _, y_test = train_test_split(
        dataframe["combined_text"],
        dataframe["label"],
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        stratify=dataframe["label"],
    )

    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    predictions = classifier.predict(vectorizer.transform(x_test))

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "label_count": len(keep_labels),
        "test_size": metadata["test_size"],
        "random_state": metadata["random_state"],
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
