from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "article_template_classifier.joblib"


def prepare_dataframe(data_path: Path, min_samples: int) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.dropna(subset=["title", "text", "label"]).copy()
    dataframe["title"] = dataframe["title"].astype(str)
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(str)
    label_counts = dataframe["label"].value_counts()
    keep_labels = label_counts[label_counts >= min_samples].index
    filtered = dataframe[dataframe["label"].isin(keep_labels)].copy()
    if filtered.empty:
        raise ValueError("No training rows remain after label frequency filtering.")
    filtered["combined_text"] = filtered["title"] + "\n\n" + filtered["text"]
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the article template classifier.")
    parser.add_argument("--data", required=True, help="CSV dataset with title,text,label columns.")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH), help="Output joblib path.")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum examples per label.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = prepare_dataframe(Path(args.data), args.min_samples)

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["combined_text"],
        dataframe["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataframe["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=2,
        sublinear_tf=True,
    )
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    classifier = LogisticRegression(
        max_iter=2_000,
    )
    classifier.fit(x_train_features, y_train)

    predictions = classifier.predict(x_test_features)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "metadata": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "min_samples": args.min_samples,
            "labels": sorted(dataframe["label"].unique().tolist()),
            "accuracy": accuracy,
        },
    }
    joblib.dump(artifact, model_out)

    print(f"Saved model to {model_out}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)


if __name__ == "__main__":
    main()
