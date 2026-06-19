from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "article_template_classifier.joblib"


class ModelNotTrainedError(RuntimeError):
    pass


class ArticleTemplateClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._artifact: dict | None = None

    def _load(self) -> dict:
        if self._artifact is None:
            if not self.model_path.exists():
                raise ModelNotTrainedError("Model not trained. Run train_classifier.py first.")
            self._artifact = joblib.load(self.model_path)
        return self._artifact

    def predict_template(self, title: str, text: str) -> dict:
        artifact = self._load()
        vectorizer = artifact["vectorizer"]
        classifier = artifact["classifier"]
        features = vectorizer.transform([f"{title}\n\n{text}"])

        probabilities = classifier.predict_proba(features)[0]
        classes = classifier.classes_
        best_index = int(np.argmax(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:5]

        top_predictions = [
            {
                "label": str(classes[index]),
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]

        return {
            "label": str(classes[best_index]),
            "confidence": float(probabilities[best_index]),
            "top_predictions": top_predictions,
        }


_runtime_classifier = ArticleTemplateClassifier()


def predict_template(title: str, text: str) -> dict:
    return _runtime_classifier.predict_template(title, text)
