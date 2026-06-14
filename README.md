# NLP Article Structuring

Prototype backend for a master's thesis on structuring plain article text into Wikipedia-style article components using NLP methods.

## Features

- `POST /api/structure` FastAPI endpoint
- supervised article template classification from Wikipedia-derived labels
- section splitting from plain text
- spaCy NER extraction
- template-aware infobox field extraction
- simple relation extraction
- optional Wikipedia-style wikitext generation
- SQLite persistence for requests and outputs

## Project structure

```text
app/
  main.py
  schemas.py
  api/
    routes.py
  services/
    structure_service.py
    section_service.py
    infobox_service.py
    ner_service.py
    relation_service.py
    wikitext_service.py
  ml/
    classifier.py
    train_classifier.py
    dataset_builder.py
    evaluate_classifier.py
  storage/
    database.py
models/
data/
outputs/
requirements.txt
README.md
```

## Install

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Build training data

Generate labeled training rows from a Wikipedia XML export or dump. Only pages with an infobox template are kept.

```powershell
python app/ml/dataset_builder.py --xml data/sample.xml --out data/training.csv --limit 5000
```

Output columns:

```text
title,text,label
```

## Train the classifier

```powershell
python app/ml/train_classifier.py --data data/training.csv
```

The script trains a TF-IDF + Logistic Regression classifier and saves:

```text
models/article_template_classifier.joblib
```

## Evaluate the classifier

```powershell
python app/ml/evaluate_classifier.py --data data/training.csv
```

Metrics are saved to:

```text
outputs/classification_metrics.json
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API request

```json
{
  "title": "Ada Lovelace",
  "text": "Ada Lovelace was an English mathematician and writer...",
  "options": {
    "generateWikitext": true,
    "template": "Infobox person",
    "language": "en",
    "returnDebug": false,
    "extra": {}
  }
}
```

## Notes

- The classifier is required at runtime. If the model file is missing, the API returns: `"Model not trained. Run train_classifier.py first."`
- The first local NER setup should use `en_core_web_sm`.
- The current prototype starts with scikit-learn so it remains lightweight and runnable locally.
