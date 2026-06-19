# NLP Article Structuring

Prototype backend for a master's thesis on structuring plain article text into Wikipedia-style article components using NLP methods.

## Thesis pipeline

The article structuring workflow in this project follows these stages:

`article input -> text preprocessing -> paragraph reconstruction and section assignment -> article/template classification -> named entity recognition -> information extraction -> schema/field mapping -> structured data combination -> optional wikitext generation`

In practical terms, this means the system does not only classify raw text. It reconstructs readable paragraph blocks from plain text, assigns Wikipedia-style section headings heuristically, maps extracted facts into template-aware fields, and combines the results into a structured output.

## Features

- `POST /api/structure` FastAPI endpoint
- `GET /api/unstructured/random` endpoint for fetching a random dataset sample
- supervised article template classification from Wikipedia-derived labels
- paragraph reconstruction and section assignment from plain text
- spaCy NER extraction
- template-aware infobox field extraction and schema/field mapping
- simple relation extraction
- structured output assembly for Wikipedia-style components
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
    dataset_service.py
    infobox_service.py
    ner_service.py
    relation_service.py
    wikitext_service.py
  ml/
    classifier.py
    train_classifier.py
    dataset_builder.py
    evaluate_classifier.py
    filter_people_dump.py
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

If you want to start with people only, first create a smaller XML subset:

```powershell
python app/ml/filter_people_dump.py --xml data/enwiki.xml --out data/people_only.xml
```

You can also cap the output while testing:

```powershell
python app/ml/filter_people_dump.py --xml data/enwiki.xml --out data/people_only_sample.xml --limit 20000
```

Then build the CSV from the filtered XML:

```powershell
python app/ml/dataset_builder.py --xml data/people_only.xml --out data/training.csv --limit 5000
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

## API response highlights

The structure response can include:

- `predicted_template`
- `sections` generated from reconstructed paragraphs and assigned headings
- `infobox` fields mapped to the predicted template
- `entities` from spaCy NER
- `relations` from simple context-based extraction
- `generated_wikitext` when requested

## Supported infobox extraction

The current prototype includes template-specific extraction logic for:

- `Infobox person`
- `Infobox musical artist`
- `Infobox country`
- `Infobox company`

Other predicted templates currently fall back to a generic minimal infobox.

## Notes

- The classifier is required at runtime. If the model file is missing, the API returns: `"Model not trained. Run train_classifier.py first."`
- The NER model is required at runtime. If `en_core_web_sm` is missing, the API returns a `503` with installation guidance instead of silently degrading extraction quality.
- Plain-text section generation is heuristic. It reconstructs paragraphs and assigns headings such as `Overview`, `Early life`, `Career`, and `Personal life`, but it is not yet a learned paragraph classifier.
- The current prototype starts with scikit-learn so it remains lightweight and runnable locally.
