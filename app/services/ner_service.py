from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.language import Language

from app.schemas import Entity

MAX_SPACY_CHARS = 200_000
ALLOWED_ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "NORP",
    "FAC",
    "EVENT",
}


class NERModelUnavailableError(RuntimeError):
    pass


def _truncate_text(text: str, max_chars: int = MAX_SPACY_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    window = cleaned[:max_chars]
    match = list(re.finditer(r"(?<=[.!?])\s+", window))
    if match:
        return window[: match[-1].end()].strip()
    return window.strip()


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise NERModelUnavailableError(
            "spaCy model 'en_core_web_sm' is not installed. Run "
            "'python -m spacy download en_core_web_sm' before starting the API."
        )


def extract_entities(text: str) -> list[Entity]:
    doc = get_nlp()(_truncate_text(text))
    entities: list[Entity] = []
    for ent in doc.ents:
        if ent.label_ not in ALLOWED_ENTITY_LABELS:
            continue
        entities.append(
            Entity(
                text=ent.text.strip(),
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
        )
    return entities
