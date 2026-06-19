from __future__ import annotations

import re

from app.schemas import Entity, Relation


def extract_relations(title: str, text: str, entities: list[Entity], predicted_template: str) -> list[Relation]:
    relations: list[Relation] = []

    def add_relation(subject: str, relation: str, obj: str, confidence: float, method: str) -> None:
        if not obj:
            return
        relations.append(
            Relation(
                subject=subject,
                relation=relation,
                object=obj,
                confidence=confidence,
                method=method,
            )
        )

    for entity in entities:
        context_start = max(0, entity.start - 80)
        context_end = min(len(text), entity.end + 80)
        context = text[context_start:context_end].lower()

        if entity.label in {"GPE", "LOC"} and "born" in context:
            add_relation(title, "born_in", entity.text, 0.78, "ner+context")
        if entity.label == "DATE" and "born" in context:
            add_relation(title, "born_on", entity.text, 0.76, "ner+context")
        if entity.label == "DATE" and ("founded" in context or "established" in context):
            add_relation(title, "founded_in", entity.text, 0.79, "ner+context")
        if entity.label in {"PERSON", "ORG"} and ("founded by" in context or "founder" in context):
            add_relation(title, "founded_by", entity.text, 0.8, "ner+context")
        if entity.label in {"GPE", "LOC"} and ("located in" in context or "headquartered in" in context):
            add_relation(title, "located_in", entity.text, 0.77, "ner+context")

    if predicted_template == "Infobox country":
        capital_match = re.search(r"\bcapital(?: city)?\s+(?:is|was)?\s*([^.,;\n]{2,80})", text, re.IGNORECASE)
        if capital_match:
            add_relation(capital_match.group(1).strip(), "capital_of", title, 0.72, "regex")

    deduped: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (relation.subject, relation.relation, relation.object)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relation)
    return deduped
