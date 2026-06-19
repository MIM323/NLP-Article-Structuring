from __future__ import annotations

import re

from app.ml.classifier import ModelNotTrainedError, predict_template
from app.schemas import DebugInfo, PredictionCandidate, StructureRequest, StructureResponse
from app.services.infobox_service import extract_infobox
from app.services.ner_service import NERModelUnavailableError, extract_entities
from app.services.relation_service import extract_relations
from app.services.section_service import split_into_sections
from app.services.wikitext_service import generate_wikitext
from app.storage.database import save_structure_request


def _lead_text(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[0] if paragraphs else text.strip()


def structure_article(payload: StructureRequest) -> StructureResponse:
    prediction = predict_template(payload.title, payload.text)
    predicted_template = prediction["label"]

    entities = extract_entities(payload.text)
    sections = split_into_sections(payload.text, predicted_template)
    infobox = extract_infobox(payload.title, payload.text, predicted_template, entities)
    relations = extract_relations(payload.title, payload.text, entities, predicted_template)

    generated_wikitext = None
    if payload.options.generateWikitext:
        generated_wikitext = generate_wikitext(
            title=payload.title,
            infobox=infobox,
            lead_text=_lead_text(payload.text),
            sections=sections,
        )

    debug = None
    if payload.options.returnDebug:
        debug = DebugInfo(
            model_confidence=prediction["confidence"],
            requested_template=payload.options.template,
            top_predictions=[
                PredictionCandidate(**candidate) for candidate in prediction["top_predictions"]
            ],
            extra=payload.options.extra,
        )

    response = StructureResponse(
        title=payload.title,
        predicted_template=predicted_template,
        sections=sections,
        infobox=infobox,
        entities=entities,
        relations=relations,
        generated_wikitext=generated_wikitext,
        debug=debug,
    )
    save_structure_request(
        title=payload.title,
        input_text=payload.text,
        predicted_template=predicted_template,
        output_json=response.model_dump(mode="json"),
    )
    return response
