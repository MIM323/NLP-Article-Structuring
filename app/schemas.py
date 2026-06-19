from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StructureOptions(BaseModel):
    generateWikitext: bool = True
    template: str = "Infobox person"
    language: str = "en"
    returnDebug: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class StructureRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    options: StructureOptions = Field(default_factory=StructureOptions)


class Section(BaseModel):
    heading: str
    content: str


class Infobox(BaseModel):
    template: str
    fields: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Relation(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float
    method: str


class PredictionCandidate(BaseModel):
    label: str
    confidence: float


class DebugInfo(BaseModel):
    model_confidence: float
    requested_template: str | None = None
    top_predictions: list[PredictionCandidate] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class StructureResponse(BaseModel):
    title: str
    predicted_template: str
    sections: list[Section] = Field(default_factory=list)
    infobox: Infobox
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    generated_wikitext: str | None = None
    debug: DebugInfo | None = None


class UnstructuredArticleSample(BaseModel):
    title: str
    text: str
    label: str
