from __future__ import annotations

import re
from collections import Counter

from app.schemas import Entity, Infobox

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?(?:\s+|$)", re.DOTALL)
SECTION_HEADING_RE = re.compile(r"^\s*(?:={2,6}\s*([^=\n]+?)\s*={2,6}|([A-Z][A-Za-z /&-]{2,50}):)\s*$", re.MULTILINE)
LIST_SEPARATOR_RE = re.compile(r"\s*,\s*|\s+and\s+")


def _lead_paragraph(text: str) -> str:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return parts[0] if parts else text.strip()


def _lead_sentence(text: str) -> str:
    match = re.search(r"(.+?[.!?])(?:\s|$)", text.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _clean_field_value(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", value)
    cleaned = re.sub(r"\s+\(([^)]*)\)", r" (\1)", cleaned)
    return cleaned.strip(" .,;:")


def _strip_disambiguation(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def _trim_at_clauses(value: str, stop_phrases: tuple[str, ...]) -> str:
    lowered = value.lower()
    cutoff = len(value)
    for phrase in stop_phrases:
        index = lowered.find(phrase.lower())
        if index != -1:
            cutoff = min(cutoff, index)
    return _clean_field_value(value[:cutoff])


def _split_values(value: str) -> list[str]:
    if not value:
        return []
    return [
        item
        for item in (_clean_field_value(part) for part in LIST_SEPARATOR_RE.split(value))
        if item
    ]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        if sentence.strip():
            spans.append((match.start(), match.end()))
    if not spans and text.strip():
        spans.append((0, len(text)))
    return spans


def _sentence_index(position: int, spans: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return -1


def _iter_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = (match.group(1) or match.group(2) or "Section").strip(" :")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((heading, content))
    return sections


def _find_section_text(text: str, keywords: tuple[str, ...]) -> str:
    for heading, content in _iter_sections(text):
        lowered = heading.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return content
    return ""


def _keyword_positions(text: str, keywords: tuple[str, ...]) -> list[int]:
    positions: list[int] = []
    lowered = text.lower()
    for keyword in keywords:
        start = 0
        token = keyword.lower()
        while True:
            index = lowered.find(token, start)
            if index == -1:
                break
            positions.append(index)
            start = index + len(token)
    return sorted(positions)


def _entity_distance_score(entity: Entity, keyword_positions: list[int]) -> int:
    if not keyword_positions:
        return 0
    distance = min(abs(entity.start - position) for position in keyword_positions)
    return max(0, 30 - (distance // 12))


def _entity_length_penalty(entity: Entity) -> int:
    token_count = len(entity.text.split())
    if token_count >= 8:
        return 25
    if token_count >= 6:
        return 10
    return 0


def _find_entity(
    entities: list[Entity],
    labels: set[str],
    context_text: str | None = None,
    context_keywords: tuple[str, ...] = (),
) -> str | None:
    candidates = [entity for entity in entities if entity.label in labels]
    if not candidates:
        return None

    if not context_text or not context_keywords:
        return candidates[0].text

    spans = _sentence_spans(context_text)
    keyword_positions = _keyword_positions(context_text, context_keywords)
    keyword_sentence_indexes = {_sentence_index(position, spans) for position in keyword_positions}
    lead_length = len(_lead_paragraph(context_text))

    best_entity: Entity | None = None
    best_score: int | None = None
    for entity in candidates:
        score = 100
        if entity.end <= lead_length:
            score += 25

        sentence_idx = _sentence_index(entity.start, spans)
        if sentence_idx in keyword_sentence_indexes:
            score += 40

        score += _entity_distance_score(entity, keyword_positions)
        score -= _entity_length_penalty(entity)
        score -= entity.start // 200

        if best_score is None or score > best_score:
            best_entity = entity
            best_score = score

    return best_entity.text if best_entity else None


def _extract_pattern(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _clean_field_value(match.group(1))


def _extract_pattern_any(texts: list[str], patterns: tuple[str, ...]) -> str | None:
    for text in texts:
        if not text:
            continue
        for pattern in patterns:
            match = _extract_pattern(text, pattern)
            if match:
                return match
    return None


def _normalize_person_list(value: str) -> str:
    cleaned = _trim_at_clauses(
        value,
        (" headquartered ", " based in ", " with headquarters ", " whose headquarters "),
    )
    cleaned = re.sub(r",?\s*and others\b.*$", "", cleaned, flags=re.IGNORECASE)
    names = _dedupe_preserve_order(_split_values(cleaned))
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _normalize_occupation(value: str, nationality: str) -> str:
    cleaned = _trim_at_clauses(
        value,
        (
            ", chiefly known",
            ", widely known",
            ", best known",
            " founded in ",
            " founded by ",
            " headquartered in ",
            " based in ",
        ),
    )
    if nationality:
        cleaned = re.sub(rf"^{re.escape(nationality)}\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcompany\b.*$", "company", cleaned, flags=re.IGNORECASE)
    return _clean_field_value(cleaned)


def _normalize_location(value: str) -> str:
    cleaned = _trim_at_clauses(value, (" and ", " with ", " where ", " which "))
    return _clean_field_value(cleaned)


def _normalize_language(value: str) -> str:
    items = _dedupe_preserve_order(_split_values(value))
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items)


def _normalize_list_field(value: str) -> str:
    items = _dedupe_preserve_order(_split_values(value))
    if not items:
        return ""
    return ", ".join(items)


def _extract_birth_date(text: str, entities: list[Entity]) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    for pattern in (
        r"\(born\s+([^)]+)\)",
        r"\bborn\s+([^,.;()]+(?:\d{4}))",
    ):
        match = _extract_pattern(lead, pattern)
        if match:
            return match
    return _find_entity(entities, {"DATE"}, lead, ("born", "birth")) or ""


def _extract_death_date(text: str, entities: list[Entity]) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    for pattern in (
        r"\((?:born\s+[^)]*;\s*)?died\s+([^)]+)\)",
        r"\bdied\s+([^,.;()]+(?:\d{4}))",
    ):
        match = _extract_pattern(lead, pattern)
        if match:
            return match
    return _find_entity(entities, {"DATE"}, lead, ("died", "death")) or ""


def _extract_person_nationality(text: str, entities: list[Entity]) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    nationality = _find_entity(entities, {"NORP"}, lead, ("is a", "was a", "is an", "was an"))
    if nationality:
        return nationality

    match = re.search(r"\b(?:is|was)\s+(?:an?|the)\s+([A-Z][a-z]+)\b", lead)
    if match:
        return match.group(1)
    return ""


def _extract_person_occupation(text: str, nationality: str) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    match = re.search(
        r"\b(?:is|was)\s+(?:an?|the)\s+([^.,;\n]{3,120}?)(?:,\s+(?:best|chiefly|widely)\s+known\b|[.;]|$)",
        lead,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    return _normalize_occupation(match.group(1), nationality)


def _extract_person_known_for(text: str) -> str:
    lead = _lead_paragraph(text)
    career = _find_section_text(text, ("career", "work"))
    return _clean_field_value(
        _extract_pattern_any(
            [lead, career],
            (
                r"\b(?:best|chiefly|widely)\s+known\s+for\s+([^.;\n]{3,200})",
                r"\bknown\s+for\s+([^.;\n]{3,200})",
            ),
        )
    )


def _extract_parents(text: str) -> str:
    return _normalize_person_list(
        _extract_pattern(
            text,
            r"\b(?:only\s+legitimate\s+child|child|daughter|son)\s+of\s+(.+?)(?:\s+and\s+(?:was|is)\b|[.;\n]|$)",
        )
        or ""
    )


def _extract_birth_place(text: str, entities: list[Entity]) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    match = _extract_pattern(
        lead,
        r"\bborn(?:\s+[^,.;()]+)?\s+in\s+([^,.;()]+)",
    )
    if match:
        return _normalize_location(match)
    return _normalize_location(_find_entity(entities, {"GPE", "LOC"}, lead, ("born in", "birth place")) or "")


def _extract_life_dates(text: str) -> tuple[str, str]:
    lead = _lead_sentence(_lead_paragraph(text))
    match = re.search(r"\(([^()]*\d{4}[^()]*)\)", lead)
    if not match:
        return "", ""
    raw = _clean_field_value(match.group(1))
    parts = re.split(r"\s*[–-]\s*", raw, maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return _clean_field_value(parts[0]), _clean_field_value(parts[1])


def _extract_death_place(text: str, entities: list[Entity]) -> str:
    match = _extract_pattern(
        text,
        r"\bdied(?:\s+of\s+[^,.;()]+)?\s+in\s+([^.;()]+)",
    )
    if match:
        return _normalize_location(match)
    return _normalize_location(_find_entity(entities, {"GPE", "LOC"}, text, ("died in", "death place")) or "")


def _extract_alias_from_lead(text: str) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    match = re.search(r'"([^"]{2,60})"', lead)
    if not match:
        return ""
    return _clean_field_value(match.group(1))


def _extract_birth_name(text: str, title: str) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    title_name = _strip_disambiguation(title)
    match = re.match(r"([A-Z][^()]{3,120}?)\s*\(", lead)
    if not match:
        return ""
    candidate = _clean_field_value(match.group(1))
    if not candidate or candidate == title_name:
        return ""
    candidate = re.sub(r'\s*"[^"]+"\s*', " ", candidate)
    candidate = _clean_field_value(candidate)
    if candidate == title_name:
        return ""
    return candidate


def _extract_artist_genre(text: str) -> str:
    lead = _lead_paragraph(text)
    career = _find_section_text(text, ("career",))
    for source in (lead, career, text):
        match = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+blues)\b",
            source,
        )
        if match:
            return _clean_field_value(match.group(1))
        match = re.search(
            r"\b(?:influenced|played|performed|sang)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+blues)\b",
            source,
        )
        if match:
            return _clean_field_value(match.group(1))
    lowered = " ".join(part for part in (lead, career, text) if part).lower()
    genres = [
        genre
        for genre in (
            "blues",
            "jazz",
            "rock",
            "soul",
            "pop",
            "country",
            "folk",
            "gospel",
            "r&b",
            "rhythm and blues",
        )
        if genre in lowered
    ]
    return _normalize_list_field(", ".join(genres))


def _extract_artist_occupation(text: str, entities: list[Entity]) -> str:
    lead = _lead_sentence(_lead_paragraph(text))
    nationality = _extract_person_nationality(text, entities)
    match = re.search(
        r"\b(?:is|was)\s+(?:an?\s+)?(?:[A-Z][a-z]+\s+)?([^.,;\n]{3,120}?)(?:\s+whose\b|\s+who\b|,\s|\.\s*|$)",
        lead,
        flags=re.IGNORECASE,
    )
    if not match:
        return _extract_person_occupation(text, nationality)
    return _normalize_occupation(match.group(1), nationality)


def _extract_artist_instrument(text: str, occupation: str) -> str:
    lowered = f"{occupation} {_lead_paragraph(text)}".lower()
    instruments: list[str] = []
    instrument_map = {
        "pianist": "piano",
        "piano": "piano",
        "guitarist": "guitar",
        "guitar": "guitar",
        "drummer": "drums",
        "drums": "drums",
        "saxophonist": "saxophone",
        "saxophone": "saxophone",
        "violinist": "violin",
        "violin": "violin",
        "bassist": "bass",
        "bass": "bass",
        "vocal": "vocals",
        "singer": "vocals",
    }
    for token, canonical in instrument_map.items():
        if token in lowered:
            instruments.append(canonical)
    return _normalize_list_field(", ".join(instruments))


def _extract_artist_origin(text: str, entities: list[Entity]) -> str:
    lead = _lead_paragraph(text)
    career = _find_section_text(text, ("career", "early life"))
    match = _extract_pattern_any(
        [lead, career, text],
        (
            r"\b(?:settled|based)\s+in\s+([^,.;\n]{2,120})",
            r"\bfrom\s+([^,.;\n]{2,120})",
        ),
    )
    if match:
        return _clean_field_value(re.sub(r"\s+in\s+\d{4}\b", "", _normalize_location(match)))
    return _extract_birth_place(text, entities)


def _extract_artist_labels(text: str) -> str:
    labels = re.findall(
        r"\b([A-Z][A-Za-z&.'/-]*(?:\s+[A-Z][A-Za-z&.'/-]*)*\s+Records)\b",
        text,
    )
    return _normalize_list_field(", ".join(labels))


def _extract_artist_awards(text: str) -> str:
    awards_section = _find_section_text(text, ("awards", "honors", "tributes"))
    source = awards_section or text
    matches = re.findall(
        r"\b(?:National Heritage Fellowship|Grammy Award(?: nominations?)?|Blues Hall of Fame|Rock and Roll Hall of Fame)\b",
        source,
        flags=re.IGNORECASE,
    )
    normalized = [
        _clean_field_value(match.replace(" nominations", "").replace(" nomination", ""))
        for match in matches
    ]
    return _normalize_list_field(", ".join(normalized))


def _extract_musical_artist_fields(title: str, text: str, entities: list[Entity]) -> dict[str, str]:
    display_name = _strip_disambiguation(title)
    occupation = _extract_artist_occupation(text, entities)
    birth_date, death_date = _extract_life_dates(text)
    fields: dict[str, str] = {"name": display_name}
    fields["birth_name"] = _extract_birth_name(text, title)
    fields["alias"] = _extract_alias_from_lead(text)
    fields["birth_date"] = birth_date or _extract_birth_date(text, entities)
    fields["birth_place"] = _extract_birth_place(text, entities)
    fields["origin"] = _extract_artist_origin(text, entities)
    fields["death_date"] = death_date or _extract_death_date(text, entities)
    fields["death_place"] = _extract_death_place(text, entities)
    fields["genre"] = _extract_artist_genre(text)
    fields["occupation"] = occupation
    fields["instrument"] = _extract_artist_instrument(text, occupation)
    fields["label"] = _extract_artist_labels(text)
    fields["awards"] = _extract_artist_awards(text)
    return {key: value for key, value in fields.items() if value}


def _extract_person_fields(title: str, text: str, entities: list[Entity]) -> dict[str, str]:
    fields: dict[str, str] = {"name": _strip_disambiguation(title)}
    nationality = _extract_person_nationality(text, entities)
    fields["birth_date"] = _extract_birth_date(text, entities)
    fields["death_date"] = _extract_death_date(text, entities)
    fields["birth_place"] = _extract_birth_place(text, entities)
    fields["occupation"] = _extract_person_occupation(text, nationality)
    fields["known_for"] = _extract_person_known_for(text)
    fields["nationality"] = nationality
    fields["parents"] = _extract_parents(text)
    return {key: value for key, value in fields.items() if value}


def _extract_country_fields(title: str, text: str, entities: list[Entity]) -> dict[str, str]:
    lead = _lead_paragraph(text)
    geography = _find_section_text(text, ("geography", "administrative", "overview"))
    fields: dict[str, str] = {"name": title}
    fields["capital"] = _normalize_location(
        _extract_pattern_any(
            [lead, geography],
            (
                r"\bcapital(?:\s+city)?\s+(?:is|was)?\s*([^.,;\n]{2,80}?)(?=\s+and\b|[.,;\n]|$)",
            ),
        )
        or _find_entity(entities, {"GPE"}, lead, ("capital",))
        or ""
    )
    fields["largest_city"] = _normalize_location(
        _extract_pattern_any([lead, geography], (r"\blargest city\s+(?:is|was)?\s*([^.,;\n]{2,80})",)) or ""
    )
    fields["population"] = _extract_pattern(text, r"\bpopulation(?: of)?\s+([\d,]+)") or ""
    fields["area_km2"] = _extract_pattern(
        text,
        r"\barea(?: of)?\s+([\d,.\s]+(?:km2|square kilometres|sq mi)[^.,;\n]*)",
    ) or ""
    fields["official_languages"] = _normalize_language(
        _extract_pattern_any(
            [lead, geography, text],
            (r"\bofficial language(?:s)?\s+(?:is|are)?\s*([^.,;\n]{2,80})",),
        )
        or ""
    )
    return {key: value for key, value in fields.items() if value}


def _extract_company_fields(title: str, text: str, entities: list[Entity]) -> dict[str, str]:
    lead = _lead_paragraph(text)
    history = _find_section_text(text, ("history", "background"))
    operations = _find_section_text(text, ("products", "operations", "services"))
    fields: dict[str, str] = {"name": title}
    fields["founded"] = (
        _extract_pattern_any([lead, history], (r"\b(?:founded|established)\s+in\s+(\d{4})\b",))
        or _find_entity(entities, {"DATE"}, lead, ("founded", "established"))
        or ""
    )
    fields["founder"] = _normalize_person_list(
        (
            _extract_pattern_any(
                [lead, history],
                (r"\bfounded(?:\s+in\s+\d{4})?\s+by\s+([^.;\n]{2,160})",),
            )
            or _find_entity(entities, {"PERSON", "ORG"}, lead, ("founded by", "founder"))
            or ""
        )
    )
    fields["headquarters"] = _normalize_location(
        _extract_pattern_any(
            [lead, history],
            (
                r"\b(?:headquartered|based)\s+in\s+([^,.;\n]{2,120})",
                r"\bheadquarters(?: are| is)?\s+in\s+([^,.;\n]{2,120})",
            ),
        )
        or _find_entity(
            entities,
            {"GPE", "LOC", "FAC"},
            lead,
            ("headquartered", "based in", "headquarters"),
        )
        or ""
    )
    fields["industry"] = _normalize_occupation(
        _extract_pattern_any(
            [lead, history],
            (r"\b(?:is|was)\s+(?:an?|the)\s+([^.,;\n]{3,120})",),
        )
        or "",
        "",
    )
    fields["products"] = _clean_field_value(
        _extract_pattern_any(
            [operations, text],
            (r"\bproducts?(?: include| are|:)?\s*([^.;\n]{3,120})",),
        )
    )
    return {key: value for key, value in fields.items() if value}


def _generic_fields(title: str, entities: list[Entity]) -> dict[str, str]:
    fields = {"name": _strip_disambiguation(title)}
    label_counts = Counter(entity.label for entity in entities)
    for label in ("PERSON", "ORG", "GPE"):
        if label_counts[label]:
            fields["primary_entity_type"] = label
            break
    return fields


def extract_infobox(title: str, text: str, predicted_template: str, entities: list[Entity]) -> Infobox:
    template_key = predicted_template.strip().lower()

    if template_key == "infobox person":
        fields = _extract_person_fields(title, text, entities)
    elif template_key == "infobox musical artist":
        fields = _extract_musical_artist_fields(title, text, entities)
    elif template_key == "infobox country":
        fields = _extract_country_fields(title, text, entities)
    elif template_key == "infobox company":
        fields = _extract_company_fields(title, text, entities)
    else:
        fields = _generic_fields(title, entities)

    return Infobox(template=predicted_template, fields=fields)
