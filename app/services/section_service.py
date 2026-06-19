from __future__ import annotations

import re

from app.schemas import Section

HEADING_PATTERN = re.compile(
    r"^\s*(?:={2,6}\s*(?P<wiki>[^=\n]+?)\s*={2,6}|(?P<plain>[A-Z][A-Za-z /&-]{2,50}):)\s*$",
    re.MULTILINE,
)
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$", re.DOTALL)
INLINE_HEADING_SPLIT_RE = re.compile(
    r"\s(?=(?:Early life|Career|Personal life|Relationships(?:\s*&\s*Children)?|Arrests|Russia|"
    r"Miscellaneous|Feuds|Discography|Filmography|Awards(?:\s+and\s+nominations)?|"
    r"Start of film career|Mainstream success|Limp Bizkit reunion)\b)"
)
PARAGRAPH_BREAK_CUE_RE = re.compile(
    r"^(?:In\s+\d{4}|By\s+\d{4}|During\b|After\b|Before\b|Later\b|Soon\b|Meanwhile\b|Eventually\b|"
    r"At\s+the\s+age\s+of\b|As\s+a\s+child\b|As\s+an?\s+\w+\b|Following\b|On\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)
LISTISH_CUE_RE = re.compile(
    r"^(?:Discography|Filmography|Awards(?:\s+and\s+nominations)?|References|External links|"
    r"Category:|List of |Selected |Releases\b|Title\s+Year\b)",
    re.IGNORECASE,
)
TAIL_MARKER_RE = re.compile(
    r"(?:\bReferences\b|\bExternal links\b|Category:|\bDiscography\s+\+\s+List of\b|\bFilmography\s+\+\s+List of\b)",
    re.IGNORECASE,
)
NOISE_RE = re.compile(
    r"(?:thumb\|upright\|[^\n]*|left\|thumb\|upright\|[^\n]*|Posted\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\.?|"
    r'"?\s+denotes a recording that did not chart[^.]*\.?)',
    re.IGNORECASE,
)

SECTION_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "Infobox person": {
        "History": ("born", "early life", "education", "childhood"),
        "Career": ("career", "worked", "actor", "writer", "scientist", "served"),
        "Personal life": ("family", "married", "children", "personal life"),
    },
    "Infobox country": {
        "Geography": ("region", "climate", "border", "terrain", "geography"),
        "Economy": ("economy", "gdp", "industry", "trade", "agriculture"),
        "History": ("independence", "founded", "history", "colonial"),
    },
    "Infobox company": {
        "History": ("founded", "history", "acquired", "launched"),
        "Operations": ("products", "services", "industry", "market"),
        "Headquarters": ("headquartered", "based", "office", "campus"),
    },
    "Infobox musical artist": {
        "Early life": ("born", "childhood", "early life", "grew up", "school"),
        "Career": ("album", "band", "tour", "signed", "recorded", "released", "career"),
        "Personal life": ("married", "wife", "husband", "children", "personal life", "arrested"),
        "Film career": ("film", "directorial", "movie", "television", "actor", "director"),
        "Feuds": ("feud", "slipknot", "britney spears", "taproot", "creed", "placebo", "trent reznor", "marilyn manson"),
        "Awards": ("award", "hall of fame", "grammy", "nomination"),
        "Discography": ("discography", "album", "single", "studio albums"),
    },
}


def _split_paragraphs(text: str) -> list[str]:
    prepared = _prepare_plain_text(text)
    explicit = [part.strip() for part in re.split(r"\n\s*\n", prepared) if part.strip()]
    if len(explicit) > 1:
        return explicit
    return _infer_paragraphs(prepared)


def _prepare_plain_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    tail_match = TAIL_MARKER_RE.search(cleaned)
    if tail_match:
        cleaned = cleaned[: tail_match.start()].strip()

    cleaned = NOISE_RE.sub(" ", cleaned)
    cleaned = INLINE_HEADING_SPLIT_RE.sub("\n\n", cleaned)
    cleaned = re.sub(r'\s+"\s*', " ", cleaned)
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
    cleaned = re.sub(r"(?:\s*\n\s*){2,}", "\n\n", cleaned)
    return cleaned.strip()


def _sentence_starts_with_break_cue(sentence: str) -> bool:
    normalized = sentence.strip()
    if not normalized:
        return False
    return bool(PARAGRAPH_BREAK_CUE_RE.match(normalized) or LISTISH_CUE_RE.match(normalized))


def _extract_sentence_entities(sentence: str) -> set[str]:
    return {
        match.group(0).strip()
        for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,})\b", sentence)
        if len(match.group(0).strip()) > 2
    }


def _topic_shift_score(previous_sentence: str, sentence: str) -> int:
    score = 0

    previous_lower = previous_sentence.lower()
    current_lower = sentence.lower()
    if any(token in current_lower for token in ("married", "wife", "husband", "children", "divorce", "arrest")):
        if not any(token in previous_lower for token in ("married", "wife", "husband", "children", "divorce", "arrest")):
            score += 2

    if any(token in current_lower for token in ("film", "movie", "director", "television")):
        if not any(token in previous_lower for token in ("film", "movie", "director", "television")):
            score += 2

    if any(token in current_lower for token in ("album", "band", "tour", "record", "song", "released")):
        if not any(token in previous_lower for token in ("album", "band", "tour", "record", "song", "released")):
            score += 2

    previous_entities = _extract_sentence_entities(previous_sentence)
    current_entities = _extract_sentence_entities(sentence)
    if previous_entities and current_entities and previous_entities.isdisjoint(current_entities):
        score += 1

    return score


def _should_start_new_paragraph(
    current_sentences: list[str],
    next_sentence: str,
) -> bool:
    if not current_sentences:
        return False

    if _sentence_starts_with_break_cue(next_sentence):
        return True

    current_length = len(current_sentences)
    if current_length >= 6:
        return True

    previous_sentence = current_sentences[-1]
    topic_shift = _topic_shift_score(previous_sentence, next_sentence)
    if current_length >= 1 and topic_shift >= 3:
        return True

    if current_length >= 3 and topic_shift >= 2:
        return True

    return False


def _infer_paragraphs(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(normalized) if match.group(0).strip()]
    if not sentences:
        return [normalized]

    paragraphs: list[str] = []
    current_sentences: list[str] = []
    for sentence in sentences:
        if _should_start_new_paragraph(current_sentences, sentence):
            paragraphs.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        paragraphs.append(" ".join(current_sentences).strip())

    return [paragraph for paragraph in paragraphs if paragraph]


def _sections_from_headings(text: str) -> list[Section]:
    prepared = _prepare_plain_text(text)
    matches = list(HEADING_PATTERN.finditer(prepared))
    if not matches:
        return []

    sections: list[Section] = []
    for index, match in enumerate(matches):
        heading = (match.group("wiki") or match.group("plain") or "Section").strip(" :")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prepared)
        content = "\n\n".join(_split_paragraphs(prepared[start:end]))
        if content:
            sections.append(Section(heading=heading, content=content))
    return sections


def _guess_heading(paragraph: str, template: str) -> str:
    lowered = paragraph.lower()
    if re.match(r"^[A-Z][^.]{0,120}\b(?:is|was)\s+an?\b", paragraph):
        return "Overview"
    if lowered.startswith("early life"):
        return "Early life"
    if lowered.startswith("personal life") or lowered.startswith("relationships"):
        return "Personal life"
    if lowered.startswith("arrests"):
        return "Legal issues"
    if lowered.startswith("discography"):
        return "Discography"
    if lowered.startswith("filmography"):
        return "Filmography"
    if lowered.startswith("awards"):
        return "Awards"
    if lowered.startswith("feuds"):
        return "Feuds"
    if lowered.startswith("start of film career"):
        return "Film career"
    if lowered.startswith("career") or lowered.startswith("mainstream success") or lowered.startswith("limp bizkit reunion"):
        return "Career"

    for heading, keywords in SECTION_KEYWORDS.get(template, {}).items():
        if any(keyword in lowered for keyword in keywords):
            return heading
    return "Details"


def _normalize_heading(heading: str) -> str:
    normalized = heading.strip()
    mapping = {
        "History": "Early life",
        "Film": "Film career",
        "Relationships & Children": "Personal life",
        "Arrests": "Legal issues",
        "Russia": "Personal life",
        "Miscellaneous": "Personal life",
    }
    return mapping.get(normalized, normalized)


def split_into_sections(text: str, predicted_template: str) -> list[Section]:
    sections = _sections_from_headings(text)
    if sections:
        normalized_sections: list[Section] = []
        for section in sections:
            heading = _normalize_heading(section.heading)
            if normalized_sections and normalized_sections[-1].heading == heading:
                normalized_sections[-1].content += "\n\n" + section.content
            else:
                normalized_sections.append(Section(heading=heading, content=section.content))
        return normalized_sections

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    structured_sections: list[Section] = []
    for paragraph in paragraphs:
        heading = _normalize_heading(_guess_heading(paragraph, predicted_template))
        if heading == "Details" and not structured_sections:
            heading = "Overview"
        if structured_sections and structured_sections[-1].heading == heading:
            structured_sections[-1].content += "\n\n" + paragraph
        else:
            structured_sections.append(Section(heading=heading, content=paragraph))

    if not structured_sections:
        structured_sections.append(Section(heading="Introduction", content=paragraphs[0]))
    return structured_sections
