from __future__ import annotations

from app.schemas import Infobox, Section


def _render_infobox(infobox: Infobox) -> str:
    lines = ["{{" + infobox.template]
    for field, value in infobox.fields.items():
        if value in (None, "", []):
            continue
        lines.append(f"| {field} = {value}")
    lines.append("}}")
    return "\n".join(lines)


def generate_wikitext(title: str, infobox: Infobox, lead_text: str, sections: list[Section]) -> str:
    blocks: list[str] = [_render_infobox(infobox), lead_text.strip()]
    for section in sections:
        if not section.content.strip():
            continue
        blocks.append(f"== {section.heading.strip()} ==\n{section.content.strip()}")
    return "\n\n".join(block for block in blocks if block.strip()).strip() + "\n"
