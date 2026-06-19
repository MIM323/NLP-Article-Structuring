from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell


def extract_text_element(element: ET.Element, name: str, namespace: dict[str, str]) -> str:
    namespaced = element.findtext(f"mw:{name}", default="", namespaces=namespace)
    if namespaced:
        return namespaced
    return element.findtext(name, default="")


def iter_pages(xml_path: Path):
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    namespace_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    namespace = {"mw": namespace_uri} if namespace_uri else {}
    page_tag = f"{{{namespace_uri}}}page" if namespace_uri else "page"

    for event, elem in context:
        if event == "end" and elem.tag == page_tag:
            yield elem, root, namespace, namespace_uri
            elem.clear()
            root.clear()


def normalize_template_name(template_name: str) -> str:
    name = template_name.strip().replace("_", " ")
    name = re.sub(r"\s+", " ", name).lower()
    return name


def is_person_article(wikitext: str) -> bool:
    try:
        wikicode = mwparserfromhell.parse(wikitext)
        for template in wikicode.filter_templates(recursive=True):
            name = normalize_template_name(str(template.name))
            if name == "infobox person":
                return True
            if name.startswith("infobox ") and any(
                keyword in name
                for keyword in (
                    "biography",
                    "scientist",
                    "writer",
                    "artist",
                    "officeholder",
                    "musical artist",
                    "football biography",
                    "actor",
                )
            ):
                return True
    except Exception:
        return False
    return False


def write_xml_header(handle, namespace_uri: str) -> None:
    handle.write('<?xml version="1.0" encoding="utf-8"?>\n')
    if namespace_uri:
        handle.write(f'<mediawiki xmlns="{namespace_uri}">\n')
    else:
        handle.write("<mediawiki>\n")


def filter_people_dump(xml_path: Path, out_path: Path, limit: int | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    wrote_header = False

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        for page, _, namespace, namespace_uri in iter_pages(xml_path):
            if not wrote_header:
                write_xml_header(handle, namespace_uri)
                wrote_header = True

            title = extract_text_element(page, "title", namespace).strip()
            revision = page.find("mw:revision", namespace) if namespace else page.find("revision")
            if revision is None:
                continue

            text = extract_text_element(revision, "text", namespace)
            if not title or not text or text.lstrip().upper().startswith("#REDIRECT"):
                continue

            if not is_person_article(text):
                continue

            xml_fragment = ET.tostring(page, encoding="unicode")
            handle.write(xml_fragment)
            handle.write("\n")
            kept += 1

            if kept % 1000 == 0:
                print(f"Kept {kept} people pages...")

            if limit is not None and kept >= limit:
                break

        if not wrote_header:
            write_xml_header(handle, "")
        handle.write("</mediawiki>\n")

    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-filter a Wikipedia XML dump into a smaller people-only XML file."
    )
    parser.add_argument("--xml", required=True, help="Path to the input Wikipedia XML dump.")
    parser.add_argument("--out", required=True, help="Path to the output filtered XML file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of kept people pages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = filter_people_dump(Path(args.xml), Path(args.out), args.limit)
    print(f"Wrote {count} people pages to {args.out}")


if __name__ == "__main__":
    main()
