from __future__ import annotations

import unittest

from app.services.section_service import split_into_sections


class SectionServiceTests(unittest.TestCase):
    def test_plain_text_is_split_into_multiple_paragraphs(self) -> None:
        text = (
            "William Frederick Durst is an American rapper and singer. "
            "He is the frontman of Limp Bizkit. "
            "In 1994, he formed the band with Sam Rivers and John Otto. "
            "The band signed with Flip Records and released its debut album. "
            "In 2006, Durst began working on independent films. "
            "He made his directorial debut with The Education of Charlie Banks. "
            "In 2009, Limp Bizkit reunited and resumed touring. "
            "Durst later married again and continued directing films."
        )

        sections = split_into_sections(text, "Infobox musical artist")

        self.assertGreaterEqual(len(sections), 2)
        self.assertEqual(sections[0].heading, "Overview")
        self.assertTrue(any("In 1994, he formed the band" in section.content for section in sections))
        self.assertTrue(any(section.heading == "Personal life" for section in sections))

    def test_headed_sections_reconstruct_internal_paragraphs(self) -> None:
        text = (
            "== Career ==\n"
            "Durst formed Limp Bizkit in 1994. The band signed with Flip Records. "
            "In 2006, he started working in film. He directed The Education of Charlie Banks.\n"
            "== Personal life ==\n"
            "Durst married Rachel Tergesen in 1990. They later divorced. "
            "In 2009, he married Esther Nazarov."
        )

        sections = split_into_sections(text, "Infobox musical artist")

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].heading, "Career")
        self.assertIn("\n\n", sections[0].content)
        self.assertEqual(sections[1].heading, "Personal life")
        self.assertIn("\n\n", sections[1].content)

    def test_messy_plain_text_gets_named_sections_and_strips_tail_noise(self) -> None:
        text = (
            "William Frederick Durst is an American rapper, singer, songwriter, actor, and director. "
            "Since 2006, Durst has worked on a number of independent films. He made his directorial debut in 2007. "
            "Early life Durst was born in Jacksonville, Florida, and moved to North Carolina as a child. "
            "He graduated from Hunter Huss High School. Career In 1994, Durst formed Limp Bizkit with Sam Rivers and John Otto. "
            "In 1997, Limp Bizkit signed with Flip Records, a subsidiary of Interscope Records. "
            "Personal life Durst married Rachel Tergesen in 1990. They later divorced. "
            "Arrests On July 13, 1999, Durst was arrested after kicking a security guard. "
            "Feuds Britney Spears Durst later claimed he was in a relationship with Britney Spears. "
            "References External links Limp Bizkit website Category:1970 births"
        )

        sections = split_into_sections(text, "Infobox musical artist")
        headings = [section.heading for section in sections]

        self.assertIn("Overview", headings)
        self.assertIn("Early life", headings)
        self.assertIn("Career", headings)
        self.assertIn("Personal life", headings)
        self.assertIn("Legal issues", headings)
        self.assertIn("Feuds", headings)
        self.assertIn("independent films", sections[0].content)
        self.assertTrue(all("Category:" not in section.content for section in sections))
        self.assertTrue(all("External links" not in section.content for section in sections))


if __name__ == "__main__":
    unittest.main()
