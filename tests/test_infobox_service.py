from __future__ import annotations

import unittest

from app.schemas import Entity
from app.services.infobox_service import extract_infobox


class InfoboxServiceTests(unittest.TestCase):
    def test_person_fields_prefer_lead_context(self) -> None:
        text = (
            "Ada Lovelace (born 10 December 1815) was an English mathematician and writer, "
            "chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer.\n\n"
            "She was the daughter of Lord Byron and Anne Isabella Milbanke."
        )
        entities = [
            Entity(text="10 December 1815", label="DATE", start=19, end=35),
            Entity(text="English", label="NORP", start=45, end=52),
            Entity(text="Charles Babbage", label="PERSON", start=110, end=126),
            Entity(text="Lord Byron", label="PERSON", start=203, end=213),
            Entity(text="Anne Isabella Milbanke", label="PERSON", start=218, end=241),
        ]

        infobox = extract_infobox("Ada Lovelace", text, "Infobox person", entities)

        self.assertEqual(infobox.fields["name"], "Ada Lovelace")
        self.assertEqual(infobox.fields["birth_date"], "10 December 1815")
        self.assertEqual(infobox.fields["nationality"], "English")
        self.assertEqual(infobox.fields["occupation"], "mathematician and writer")
        self.assertIn("work on Charles Babbage", infobox.fields["known_for"])
        self.assertIn("Lord Byron", infobox.fields["parents"])

    def test_country_fields_use_canonical_names(self) -> None:
        text = (
            "France is a country in Western Europe. Its capital is Paris and its largest city is Paris. "
            "The official language is French. It has a population of 67,000,000 and an area of 551,695 km2."
        )
        entities = [
            Entity(text="France", label="GPE", start=0, end=6),
            Entity(text="Western Europe", label="LOC", start=23, end=37),
            Entity(text="Paris", label="GPE", start=55, end=60),
            Entity(text="Paris", label="GPE", start=85, end=90),
            Entity(text="French", label="NORP", start=122, end=128),
        ]

        infobox = extract_infobox("France", text, "Infobox country", entities)

        self.assertEqual(infobox.fields["capital"], "Paris")
        self.assertEqual(infobox.fields["largest_city"], "Paris")
        self.assertEqual(infobox.fields["official_languages"], "French")
        self.assertEqual(infobox.fields["population"], "67,000,000")
        self.assertEqual(infobox.fields["area_km2"], "551,695 km2")

    def test_company_fields_choose_founder_and_headquarters_from_lead(self) -> None:
        text = (
            "OpenAI is an artificial intelligence company founded in 2015 by Sam Altman, Elon Musk, "
            "Ilya Sutskever, Greg Brockman, and others, headquartered in San Francisco. "
            "Its products include ChatGPT and DALL-E."
        )
        entities = [
            Entity(text="OpenAI", label="ORG", start=0, end=6),
            Entity(text="2015", label="DATE", start=55, end=59),
            Entity(text="Sam Altman", label="PERSON", start=63, end=74),
            Entity(text="Elon Musk", label="PERSON", start=76, end=85),
            Entity(text="Ilya Sutskever", label="PERSON", start=87, end=102),
            Entity(text="Greg Brockman", label="PERSON", start=104, end=118),
            Entity(text="San Francisco", label="GPE", start=154, end=167),
            Entity(text="ChatGPT", label="ORG", start=191, end=198),
            Entity(text="DALL-E", label="ORG", start=203, end=209),
        ]

        infobox = extract_infobox("OpenAI", text, "Infobox company", entities)

        self.assertEqual(infobox.fields["founded"], "2015")
        self.assertTrue(infobox.fields["founder"].startswith("Sam Altman"))
        self.assertEqual(infobox.fields["headquarters"], "San Francisco")
        self.assertEqual(infobox.fields["products"], "ChatGPT and DALL-E")

    def test_company_fields_can_use_sections_for_products(self) -> None:
        text = (
            "ExampleSoft is a software company based in Vilnius.\n\n"
            "History:\n"
            "The company was founded in 2019 by Lina Example.\n\n"
            "Products:\n"
            "Products include Alpha Suite and Beta Cloud."
        )
        entities = [
            Entity(text="ExampleSoft", label="ORG", start=0, end=11),
            Entity(text="Vilnius", label="GPE", start=40, end=47),
            Entity(text="2019", label="DATE", start=85, end=89),
            Entity(text="Lina Example", label="PERSON", start=93, end=105),
        ]

        infobox = extract_infobox("ExampleSoft", text, "Infobox company", entities)

        self.assertEqual(infobox.fields["founded"], "2019")
        self.assertEqual(infobox.fields["founder"], "Lina Example")
        self.assertEqual(infobox.fields["headquarters"], "Vilnius")
        self.assertEqual(infobox.fields["products"], "Alpha Suite and Beta Cloud")

    def test_musical_artist_fields_extract_core_infobox_values(self) -> None:
        text = (
            'Tony Russell "Charles" Brown (September 13, 1922 - January 21, 1999) '
            "was an American singer and pianist whose soft-toned nightclub style influenced West Coast blues. "
            "Brown was born in Texas City, Texas. He settled in Los Angeles in 1943. "
            "He signed with Exclusive Records and Aladdin Records. "
            "Brown died in Oakland, California.\n\n"
            "Awards:\n"
            "Brown was inducted into the Blues Hall of Fame in 1996 and the Rock and Roll Hall of Fame in 1999. "
            "He also received a National Heritage Fellowship."
        )
        entities = [
            Entity(text="September 13, 1922", label="DATE", start=31, end=49),
            Entity(text="January 21, 1999", label="DATE", start=52, end=68),
            Entity(text="American", label="NORP", start=76, end=84),
            Entity(text="West Coast blues", label="EVENT", start=137, end=153),
            Entity(text="Texas City, Texas", label="GPE", start=173, end=190),
            Entity(text="Los Angeles", label="GPE", start=210, end=221),
            Entity(text="Oakland, California", label="GPE", start=307, end=326),
        ]

        infobox = extract_infobox("Charles Brown (musician)", text, "Infobox musical artist", entities)

        self.assertEqual(infobox.fields["name"], "Charles Brown")
        self.assertEqual(infobox.fields["birth_name"], "Tony Russell Brown")
        self.assertEqual(infobox.fields["alias"], "Charles")
        self.assertEqual(infobox.fields["birth_date"], "September 13, 1922")
        self.assertEqual(infobox.fields["birth_place"], "Texas City, Texas")
        self.assertEqual(infobox.fields["origin"], "Los Angeles")
        self.assertEqual(infobox.fields["death_date"], "January 21, 1999")
        self.assertEqual(infobox.fields["death_place"], "Oakland, California")
        self.assertEqual(infobox.fields["genre"], "West Coast blues")
        self.assertEqual(infobox.fields["occupation"], "singer and pianist")
        self.assertEqual(infobox.fields["instrument"], "piano, vocals")
        self.assertEqual(infobox.fields["label"], "Exclusive Records, Aladdin Records")
        self.assertIn("Blues Hall of Fame", infobox.fields["awards"])


if __name__ == "__main__":
    unittest.main()
