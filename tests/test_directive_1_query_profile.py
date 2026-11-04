import unittest

from rag_engine.rag_query_profile import build_universal_query_profile


class DirectiveOneQueryProfileTests(unittest.TestCase):
    def test_explicit_directive_name_sets_target_document(self):
        profile = build_universal_query_profile(
            "Согласно Директиве Президента Республики Беларусь от 11 марта 2004 г. "
            "№ 1 «О мерах по укреплению общественной безопасности и дисциплины», "
            "какие обязанности возлагаются на руководителей организаций?"
        )
        self.assertEqual(
            profile["target_document"],
            "общественной безопасности и дисциплины",
        )
        self.assertEqual(profile["event"], "organizational_discipline")
        self.assertIn(
            "directive_1_organizational_discipline",
            profile["qualifiers"],
        )

    def test_distinctive_leadership_discipline_question_sets_target(self):
        profile = build_universal_query_profile(
            "Какие обязанности возлагаются на руководителей организаций "
            "по обеспечению дисциплины и безопасности?"
        )
        self.assertEqual(
            profile["target_document"],
            "общественной безопасности и дисциплины",
        )

    def test_unrelated_question_does_not_force_directive(self):
        profile = build_universal_query_profile(
            "Какие требования охраны труда действуют при работе на токарном станке?"
        )
        self.assertNotEqual(
            profile["target_document"],
            "общественной безопасности и дисциплины",
        )


if __name__ == "__main__":
    unittest.main()
