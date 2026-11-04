import unittest

from rag_engine.rag_query_classifier import detect_topic
from rag_engine.rag_query_profile import build_universal_query_profile
from rag_engine.rag_query_generator import build_search_queries


class ExternalServiceBriefingRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.query = (
            "Надо ли проводить вводный инструктаж медицинскому работнику, "
            "который будет проводить медицинский осмотр у нас на территории"
        )

    def test_medical_context_does_not_override_briefing_topic(self):
        self.assertEqual(detect_topic(self.query), "occupational_briefing")

    def test_profile_targets_instruction_175_paragraph_46(self):
        profile = build_universal_query_profile(self.query)
        self.assertEqual(profile["target_document"], "175")
        self.assertEqual(profile["target_article"], "46")
        self.assertIn("external_service_provider_briefing", profile["qualifiers"])

    def test_search_queries_include_external_provider_legal_phrases(self):
        profile = build_universal_query_profile(self.query)
        plan = {
            "jurisdiction": "BY",
            "domain": "occupational_safety",
            "legal_phrases": [],
            "legal_concepts": [],
            "search_queries": [],
            "legal_targets": [],
        }
        queries = build_search_queries(
            self.query,
            "occupational_briefing",
            "occupational_safety",
            [],
            universal_profile=profile,
            legal_plan=plan,
        )
        joined = " ".join(queries).lower()
        self.assertIn("инструкция № 175 пункт 46", joined)
        self.assertIn("оказание услуг на территории организации", joined)


if __name__ == "__main__":
    unittest.main()
