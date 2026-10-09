"""Regression tests for lathe-safety query routing and topic classification."""

import unittest

from rag_query_classifier import detect_legal_domain, detect_topic


class LatheWorkClassificationTests(unittest.TestCase):
    def test_lathe_query_routes_to_occupational_safety(self):
        query = "Какие требования охраны труда необходимо соблюдать при работе на токарном станке?"
        self.assertEqual(detect_legal_domain(query), "occupational_safety")
        self.assertEqual(detect_topic(query), "lathe_work")

    def test_lathe_synonym_routes_to_lathe_topic(self):
        self.assertEqual(
            detect_topic("Требования безопасности при работе на токарном оборудовании"),
            "lathe_work",
        )

    def test_cold_metalworking_query_routes_to_lathe_topic(self):
        self.assertEqual(
            detect_topic("Правила охраны труда при холодной обработке металлов"),
            "lathe_work",
        )

    def test_unrelated_query_stays_general(self):
        self.assertEqual(detect_topic("Расскажи об общих требованиях"), "general")
        self.assertEqual(detect_legal_domain("Расскажи об общих требованиях"), "general")


if __name__ == "__main__":
    unittest.main()
