"""Regression tests for conveyor-safety query routing."""

import unittest

from rag_query_classifier import detect_legal_domain


class ConveyorDomainClassificationTests(unittest.TestCase):
    def test_conveyor_operation_routes_to_occupational_safety(self):
        self.assertEqual(
            detect_legal_domain("Каковы требования безопасности при эксплуатации конвейеров?"),
            "occupational_safety",
        )

    def test_emergency_conveyor_stop_routes_to_occupational_safety(self):
        self.assertEqual(
            detect_legal_domain("Какие требования к аварийной остановке конвейера?"),
            "occupational_safety",
        )

    def test_continuous_transport_equipment_routes_to_occupational_safety(self):
        self.assertEqual(
            detect_legal_domain("Правила эксплуатации транспортных средств непрерывного действия"),
            "occupational_safety",
        )

    def test_explicit_industrial_safety_remains_industrial_safety(self):
        self.assertEqual(
            detect_legal_domain("Промышленная безопасность опасного производственного объекта"),
            "industrial_safety",
        )

    def test_unrelated_general_query_stays_general(self):
        self.assertEqual(
            detect_legal_domain("Расскажи об общих требованиях"),
            "general",
        )


if __name__ == "__main__":
    unittest.main()
