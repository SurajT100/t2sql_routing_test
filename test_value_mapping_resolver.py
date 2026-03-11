import json
import sys
import types
import unittest
from unittest.mock import patch

from context_agent import ContextAgent
from entity_resolver import resolve_entities


class ValueMappingTests(unittest.TestCase):
    def test_context_agent_adds_canonical_candidates(self):
        sfc = [{
            "table": "public.orders",
            "column": "status",
            "user_value": "closed or inprogress",
            "filter_type": "include",
        }]
        rules = json.dumps([
            {
                "type": "mapping",
                "table": "orders",
                "column": "status",
                "rule_data": {
                    "mapping_type": "value_mapping",
                    "mappings": {
                        "closed": ["clsd"],
                        "inprogress": ["in_prgs"],
                    },
                },
            }
        ])

        updated = ContextAgent._attach_canonical_candidates(sfc, rules)
        self.assertEqual(updated[0].get("canonical_candidates"), ["clsd", "in_prgs"])
        self.assertEqual(updated[0].get("normalization_source"), "business_rule")

    def test_resolver_prefers_canonical_candidates(self):
        sfc = [{
            "table": "orders",
            "column": "status",
            "user_value": "closed or inprogress",
            "canonical_candidates": ["clsd", "in_prgs"],
        }]
        metadata = {"orders": {"status": {"data_type": "text"}}}

        def fake_run_query(_engine, query: str):
            if "= 'clsd'" in query:
                return ["clsd"]
            if "= 'in_prgs'" in query:
                return ["in_prgs"]
            return []

        fake_sqlalchemy = types.ModuleType("sqlalchemy")
        fake_sqlalchemy.text = lambda x: x

        with patch.dict(sys.modules, {"sqlalchemy": fake_sqlalchemy}), patch(
            "entity_resolver._run_resolve_query", side_effect=fake_run_query
        ):
            result = resolve_entities(
                user_engine=None,
                string_filter_columns=sfc,
                metadata=metadata,
                dialect_info={"dialect": "postgresql", "quote_char": '"'},
            )

        self.assertEqual(len(result.resolutions), 1)
        resolution = result.resolutions[0]
        self.assertEqual(resolution.strategy, "in_list")
        self.assertEqual(resolution.filter_condition, "IN ('clsd', 'in_prgs')")
        self.assertEqual(resolution.confidence, "high")


if __name__ == "__main__":
    unittest.main()
