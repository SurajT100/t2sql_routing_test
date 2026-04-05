import json
import unittest

from context_agent import ContextAgent


class ContextAgentRuleFilteringTests(unittest.TestCase):
    def test_filter_rules_matches_quoted_and_unquoted_table_names(self):
        rules = [
            {"rule_name": "r1", "type": "metric", "tables": ['"sales"."orders"']},
            {"rule_name": "r2", "type": "metric", "tables": ["`sales`.`orders`"]},
            {"rule_name": "r3", "type": "metric", "tables": ["[sales].[orders]"]},
            {"rule_name": "r4", "type": "metric", "tables": ["sales.orders"]},
            {"rule_name": "r5", "type": "metric", "table": '"sales"."orders"'},
            {"rule_name": "r6", "type": "metric", "tables": ["orders"]},
            {"rule_name": "r7", "type": "metric", "tables": ["sales.customers"]},
        ]

        filtered_json, _ = ContextAgent._filter_rules_by_tables(
            rules_compressed=json.dumps(rules),
            identified_tables=["sales.orders"],
            joins_needed=True,
        )
        filtered = json.loads(filtered_json)
        kept_names = [r["rule_name"] for r in filtered]

        self.assertEqual(kept_names, ["r1", "r2", "r3", "r4", "r5", "r6"])


if __name__ == "__main__":
    unittest.main()
