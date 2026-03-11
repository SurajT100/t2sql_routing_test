import unittest

from sql_static_validator import _check_multiple_statements


class MultipleStatementDetectionTests(unittest.TestCase):
    def _has_issue(self, sql: str) -> bool:
        return any(i.issue_type == "multiple_statements" for i in _check_multiple_statements(sql))

    def test_union_is_valid_single_statement(self):
        sql = "SELECT customer_id FROM orders UNION SELECT customer_id FROM archived_orders"
        self.assertFalse(self._has_issue(sql))

    def test_with_union_all_is_valid_single_statement(self):
        sql = (
            "WITH cte AS (SELECT customer_id FROM orders) "
            "SELECT customer_id FROM cte UNION ALL SELECT customer_id FROM archived_orders"
        )
        self.assertFalse(self._has_issue(sql))

    def test_adjacent_select_without_set_operator_is_invalid(self):
        sql = "SELECT customer_id FROM orders SELECT customer_id FROM archived_orders"
        self.assertTrue(self._has_issue(sql))

    def test_semicolon_split_statements_is_invalid(self):
        sql = "SELECT customer_id FROM orders; SELECT customer_id FROM archived_orders"
        self.assertTrue(self._has_issue(sql))


if __name__ == "__main__":
    unittest.main()
