"""
Unit tests for detect_fy_start_month() and build_fy_context().

Run with: pytest test_fy_context.py -v

All date assertions use dynamically passed `today` — no hardcoded dates
except the mock values for the test scenarios themselves.
"""

import pytest
from datetime import date
from reasoning_prompts import detect_fy_start_month, build_fy_context


# ---------------------------------------------------------------------------
# Helper: build a compressed-format "default" rule dict
# ---------------------------------------------------------------------------

def _fy_rule(desc: str, apply: str = "", name: str = "Fiscal Year Rule") -> dict:
    """Build a compressed default rule that detect_fy_start_month() will parse."""
    return {
        "type": "default",
        "name": name,
        "auto_apply": True,
        "applies_to_queries": ["revenue", "sales", "pipeline"],
        "apply": apply,
        "desc": desc,
    }


# ---------------------------------------------------------------------------
# Tests for detect_fy_start_month()
# ---------------------------------------------------------------------------

class TestDetectFyStartMonth:

    def test_april_fy_from_description(self):
        """Detects April FY start when description mentions 'april 1' and 'march'."""
        rules = [_fy_rule(desc="fiscal year from april 1 to march 31")]
        assert detect_fy_start_month(rules) == 4

    def test_april_fy_from_apply_text(self):
        """Detects April FY start when apply field mentions 'from april' and 'march'."""
        rules = [_fy_rule(desc="financial year", apply="date from april 1 to march")]
        assert detect_fy_start_month(rules) == 4

    def test_august_fy(self):
        """Detects August FY start when description mentions 'august 1' and 'july'."""
        rules = [_fy_rule(desc="fiscal year from august 1 to july 31")]
        assert detect_fy_start_month(rules) == 8

    def test_no_fy_rule(self):
        """Returns None when no auto_apply default rule with FY indicators exists."""
        rules = [
            {"type": "metric", "name": "Revenue", "formula": "SUM(amount)"},
            {"type": "filter", "name": "Exclude Rebate", "apply": "category <> 'Rebate'"},
        ]
        assert detect_fy_start_month(rules) is None

    def test_empty_rules(self):
        """Returns None for empty rules list."""
        assert detect_fy_start_month([]) is None

    def test_non_default_rule_ignored(self):
        """A 'filter' rule with FY keywords is ignored (not a default rule)."""
        rules = [{"type": "filter", "name": "FY Filter", "auto_apply": True,
                  "desc": "fiscal year from april 1 to march"}]
        assert detect_fy_start_month(rules) is None

    def test_no_auto_apply_ignored(self):
        """A default rule without auto_apply=True is ignored."""
        rules = [{"type": "default", "name": "FY Rule", "auto_apply": False,
                  "desc": "fiscal year from april 1 to march"}]
        assert detect_fy_start_month(rules) is None

    def test_explicit_fy_start_month_field(self):
        """Explicit fy_start_month field takes precedence over text parsing."""
        rules = [{"type": "default", "auto_apply": True, "fy_start_month": 7,
                  "desc": "fiscal year runs from july"}]
        assert detect_fy_start_month(rules) == 7

    def test_unparseable_fy_rule_returns_none(self, caplog):
        """FY keyword found but no month pattern matched → returns None + logs warning."""
        import logging
        rules = [_fy_rule(desc="apply custom fiscal period with no month names")]
        with caplog.at_level(logging.WARNING):
            result = detect_fy_start_month(rules)
        assert result is None
        # Warning should have been logged (via logging module)

    def test_january_start_ignored(self):
        """January FY start = calendar year — skipped by design."""
        rules = [_fy_rule(desc="fiscal year from january 1 to december 31")]
        assert detect_fy_start_month(rules) is None


# ---------------------------------------------------------------------------
# Tests for build_fy_context()
# ---------------------------------------------------------------------------

class TestBuildFyContext:

    # ---- Test 1: today=2026-04-09, FY starts April ----

    def test_april_fy_early_april(self):
        """Test 1: 2026-04-09, FY start=April."""
        today = date(2026, 4, 9)
        result = build_fy_context(today, fy_start_month=4, rule_name="FY Rule")

        assert "2026-04-01" in result  # current FY start
        assert "2027-03-31" in result  # current FY end
        assert "2025-04-01" in result  # prev FY start
        assert "2026-03-31" in result  # prev FY end

        # Q1 of April FY: Apr-Jun
        assert "Q1" in result
        assert "2026-04-01" in result  # Q1 start
        assert "2026-06-30" in result  # Q1 end

        # "last quarter" = Q4 of prev FY: Jan-Mar 2026
        assert "Q4" in result
        assert "2026-01-01" in result  # prev Q start
        assert "2026-03-31" in result  # prev Q end

    # ---- Test 2: today=2027-02-15, FY starts April ----

    def test_april_fy_february(self):
        """Test 2: 2027-02-15, FY start=April. February is in Q4 of FY 2026."""
        today = date(2027, 2, 15)
        result = build_fy_context(today, fy_start_month=4)

        # Current FY is 2026-04-01 → 2027-03-31 (Feb 2027 is before Apr 2027)
        assert "2026-04-01" in result
        assert "2027-03-31" in result
        # Prev FY is 2025-04-01 → 2026-03-31
        assert "2025-04-01" in result
        assert "2026-03-31" in result

        # Q4 (Jan-Mar): Jan 2027 – Mar 2027
        assert "Q4" in result
        assert "2027-01-01" in result
        assert "2027-03-31" in result

        # Previous quarter = Q3 (Oct-Dec 2026)
        assert "Q3" in result
        assert "2026-10-01" in result
        assert "2026-12-31" in result

    # ---- Test 3: today=2026-10-15, FY starts August ----

    def test_august_fy_october(self):
        """Test 3: 2026-10-15, FY start=August."""
        today = date(2026, 10, 15)
        result = build_fy_context(today, fy_start_month=8)

        # Current FY: Aug 2026 – Jul 2027
        assert "2026-08-01" in result
        assert "2027-07-31" in result
        # Prev FY: Aug 2025 – Jul 2026
        assert "2025-08-01" in result
        assert "2026-07-31" in result

        # Q1 of Aug FY: Aug-Oct 2026
        assert "Q1" in result
        assert "2026-08-01" in result
        assert "2026-10-31" in result

        # Previous quarter = Q4 of prev FY: May-Jul 2026
        assert "Q4" in result
        assert "2026-05-01" in result
        assert "2026-07-31" in result

    # ---- Test 4: no FY rule → context not injected ----

    def test_no_fy_rule_no_injection(self):
        """Test 4: detect_fy_start_month returns None → build_fy_context not called."""
        rules = [{"type": "metric", "name": "Revenue"}]
        fy_start = detect_fy_start_month(rules)
        assert fy_start is None

        # If caller checks and skips build_fy_context when fy_start is None,
        # the fy_context string should be empty (simulating the call site logic).
        today = date(2026, 4, 9)
        fy_context = build_fy_context(today, fy_start) if fy_start else ""
        assert fy_context == ""

    # ---- Test 5: unparseable rule → None → no context ----

    def test_unparseable_rule_no_context(self):
        """Test 5: FY keyword present but unparseable → detect returns None."""
        rules = [_fy_rule(desc="apply custom fiscal period")]
        fy_start = detect_fy_start_month(rules)
        assert fy_start is None

        today = date(2026, 4, 9)
        fy_context = build_fy_context(today, fy_start) if fy_start else ""
        assert fy_context == ""

    # ---- Structural checks ----

    def test_output_contains_header(self):
        """FISCAL YEAR CONTEXT header always present in non-empty output."""
        result = build_fy_context(date(2026, 4, 9), 4, rule_name="My FY Rule")
        assert "FISCAL YEAR CONTEXT" in result
        assert '"My FY Rule"' in result

    def test_invalid_month_returns_empty(self):
        """build_fy_context returns '' for invalid fy_start_month."""
        assert build_fy_context(date(2026, 4, 9), 0) == ""
        assert build_fy_context(date(2026, 4, 9), 13) == ""

    def test_applicable_tables_in_output(self):
        """applicable_tables list appears in the output header."""
        result = build_fy_context(
            date(2026, 4, 9), 4,
            rule_name="FY Rule",
            applicable_tables=["revenue", "sales", "pipeline"]
        )
        assert "revenue" in result
        assert "sales" in result

    def test_no_hardcoded_years(self):
        """Verify output changes with a different year (not hardcoded to 2026)."""
        result_2026 = build_fy_context(date(2026, 4, 9), 4)
        result_2028 = build_fy_context(date(2028, 4, 9), 4)
        assert "2028-04-01" in result_2028
        assert "2026-04-01" not in result_2028
        assert "2026-04-01" in result_2026
        assert "2028-04-01" not in result_2026
