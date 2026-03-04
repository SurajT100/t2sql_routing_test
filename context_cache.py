"""
Context Cache
=============
Caches reusable query context (schema text + rulebook) across questions.

Purpose:
- Avoid rebuilding and re-sending heavy context for every question
- Invalidate automatically when schema or business rules change
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class ContextBundle:
    schema_version: str
    rules_version: str
    bare_schema: str
    schema_text: str
    rules_compressed: str
    rules_retrieved: int


class ContextCache:
    """Simple in-memory context cache keyed by schema/rules/table selection."""

    def __init__(self) -> None:
        self._store: Dict[str, ContextBundle] = {}

    @staticmethod
    def _normalize_tables(selected_tables: List[str]) -> List[str]:
        out = []
        for t in selected_tables or []:
            if ": " in t:
                out.append(t.split(": ", 1)[1])
            else:
                out.append(t)
        return sorted(set(out))

    @staticmethod
    def compute_schema_version(engine, selected_tables: List[str]) -> str:
        from sqlalchemy import inspect
        from query_cache import QueryCache

        inspector = inspect(engine)
        schema_info: Dict[str, Dict[str, str]] = {}
        for full in ContextCache._normalize_tables(selected_tables):
            if "." in full:
                schema, table = full.split(".", 1)
            else:
                schema, table = "public", full
            try:
                cols = inspector.get_columns(table, schema=schema)
                schema_info[f"{schema}.{table}"] = {
                    c["name"]: str(c.get("type", "")) for c in cols
                }
            except Exception:
                continue

        return QueryCache.compute_schema_version(schema_info)

    @staticmethod
    def compute_rules_version(vector_engine) -> str:
        from sqlalchemy import text
        from query_cache import QueryCache

        rules: List[Dict[str, Any]] = []
        try:
            with vector_engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, rule_name, rule_type, rule_data, trigger_keywords,
                               priority, is_mandatory
                        FROM business_rules_v2
                        WHERE is_active = TRUE
                        ORDER BY id ASC
                        """
                    )
                ).fetchall()
            for r in rows:
                rules.append(
                    {
                        "id": r[0],
                        "rule_name": r[1],
                        "rule_type": r[2],
                        "rule_data": r[3],
                        "trigger_keywords": r[4],
                        "priority": r[5],
                        "is_mandatory": r[6],
                    }
                )
        except Exception:
            return "empty"

        return QueryCache.compute_rules_version(rules)

    @staticmethod
    def make_key(
        selected_tables: List[str],
        dialect: str,
        include_opus: bool,
        schema_version: str,
        rules_version: str,
    ) -> str:
        payload = "|".join(
            [
                ",".join(ContextCache._normalize_tables(selected_tables)),
                dialect,
                str(include_opus),
                schema_version,
                rules_version,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:40]

    def get(self, key: str) -> Optional[ContextBundle]:
        return self._store.get(key)

    def set(self, key: str, bundle: ContextBundle) -> None:
        self._store[key] = bundle


GLOBAL_CONTEXT_CACHE = ContextCache()
