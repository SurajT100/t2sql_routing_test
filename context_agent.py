"""
Context Agent — Focused Context Assembly After Column Identification
=====================================================================
The core architectural shift: instead of dumping ALL schema descriptions + ALL 
business rules into every LLM call upfront, this agent waits for Pass 1 to 
identify which columns are needed, then fetches ONLY:

1. Opus descriptions for those specific columns
2. Sample values + data types for those columns  
3. RAG business rules relevant to those columns/tables + question
4. Entity resolution for string filter columns (via live DB)

This produces a FOCUSED context bundle that Pass 2 and SQL Coder receive,
instead of the kitchen-sink approach.

Architecture position:
    Classification → Bare Schema → Pass 1 → **Context Agent** → Pass 2 → SQL Coder

The agent is NOT an LLM call — it's an orchestrator that makes targeted 
DB queries to Supabase (metadata) and user DB (resolver).

Usage:
    from context_agent import ContextAgent

    agent = ContextAgent(
        user_engine=engine,
        vector_engine=vector_engine,
        selected_tables=selected_tables,
        dialect_info=config.dialect_info,
        enable_resolver=True,
        enable_rule_rag=True,
        rule_rag_threshold=0.65,
        compress_rules=True
    )
    
    # After Pass 1:
    bundle = agent.fetch_context(
        question=question,
        pass1_data=parsed_pass1_output
    )
    
    # bundle.focused_schema  → schema with descriptions for ONLY relevant columns
    # bundle.rules_compressed → RAG rules relevant to identified columns + question
    # bundle.metadata → sample values, data types, descriptions per column
    # bundle.resolver_result → live entity resolutions
    # bundle.resolver_text → formatted for prompt injection
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ContextBundle:
    """Everything Pass 2 and SQL Coder need — focused, minimal, precise."""
    
    # Focused schema: only relevant columns with descriptions
    focused_schema: str = ""
    
    # Business rules: relevant to identified columns + question
    rules_compressed: str = "[]"
    rules_retrieved: int = 0
    
    # Column metadata: sample values, data types, descriptions per column
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Entity resolution results
    resolver_result: Any = None  # ResolverResult from entity_resolver
    resolver_text: str = ""      # Formatted for prompt injection
    entities_resolved: int = 0
    
    # Diagnostics
    total_time_ms: int = 0
    opus_descriptions_fetched: int = 0
    columns_fetched: int = 0
    resolver_queries: int = 0
    stage_times: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CONTEXT AGENT
# =============================================================================

class ContextAgent:
    """
    Orchestrates focused context retrieval after Pass 1 identifies columns.
    
    NOT an LLM — pure DB queries and data assembly.
    """
    
    def __init__(
        self,
        user_engine,
        vector_engine,
        selected_tables: List[str],
        dialect_info: Dict,
        enable_resolver: bool = True,
    ):
        self.user_engine = user_engine
        self.vector_engine = vector_engine
        self.selected_tables = self._clean_table_names(selected_tables)
        self.dialect_info = dialect_info
        self.dialect = dialect_info.get("dialect", "postgresql")
        self.quote_char = dialect_info.get("quote_char", '"')
        self.enable_resolver = enable_resolver
    
    def fetch_context(
        self,
        question: str,
        pass1_data: Dict,
        rules_compressed: str = "[]"
    ) -> ContextBundle:
        """
        Main entry point. Takes Pass 1 output and fetches everything needed.

        Rules are pre-fetched in Stage 2 (needed for Pass 1 column mapping).
        This agent fetches: Opus descriptions, sample values, entity resolution.
        Rules are filtered to identified tables before being passed to Pass 2.

        Args:
            question: Original user question
            pass1_data: Parsed Pass 1 output with columns, tables, string_filter_columns
            rules_compressed: Pre-fetched business rules from Stage 2 (all rules, unfiltered)

        Returns:
            ContextBundle with focused schema, rules, metadata, resolver results
        """
        bundle = ContextBundle()
        start_time = time.time()

        columns_by_table = pass1_data.get("columns", {})
        string_filter_columns = pass1_data.get("string_filter_columns", [])
        identified_tables = pass1_data.get("tables", [])
        joins_needed = pass1_data.get("joins_needed", False)

        if not columns_by_table:
            print("[CONTEXT AGENT] No columns identified by Pass 1 — returning empty bundle")
            bundle.rules_compressed = rules_compressed
            bundle.total_time_ms = int((time.time() - start_time) * 1000)
            return bundle

        # ── Step 0: Inject mandatory columns from rule_column_dependencies ──
        # Deterministically ensures columns required by business rules are
        # always present — regardless of whether Pass 1 mentioned them.
        columns_by_table = self._inject_dependency_columns(
            columns_by_table, identified_tables
        )

        total_cols = sum(len(v) for v in columns_by_table.values())
        print(f"[CONTEXT AGENT] Fetching context for {total_cols} columns across {len(columns_by_table)} tables")

        # ── Step 1: Fetch column metadata + Opus descriptions ──
        stage_start = time.time()
        bundle.metadata = self._fetch_column_metadata(
            columns_by_table, string_filter_columns
        )
        bundle.columns_fetched = total_cols
        bundle.opus_descriptions_fetched = self._count_descriptions(bundle.metadata)
        bundle.stage_times["metadata_fetch"] = int((time.time() - stage_start) * 1000)
        print(f"[CONTEXT AGENT] Metadata: {total_cols} columns, {bundle.opus_descriptions_fetched} descriptions ({bundle.stage_times['metadata_fetch']}ms)")

        # ── Step 2: Filter rules to only those relevant to identified tables ──
        # Pass 1 received ALL rules (needed for column mapping).
        # Pass 2 only needs rules for the tables it will actually query.
        filtered_rules, join_extra_tables = self._filter_rules_by_tables(
            rules_compressed, identified_tables, joins_needed
        )
        bundle.rules_compressed = filtered_rules
        print(f"[CONTEXT AGENT] Rules filtered: {bundle.rules_retrieved} → kept for identified tables "
              f"(joins_needed={joins_needed})")

        # ── Step 3: Build focused schema (only identified tables; only relevant cols get descriptions) ──
        # Pass 1 column keys give us the exact set of tables needed.
        # When joins are needed, also include tables referenced by kept join rules.
        stage_start = time.time()
        bundle.focused_schema = self._build_focused_schema(
            columns_by_table, bundle.metadata,
            extra_join_tables=join_extra_tables if joins_needed else set()
        )
        bundle.stage_times["schema_build"] = int((time.time() - stage_start) * 1000)

        # Count filtered rules for diagnostics
        try:
            rules_list = json.loads(bundle.rules_compressed)
            bundle.rules_retrieved = len(rules_list) if isinstance(rules_list, list) else 0
        except Exception:
            bundle.rules_retrieved = 0
        
        # ── Step 4: Entity resolution (live DB) ──
        if self.enable_resolver and string_filter_columns:
            stage_start = time.time()
            try:
                from entity_resolver import (
                    resolve_entities,
                    format_resolutions_for_prompt,
                    merge_resolutions_into_metadata
                )
                
                resolver_result = resolve_entities(
                    user_engine=self.user_engine,
                    string_filter_columns=string_filter_columns,
                    metadata=bundle.metadata,
                    dialect_info=self.dialect_info
                )
                
                bundle.resolver_result = resolver_result
                bundle.entities_resolved = len(resolver_result.resolutions)
                bundle.resolver_queries = resolver_result.queries_run
                bundle.resolver_text = format_resolutions_for_prompt(resolver_result)
                
                # Merge resolved values back into metadata
                bundle.metadata = merge_resolutions_into_metadata(
                    bundle.metadata, resolver_result
                )
                
                print(f"[CONTEXT AGENT] Resolver: {bundle.entities_resolved} entities, "
                      f"{resolver_result.queries_run} queries, "
                      f"all_resolved={resolver_result.all_resolved}")
                
            except ImportError:
                print("[CONTEXT AGENT] entity_resolver.py not found — skipping")
            except Exception as e:
                print(f"[CONTEXT AGENT] Resolver error (non-fatal): {e}")
            
            bundle.stage_times["resolver"] = int((time.time() - stage_start) * 1000)
        
        bundle.total_time_ms = int((time.time() - start_time) * 1000)
        print(f"[CONTEXT AGENT] Total: {bundle.total_time_ms}ms | "
              f"Cols: {bundle.columns_fetched} | Descs: {bundle.opus_descriptions_fetched} | "
              f"Rules: {bundle.rules_retrieved} | Entities: {bundle.entities_resolved}")
        
        return bundle
    
    # ═════════════════════════════════════════════════════════════════════
    # INTERNAL: Rule Filtering by Identified Tables
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _filter_rules_by_tables(
        rules_compressed: str,
        identified_tables: List[str],
        joins_needed: bool
    ):
        """
        Filter rules_compressed to only rules relevant to identified tables.

        Keep a rule if ANY of:
          - No tables/table field (general rule, applies everywhere)
          - tables field contains at least one identified table
          - table field matches an identified table
          - mandatory=true AND no tables/table field (explicit: always keep mandatory globals)
          - mandatory=true AND tables/table includes an identified table

        Drop a rule if:
          - tables field references ONLY tables not in the identified set
          - table field doesn't match any identified table

        Join rule handling:
          - joins_needed=false  → drop ALL type="join" rules
          - joins_needed=true   → keep join rules that reference ≥1 identified table

        Returns:
            (filtered_rules_json, join_extra_tables)
            join_extra_tables: set of normalized table names from kept join rules
                               (for expanding the focused schema when joins are needed)
        """
        try:
            rules_list = json.loads(rules_compressed)
            if not isinstance(rules_list, list):
                return rules_compressed, set()

            # Build normalized lookup set for fast membership tests
            identified_set: set = set()
            for t in identified_tables:
                identified_set.add(t.lower())
                if "." in t:
                    identified_set.add(t.split(".")[-1].lower())

            filtered = []
            join_extra_tables: set = set()

            for rule in rules_list:
                rule_type = rule.get("type", "")

                # ── join rules: drop wholesale when joins not needed ──
                if rule_type == "join" and not joins_needed:
                    continue

                tables_field = rule.get("tables")
                table_field = rule.get("table")

                # ── No table reference → general rule, always keep ──
                if tables_field is None and table_field is None:
                    filtered.append(rule)
                    continue

                # ── tables field (list or string) ──
                if tables_field is not None:
                    items = tables_field if isinstance(tables_field, list) else [tables_field]
                    norm = set()
                    for t in items:
                        s = str(t)
                        norm.add(s.lower())
                        if "." in s:
                            norm.add(s.split(".")[-1].lower())

                    if norm & identified_set:
                        filtered.append(rule)
                        # Collect all tables referenced by kept join rules
                        if rule_type == "join":
                            join_extra_tables |= norm
                    # else: ONLY non-identified tables → drop
                    continue

                # ── table field (single value) ──
                if table_field is not None:
                    s = str(table_field)
                    t_norm = s.lower()
                    t_clean = s.split(".")[-1].lower() if "." in s else t_norm
                    if t_norm in identified_set or t_clean in identified_set:
                        filtered.append(rule)
                        if rule_type == "join":
                            join_extra_tables.add(t_norm)
                            join_extra_tables.add(t_clean)
                    # else: table doesn't match → drop

            return json.dumps(filtered), join_extra_tables

        except Exception as e:
            print(f"[CONTEXT AGENT] Rule filtering error (returning unfiltered): {e}")
            return rules_compressed, set()

    # ═════════════════════════════════════════════════════════════════════
    # INTERNAL: Rule Dependency Column Injection
    # ═════════════════════════════════════════════════════════════════════

    def _inject_dependency_columns(
        self,
        columns_by_table: Dict[str, List[str]],
        identified_tables: List[str]
    ) -> Dict[str, List[str]]:
        """
        Query rule_column_dependencies for the identified tables and inject
        any mandatory columns not already in columns_by_table.

        This makes mandatory filter columns deterministic — they are always
        present in the focused context regardless of whether Pass 1 included
        them in its column identification output.

        Injected columns receive full metadata (descriptions, samples, ★ marker)
        automatically because the metadata fetch and schema builder iterate over
        whatever is in columns_by_table at the time they run.

        Returns the (possibly augmented) columns_by_table dict.
        """
        if not identified_tables and not columns_by_table:
            return columns_by_table

        try:
            from sqlalchemy import text

            # Build a normalised lookup: lowercase bare name → canonical key
            # as it appears in columns_by_table (so we can append to the right list).
            canonical: Dict[str, str] = {}
            for key in columns_by_table.keys():
                canonical[key.lower()] = key
                if "." in key:
                    canonical[key.split(".")[-1].lower()] = key

            # Collect all candidate table names to query (full + bare forms)
            query_tables: List[str] = list(
                set(identified_tables) | set(columns_by_table.keys())
            )
            bare_names = [t.split(".")[-1] for t in query_tables if "." in t]
            query_tables = list(set(query_tables + bare_names))

            with self.vector_engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT table_name, column_name, dependency_type, reason
                        FROM   rule_column_dependencies
                        WHERE  auto_apply = TRUE
                          AND  table_name = ANY(:tables)
                        ORDER BY table_name, column_name
                    """),
                    {"tables": query_tables}
                )
                rows = result.fetchall()

            injected = 0
            for row in rows:
                dep_table = row[0]
                dep_col   = row[1]
                dep_type  = row[2]
                reason    = row[3]

                dep_lower = dep_table.lower()
                dep_bare  = dep_table.split(".")[-1].lower() if "." in dep_table else dep_lower

                # Find which key in columns_by_table this table maps to
                matched_key = canonical.get(dep_lower) or canonical.get(dep_bare)

                # If the table isn't in columns_by_table yet, add it only if it
                # belongs to one of the tables Pass 1 identified.
                if matched_key is None:
                    for t in identified_tables:
                        t_lower = t.lower()
                        t_bare  = t.split(".")[-1].lower() if "." in t else t_lower
                        if t_lower == dep_lower or t_bare == dep_lower:
                            matched_key = t
                            columns_by_table[matched_key] = []
                            canonical[t_lower] = matched_key
                            canonical[t_bare]  = matched_key
                            break

                if matched_key is not None:
                    if dep_col not in columns_by_table[matched_key]:
                        columns_by_table[matched_key].append(dep_col)
                        injected += 1
                        print(
                            f"[CONTEXT AGENT] Injected {matched_key}.{dep_col} "
                            f"(type={dep_type}) from rule_column_dependencies "
                            f"— {reason}"
                        )

            if injected:
                print(f"[CONTEXT AGENT] rule_column_dependencies: "
                      f"{injected} column(s) injected across tables")

        except Exception as e:
            # Non-fatal: if the table doesn't exist yet or query fails,
            # continue without injection rather than breaking the whole pipeline.
            print(f"[CONTEXT AGENT] rule_column_dependencies injection error "
                  f"(non-fatal): {e}")

        return columns_by_table

    # ═════════════════════════════════════════════════════════════════════
    # INTERNAL: Column Metadata + Opus Descriptions
    # ═════════════════════════════════════════════════════════════════════

    def _fetch_column_metadata(
        self,
        columns_by_table: Dict[str, List[str]],
        string_filter_columns: List[Dict]
    ) -> Dict[str, Any]:
        """
        Fetch sample values, data types, AND Opus descriptions 
        for ONLY the columns identified by Pass 1.
        
        This replaces the old approach of:
        1. get_full_schema_with_opus() → ALL columns, ALL descriptions
        2. fetch_column_metadata() → only sample values for identified columns
        
        Now: ONE function fetches everything, ONLY for identified columns.
        """
        metadata = {}
        
        # Build string filter set for quick lookup
        string_filter_set = set()
        if string_filter_columns:
            for sf in string_filter_columns:
                if not isinstance(sf, dict):
                    continue
                string_filter_set.add((sf.get("table", ""), sf.get("column", "")))
        
        try:
            for table_name, columns in columns_by_table.items():
                metadata[table_name] = {}
                
                for col in columns:
                    try:
                        # Single query per column: gets data_type + samples + opus_description + user_description
                        from sqlalchemy import text as _text
                        with self.vector_engine.connect() as _conn:
                            _row = _conn.execute(
                                _text(
                                    "SELECT column_name, data_type, sample_values, "
                                    "opus_description, user_description, friendly_name "
                                    "FROM schema_columns "
                                    "WHERE object_name = :tname AND column_name = :cname "
                                    "LIMIT 1"
                                ),
                                {"tname": table_name, "cname": col}
                            ).mappings().fetchone()

                        if _row:
                            sample_values = _row.get("sample_values") or []

                            if isinstance(sample_values, str):
                                try:
                                    sample_values = json.loads(sample_values)
                                except Exception:
                                    sample_values = []

                            sample_values = sample_values[:10]

                            # Priority: user_description > opus_description > friendly_name
                            description = (
                                _row.get("user_description") or
                                _row.get("opus_description") or
                                _row.get("friendly_name") or ""
                            )

                            needs_partial = (table_name, col) in string_filter_set

                            metadata[table_name][col] = {
                                "data_type": _row.get("data_type", "unknown"),
                                "sample_values": sample_values,
                                "description": description,
                                "needs_partial_match": needs_partial
                            }
                        else:
                            metadata[table_name][col] = {
                                "data_type": "unknown",
                                "sample_values": [],
                                "description": "",
                                "needs_partial_match": (table_name, col) in string_filter_set
                            }
                    
                    except Exception as col_err:
                        print(f"[CONTEXT AGENT] Could not fetch {table_name}.{col}: {col_err}")
                        metadata[table_name][col] = {
                            "data_type": "unknown",
                            "sample_values": [],
                            "description": "",
                            "needs_partial_match": False
                        }
        
        except Exception as e:
            print(f"[CONTEXT AGENT] Metadata fetch failed: {e}")
        
        return metadata
    
    # ═════════════════════════════════════════════════════════════════════
    # INTERNAL: Focused Schema Builder
    # ═════════════════════════════════════════════════════════════════════
    
    def _build_focused_schema(
        self,
        columns_by_table: Dict[str, List[str]],
        metadata: Dict[str, Any],
        extra_join_tables: set = None
    ) -> str:
        """
        Build schema text for ONLY the tables Pass 1 identified.

        Only tables present in columns_by_table are included, reducing token
        usage from irrelevant tables.  When joins_needed=true, tables referenced
        by kept join rules (extra_join_tables) are also included even if Pass 1
        didn't put any columns there explicitly.

        Within each included table, identified columns get full descriptions;
        other columns are listed with name+type only.
        """
        from sqlalchemy import inspect

        q = self.quote_char
        qr = ']' if q == '[' else q

        inspector = inspect(self.user_engine)

        # Normalize the set of tables Pass 1 identified (bare name, lowercase)
        pass1_table_names: set = set()
        for t in columns_by_table.keys():
            pass1_table_names.add(t.lower())
            if "." in t:
                pass1_table_names.add(t.split(".")[-1].lower())

        # Extra tables from kept join rules (already normalized by _filter_rules_by_tables)
        join_tables: set = extra_join_tables or set()

        # Union of tables to include in the schema output
        tables_to_include: set = pass1_table_names | join_tables

        # Build set of identified columns for quick lookup
        identified_cols: set = set()
        for table, cols in columns_by_table.items():
            for col in cols:
                identified_cols.add((table, col))

        lines = [f"DATABASE: {self.dialect.upper()}"]

        for full_name in self.selected_tables:
            if "." in full_name:
                schema_name, table_name = full_name.split(".", 1)
            else:
                schema_name = None
                table_name = full_name

            # ── Skip tables Pass 1 didn't identify (and aren't needed for joins) ──
            if table_name.lower() not in tables_to_include and full_name.lower() not in tables_to_include:
                continue

            try:
                columns = inspector.get_columns(table_name, schema=schema_name)

                # Format table name
                if schema_name:
                    quoted_table = f"{q}{schema_name}{qr}.{q}{table_name}{qr}"
                else:
                    quoted_table = f"{q}{table_name}{qr}"

                lines.append(f"\n{quoted_table}:")

                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])
                    nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                    quoted_col = f"{q}{col_name}{qr}"

                    # Check if this column was identified by Pass 1
                    is_identified = (full_name, col_name) in identified_cols

                    # Also check without schema prefix (Pass 1 might say "SAP" not "public.SAP")
                    if not is_identified:
                        is_identified = (table_name, col_name) in identified_cols

                    if is_identified:
                        # ── IDENTIFIED COLUMN: include full description ──
                        col_meta = metadata.get(full_name, metadata.get(table_name, {})).get(col_name, {})
                        desc = col_meta.get("description", "")

                        if desc:
                            desc_short = desc[:120] if len(desc) > 120 else desc
                            lines.append(f"  ★ {quoted_col} ({col_type}) {nullable} — {desc_short}")
                        else:
                            lines.append(f"  ★ {quoted_col} ({col_type}) {nullable}")
                    else:
                        # ── OTHER COLUMN: name + type only (no description) ──
                        lines.append(f"  • {quoted_col} ({col_type}) {nullable}")

            except Exception as e:
                print(f"[CONTEXT AGENT] Error reading {full_name}: {e}")
                lines.append(f"  [Error reading {full_name}: {str(e)[:50]}]")

        return "\n".join(lines)
    
    # ═════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _clean_table_names(tables: List[str]) -> List[str]:
        """Remove 'table: ' or 'view: ' prefix."""
        clean = []
        for t in tables:
            if ": " in t:
                clean.append(t.split(": ", 1)[1])
            else:
                clean.append(t)
        return clean
    
    @staticmethod
    def _count_descriptions(metadata: Dict[str, Any]) -> int:
        """Count how many columns have descriptions in metadata."""
        count = 0
        for table_cols in metadata.values():
            for col_info in table_cols.values():
                if col_info.get("description"):
                    count += 1
        return count


# =============================================================================
# BARE SCHEMA — Lightweight schema for Pass 1 (no descriptions, no samples)
# =============================================================================

def get_bare_schema(
    user_engine,
    selected_tables: List[str],
    dialect: str
) -> str:
    """
    Lightweight schema for Pass 1: column names + types ONLY.
    No Opus descriptions, no sample values, no Supabase queries.
    
    This is what Pass 1 sees. Its job is column identification,
    not understanding business meaning — that comes in Pass 2.
    
    Token cost: ~60-70% less than get_full_schema_with_opus.
    """
    from sqlalchemy import inspect
    
    # Clean table names
    clean_tables = []
    for t in selected_tables:
        if ": " in t:
            clean_tables.append(t.split(": ", 1)[1])
        else:
            clean_tables.append(t)
    
    # Determine quote character
    if dialect.lower() == "postgresql":
        q, qr = '"', '"'
    elif dialect.lower() == "mysql":
        q, qr = '`', '`'
    elif dialect.lower() in ("mssql", "sqlserver"):
        q, qr = '[', ']'
    else:
        q, qr = '"', '"'
    
    inspector = inspect(user_engine)
    lines = [f"DATABASE: {dialect.upper()}"]
    
    for full_name in clean_tables:
        if "." in full_name:
            schema_name, table_name = full_name.split(".", 1)
        else:
            schema_name = None
            table_name = full_name
        
        try:
            columns = inspector.get_columns(table_name, schema=schema_name)
            
            if schema_name:
                quoted_table = f"{q}{schema_name}{qr}.{q}{table_name}{qr}"
            else:
                quoted_table = f"{q}{table_name}{qr}"
            
            lines.append(f"\n{quoted_table}:")
            
            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                quoted_col = f"{q}{col_name}{qr}"
                lines.append(f"  • {quoted_col} ({col_type}) {nullable}")
        
        except Exception as e:
            lines.append(f"  [Error reading {full_name}: {str(e)[:50]}]")
    
    return "\n".join(lines)


# =============================================================================
# MAIN — Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONTEXT AGENT MODULE")
    print("=" * 70)
    print()
    print("Flow:")
    print("  1. Pass 1 identifies columns (using BARE schema)")
    print("  2. Context Agent fetches:")
    print("     a. Opus descriptions for ONLY those columns")
    print("     b. Sample values + data types")
    print("     c. RAG rules relevant to question + columns")
    print("     d. Entity resolution (live DB)")
    print("  3. Pass 2 receives FOCUSED context bundle")
    print("  4. SQL Coder receives FOCUSED schema")
    print()
    print("Token savings: ~60-70% on schema context")
    print("=" * 70)
