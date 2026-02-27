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
        Rules are passed through as-is.
        
        Args:
            question: Original user question
            pass1_data: Parsed Pass 1 output with columns, tables, string_filter_columns
            rules_compressed: Pre-fetched business rules from Stage 2
        
        Returns:
            ContextBundle with focused schema, rules, metadata, resolver results
        """
        bundle = ContextBundle()
        bundle.rules_compressed = rules_compressed  # Pass through pre-fetched rules
        start_time = time.time()
        
        columns_by_table = pass1_data.get("columns", {})
        string_filter_columns = pass1_data.get("string_filter_columns", [])
        identified_tables = pass1_data.get("tables", [])
        
        if not columns_by_table:
            print("[CONTEXT AGENT] No columns identified by Pass 1 — returning empty bundle")
            bundle.total_time_ms = int((time.time() - start_time) * 1000)
            return bundle
        
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
        
        # ── Step 2: Build focused schema (only relevant columns with descriptions) ──
        stage_start = time.time()
        bundle.focused_schema = self._build_focused_schema(
            columns_by_table, bundle.metadata
        )
        bundle.stage_times["schema_build"] = int((time.time() - stage_start) * 1000)
        
        # Rules are pre-fetched in Stage 2 — count them for diagnostics
        try:
            rules_list = json.loads(bundle.rules_compressed)
            bundle.rules_retrieved = len(rules_list) if isinstance(rules_list, list) else 0
        except Exception:
            bundle.rules_retrieved = 0
        
        # ── Step 3: Entity resolution (live DB) ──
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
                string_filter_set.add((sf.get("table", ""), sf.get("column", "")))
        
        try:
            for table_name, columns in columns_by_table.items():
                metadata[table_name] = {}
                
                for col in columns:
                    try:
                        # Single query per column: gets data_type + samples + opus_description + user_description
                        result = self.vector_engine.table("schema_columns").select(
                            "column_name, data_type, sample_values, "
                            "opus_description, user_description, friendly_name"
                        ).eq("table_name", table_name).eq("column_name", col).execute()
                        
                        if result.data:
                            row = result.data[0]
                            sample_values = row.get("sample_values") or []
                            
                            if isinstance(sample_values, str):
                                try:
                                    sample_values = json.loads(sample_values)
                                except Exception:
                                    sample_values = []
                            
                            sample_values = sample_values[:10]
                            
                            # Priority: user_description > opus_description > friendly_name
                            description = (
                                row.get("user_description") or 
                                row.get("opus_description") or 
                                row.get("friendly_name") or ""
                            )
                            
                            needs_partial = (table_name, col) in string_filter_set
                            
                            metadata[table_name][col] = {
                                "data_type": row.get("data_type", "unknown"),
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
        metadata: Dict[str, Any]
    ) -> str:
        """
        Build schema text with descriptions for ONLY identified columns.
        
        SQL Coder needs to know column names, types, and quoting.
        But it only needs descriptions for the columns Pass 1 identified.
        
        Other columns in the table are listed with name+type only (so SQL Coder 
        doesn't accidentally reference non-existent columns), but without 
        expensive descriptions.
        """
        from sqlalchemy import inspect
        
        q = self.quote_char
        qr = ']' if q == '[' else q
        
        inspector = inspect(self.user_engine)
        
        # Build set of identified columns for quick lookup
        identified_cols = set()
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
