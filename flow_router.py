"""
Flow Router - Main Query Processing Orchestrator
==================================================
Routes queries through appropriate processing paths based on complexity.
Integrates all optimization modules.

Error Recovery Strategy:
  - Attempt 1: Reasoning LLM fixes the error (cheap, fast)
  - Attempt 2: Opus fixes the error (expensive, powerful)
  - Attempt 3: Return human-readable error explanation

Usage:
    from flow_router import process_query, FlowConfig
    
    result = process_query(question, engine, vector_engine, config)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import time


# =============================================================================
# DIALECT SYNTAX RULES (Concise version for SQL Coder)
# =============================================================================

def get_dialect_syntax_rules(dialect: str) -> str:
    """
    Get concise dialect-specific syntax rules for SQL Coder.
    Keeps token count low while providing essential syntax guidance.
    
    Args:
        dialect: Database dialect (postgresql, mysql, mssql, oracle, sqlite)
    
    Returns:
        Concise syntax rules string
    """
    rules = {
        'postgresql': '''SQL SYNTAX (PostgreSQL):
- Identifiers: "schema"."table", "column" (double quotes)
- Strings: 'value' (single quotes)
- Example: SELECT "Region", SUM("Margin") FROM "public"."SAP" WHERE "Status" = 'Active'
- ❌ WRONG: SELECT Region, `Margin`, [Status]''',

        'mysql': '''SQL SYNTAX (MySQL):
- Identifiers: `database`.`table`, `column` (backticks)
- Strings: 'value' (single quotes)
- Example: SELECT `Region`, SUM(`Margin`) FROM `mydb`.`SAP` WHERE `Status` = 'Active'
- ❌ WRONG: SELECT "Region", [Margin]''',

        'mssql': '''SQL SYNTAX (SQL Server):
- Identifiers: [schema].[table], [column] (square brackets)
- Strings: 'value' (single quotes)
- Example: SELECT [Region], SUM([Margin]) FROM [dbo].[SAP] WHERE [Status] = 'Active'
- ❌ WRONG: SELECT `Region`, "Margin"''',

        'oracle': '''SQL SYNTAX (Oracle):
- Identifiers: "schema"."table", "column" (double quotes, case-sensitive) OR UPPERCASE without quotes
- Strings: 'value' (single quotes)
- Example: SELECT "Region", SUM("Margin") FROM "sales"."SAP" WHERE "Status" = 'Active'
- ❌ WRONG: SELECT `Region`, [Margin]''',

        'sqlite': '''SQL SYNTAX (SQLite):
- Identifiers: "table", "column" (double quotes) - no schema
- Strings: 'value' (single quotes)
- Example: SELECT "Region", SUM("Margin") FROM "SAP" WHERE "Status" = 'Active'
- ❌ WRONG: SELECT [Region], public.SAP'''
    }
    
    return rules.get(dialect, rules['postgresql'])


# =============================================================================
# FULL SCHEMA WITH OPUS DESCRIPTIONS
# =============================================================================

def get_full_schema_with_opus(
    user_engine,
    vector_engine,
    selected_tables: List[str],
    dialect: str,
    include_opus: bool = True
) -> str:
    """
    Get full schema with descriptions (if available and enabled).
    User enrichment ALWAYS overrides Opus descriptions.
    
    Priority: user_description > opus_description > friendly_name
    
    Args:
        user_engine: User's database engine
        vector_engine: Supabase vector engine
        selected_tables: List of table names (may have "table: " or "view: " prefix)
        dialect: Database dialect
        include_opus: Whether to include descriptions
    
    Returns:
        Formatted schema string with descriptions
    """
    from sqlalchemy import inspect, text
    
    # Clean table names - remove "table: " or "view: " prefix if present
    clean_tables = []
    for t in selected_tables:
        if ": " in t:
            clean_tables.append(t.split(": ", 1)[1])
        else:
            clean_tables.append(t)
    
    print(f"[DEBUG] get_full_schema_with_opus called")
    print(f"[DEBUG] - selected_tables (raw): {selected_tables}")
    print(f"[DEBUG] - clean_tables: {clean_tables}")
    print(f"[DEBUG] - include_opus: {include_opus}")
    
    # Get descriptions from Supabase (user overrides opus)
    descriptions = {}
    if include_opus:
        try:
            with vector_engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT 
                            object_name, 
                            column_name, 
                            user_description,
                            opus_description,
                            friendly_name
                        FROM schema_columns
                        WHERE object_name = ANY(:tables)
                    """),
                    {"tables": clean_tables}
                )
                rows = result.fetchall()
                print(f"[DEBUG] - Found {len(rows)} rows in schema_columns")
                
                for row in rows:
                    key = f"{row[0]}.{row[1]}"
                    # Priority: user_description > opus_description > friendly_name
                    desc = row[2] or row[3] or row[4]
                    if desc:
                        descriptions[key] = desc
                        
                print(f"[DEBUG] - Descriptions found: {len(descriptions)}")
                if descriptions:
                    print(f"[DEBUG] - Sample descriptions: {list(descriptions.items())[:3]}")
                    
        except Exception as e:
            print(f"[DEBUG] - ERROR fetching descriptions: {e}")
    
    # Determine quote character based on dialect
    if dialect.lower() == "postgresql":
        q = '"'
    elif dialect.lower() == "mysql":
        q = '`'
    elif dialect.lower() in ("mssql", "sqlserver"):
        q = ['[', ']']  # Special case: different open/close
    else:
        q = '"'
    
    # Build schema with descriptions
    lines = [f"DATABASE: {dialect.upper()}"]
    
    inspector = inspect(user_engine)
    
    for full_name in clean_tables:
        if "." in full_name:
            schema_name, table_name = full_name.split(".", 1)
        else:
            schema_name = None
            table_name = full_name
        
        try:
            columns = inspector.get_columns(table_name, schema=schema_name)
            print(f"[DEBUG] - Table {full_name}: {len(columns)} columns")
            
            # Format table name with proper quoting for the dialect
            if isinstance(q, list):
                # MSSQL style: [schema].[table]
                if schema_name:
                    quoted_table = f"{q[0]}{schema_name}{q[1]}.{q[0]}{table_name}{q[1]}"
                else:
                    quoted_table = f"{q[0]}{table_name}{q[1]}"
            else:
                # PostgreSQL/MySQL style: "schema"."table" or `schema`.`table`
                if schema_name:
                    quoted_table = f"{q}{schema_name}{q}.{q}{table_name}{q}"
                else:
                    quoted_table = f"{q}{table_name}{q}"
            
            lines.append(f"\n{quoted_table}:")
            
            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                
                # Format column name with proper quoting
                if isinstance(q, list):
                    quoted_col = f"{q[0]}{col_name}{q[1]}"
                else:
                    quoted_col = f"{q}{col_name}{q}"
                
                # Check for description
                desc_key = f"{full_name}.{col_name}"
                desc = descriptions.get(desc_key)
                
                if desc:
                    # Include description (truncate if too long)
                    desc_short = desc[:120] if len(desc) > 120 else desc
                    lines.append(f"  • {quoted_col} ({col_type}) {nullable} - {desc_short}")
                else:
                    lines.append(f"  • {quoted_col} ({col_type}) {nullable}")
                    
        except Exception as e:
            print(f"[DEBUG] - ERROR reading {full_name}: {e}")
            lines.append(f"  [Error reading {full_name}: {str(e)[:50]}]")
    
    return "\n".join(lines)


def _get_all_active_rules(vector_engine) -> List[Dict[str, Any]]:
    """Load all active business rules (for static context-cache mode)."""
    from sqlalchemy import text

    rules = []
    with vector_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, rule_name, rule_type, rule_description, rule_data,
                       trigger_keywords, applies_to_tables, priority, is_mandatory
                FROM business_rules_v2
                WHERE is_active = TRUE
                ORDER BY priority ASC, id ASC
                """
            )
        ).fetchall()

    for r in rows:
        rules.append({
            "id": r[0],
            "rule_name": r[1],
            "rule_type": r[2],
            "rule_description": r[3],
            "rule_data": r[4],
            "trigger_keywords": r[5],
            "applies_to_tables": r[6],
            "priority": r[7],
            "is_mandatory": r[8],
            # aliases used by some downstream logic
            "keywords": r[5],
            "tables": r[6],
        })
    return rules




# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FlowConfig:
    """Configuration for query processing."""
    # Classification
    enable_classification: bool = True
    classification_provider: str = "groq"
    
    # RAG - Only Rule RAG now (Schema RAG removed - poor accuracy)
    enable_rule_rag: bool = True
    rule_rag_threshold: float = 0.65
    
    # Opus Descriptions (from one-time enrichment)
    enable_opus_descriptions: bool = True
    
    # Entity Resolver (live DB lookups at query time)
    enable_resolver: bool = True
    
    # Query Cache
    enable_cache: bool = True
    enable_semantic_cache: bool = True
    
    # Context Cache (reuse schema/rules across questions until versions change)
    enable_context_cache: bool = True
    context_cache_use_static_rules: bool = True

    # Prompt Caching (Anthropic cache_control on system prompt)
    # When True: static content (schema/rules/dialect) goes in system prompt with cache_control.
    # When False: everything is merged into the user message (no cache_control headers).
    enable_prompt_caching: bool = True

    # LLM Providers
    reasoning_provider: str = "claude_sonnet"
    sql_provider: str = "groq"
    
    # Opus Review — now used only for error fixing (second attempt)
    # enable_opus: True/False/"auto" controls whether Opus is ever used as reviewer
    # Error recovery always follows: attempt1=reasoning LLM, attempt2=opus
    enable_opus: str = "auto"  # "auto", True, False
    opus_provider: str = "claude_opus"
    max_retries: int = 3
    
    # Optimization
    compress_rules: bool = True
    validate_sql: bool = True
    auto_fix_sql: bool = True
    
    # Analyzer Agent (multi-step decomposition for analysis-complexity queries)
    enable_analyzer: bool = True

    # Dialect
    dialect: str = "postgresql"
    dialect_info: Dict = field(default_factory=lambda: {
        "dialect": "postgresql",
        "quote_char": '"',
        "string_quote": "'"
    })


def _zero_tok() -> Dict[str, int]:
    return {"input": 0, "output": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


@dataclass
class TokenUsage:
    """Track token usage across all stages."""
    classifier: Dict[str, int] = field(default_factory=_zero_tok)
    reasoning: Dict[str, int] = field(default_factory=_zero_tok)
    reasoning_pass1: Dict[str, int] = field(default_factory=_zero_tok)
    reasoning_pass2: Dict[str, int] = field(default_factory=_zero_tok)
    opus_complex: Dict[str, int] = field(default_factory=_zero_tok)
    sql_gen: Dict[str, int] = field(default_factory=_zero_tok)
    resolver: Dict[str, int] = field(default_factory=_zero_tok)  # DB query time (no LLM tokens)
    opus: Dict[str, int] = field(default_factory=_zero_tok)
    refinement: Dict[str, int] = field(default_factory=_zero_tok)
    # Tier 1: Static Validator mini-retry (SQL Coder)
    tier1_fix: Dict[str, int] = field(default_factory=_zero_tok)
    # Tier 2: Error Classifier mini-retry (SQL Coder)
    tier2_fix: Dict[str, int] = field(default_factory=_zero_tok)
    # Tier 3: Existing LLM error fixer
    error_fix_reasoning: Dict[str, int] = field(default_factory=_zero_tok)
    error_fix_opus: Dict[str, int] = field(default_factory=_zero_tok)
    chart: Dict[str, int] = field(default_factory=_zero_tok)
    # Analyzer Agent — multi-step decomposition (analysis-complexity queries)
    analyzer: Dict[str, int] = field(default_factory=_zero_tok)

    def total(self) -> Dict[str, int]:
        all_stages = [
            self.classifier, self.reasoning,
            self.reasoning_pass1, self.reasoning_pass2, self.opus_complex,
            self.sql_gen, self.opus, self.refinement,
            self.tier1_fix, self.tier2_fix,
            self.error_fix_reasoning, self.error_fix_opus,
            self.analyzer,
        ]
        return {
            "input":  sum(s["input"]  for s in all_stages),
            "output": sum(s["output"] for s in all_stages),
        }

    def total_tokens(self) -> int:
        t = self.total()
        return t["input"] + t["output"]

    def cache_creation_total(self) -> int:
        """Total tokens written to the Anthropic prompt cache across all stages."""
        all_stages = [
            self.classifier, self.reasoning,
            self.reasoning_pass1, self.reasoning_pass2, self.opus_complex,
            self.sql_gen, self.opus, self.refinement,
            self.tier1_fix, self.tier2_fix,
            self.error_fix_reasoning, self.error_fix_opus,
            self.analyzer,
        ]
        return sum(s.get("cache_creation_input_tokens", 0) for s in all_stages)

    def cache_read_total(self) -> int:
        """Total tokens read from the Anthropic prompt cache across all stages."""
        all_stages = [
            self.classifier, self.reasoning,
            self.reasoning_pass1, self.reasoning_pass2, self.opus_complex,
            self.sql_gen, self.opus, self.refinement,
            self.tier1_fix, self.tier2_fix,
            self.error_fix_reasoning, self.error_fix_opus,
            self.analyzer,
        ]
        return sum(s.get("cache_read_input_tokens", 0) for s in all_stages)


@dataclass
class LLMTrace:
    """Track LLM input/output for debugging."""
    classifier_input: str = ""
    classifier_output: str = ""
    reasoning_input: str = ""
    reasoning_output: str = ""
    reasoning_pass1_input: str = ""
    reasoning_pass1_output: str = ""
    reasoning_pass2_input: str = ""
    reasoning_pass2_output: str = ""
    opus_complex_input: str = ""
    opus_complex_output: str = ""
    resolver_summary: str = ""  # Not LLM trace — summary of DB queries run by resolver
    sql_gen_input: str = ""
    sql_gen_output: str = ""
    opus_input: str = ""
    opus_output: str = ""
    refinement_input: str = ""
    refinement_output: str = ""
    # Tier 1 mini-retry trace
    tier1_fix_input: str = ""
    tier1_fix_output: str = ""
    # Tier 2 mini-retry trace
    tier2_fix_input: str = ""
    tier2_fix_output: str = ""
    # Tier 3 (existing) traces
    error_fix_reasoning_input: str = ""
    error_fix_reasoning_output: str = ""
    error_fix_opus_input: str = ""
    error_fix_opus_output: str = ""


@dataclass
class QueryResult:
    """Result of query processing."""
    # Core results
    sql: str
    results: Any
    success: bool
    error: Optional[str] = None
    
    # Classification
    complexity: str = "medium"
    classification_reason: str = ""
    
    # Flow info
    flow_path: str = ""
    stages_completed: List[str] = field(default_factory=list)
    
    # Token tracking
    tokens: TokenUsage = field(default_factory=TokenUsage)
    
    # LLM I/O tracking for debugging
    llm_trace: LLMTrace = field(default_factory=LLMTrace)
    
    # Validation
    validation_result: Optional[Dict] = None
    sql_fixed: bool = False
    fixes_applied: List[str] = field(default_factory=list)
    
    # Opus review
    opus_review: Optional[Dict] = None
    opus_attempts: int = 0
    final_verdict: str = "NOT_REVIEWED"
    
    # Error recovery tracking
    error_recovery_attempted: bool = False
    error_recovery_method: str = ""  # "reasoning_llm" | "opus" | "none"
    original_error: Optional[str] = None  # The first error before any recovery
    original_sql: Optional[str] = None    # The SQL before error recovery fixed it
    opus_blocked_soft: bool = False    # Opus flagged logic issue but data exists — warn don't block
    # Context used
    rules_retrieved: int = 0
    columns_retrieved: int = 0
    schema_text: str = ""
    rules_compressed: str = ""
    
    # Cache
    cache_hit: bool = False
    cache_hit_type: str = ""  # "memory", "exact", "semantic"
    cache_key: str = ""
    context_cache_hit: bool = False
    context_cache_key: str = ""
    
    # Two-pass reasoning metadata (stored for error retry reuse)
    column_metadata: Dict = field(default_factory=dict)
    pass2_plan: str = ""
    
    # Entity Resolver
    resolver_result: Any = None  # ResolverResult object
    entities_resolved: int = 0
    resolver_time_ms: int = 0
    
    # Analyzer Agent result (populated for analysis-complexity queries)
    analyzer_data: Any = None

    # Timing
    total_time_ms: int = 0
    stage_times: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# PROMPT CACHING HELPERS
# =============================================================================

def _is_claude_provider(provider: str) -> bool:
    """Return True when the provider is a Claude model (supports prompt caching)."""
    return provider.startswith("claude_")


def _build_static_system_prompt(
    dialect_syntax: str = "",
    schema: str = "",
    rules: str = "",
    extra_instructions: str = "",
) -> str:
    """
    Build a cacheable system prompt from static content.

    Claude caches system prompts byte-for-byte — so the output of this function
    must be identical across calls that should share a cache entry.

    Args:
        dialect_syntax:     get_dialect_syntax_rules() output (static)
        schema:             Schema text (stable within a session)
        rules:              Compressed rules (may vary by RAG result)
        extra_instructions: Any static model-role instructions

    Returns:
        Combined system prompt string, or "" if all inputs are empty.
    """
    parts = []
    if extra_instructions:
        parts.append(extra_instructions)
    if dialect_syntax:
        parts.append(dialect_syntax)
    if schema:
        parts.append(f"SCHEMA:\n{schema}")
    if rules and rules.strip() and rules.strip() not in ("[]", ""):
        parts.append(f"BUSINESS RULES:\n{rules}")
    return "\n\n".join(parts)


def _accumulate_cache_tokens(result_tokens_dict: Dict, llm_token_dict: Dict) -> None:
    """
    Copy cache token counts from an LLM response token dict into a stage's token dict.
    Called after every Claude call to track cache creation / read tokens.
    """
    result_tokens_dict["cache_creation_input_tokens"] = (
        result_tokens_dict.get("cache_creation_input_tokens", 0)
        + llm_token_dict.get("cache_creation_input_tokens", 0)
    )
    result_tokens_dict["cache_read_input_tokens"] = (
        result_tokens_dict.get("cache_read_input_tokens", 0)
        + llm_token_dict.get("cache_read_input_tokens", 0)
    )


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_query(
    question: str,
    engine,  # User's database engine
    vector_engine,  # Vector database engine (Supabase)
    selected_tables: List[str],
    config: FlowConfig = None
) -> QueryResult:
    """
    Main query processing function.
    Routes query through appropriate path based on complexity.
    
    Error Recovery Flow:
        SQL fails → Attempt 1: Reasoning LLM fix (cheap)
                 → Still fails → Attempt 2: Opus fix (powerful)
                 → Still fails → Return error with explanation
    """
    from query_classifier import classify_query, get_flow_config
    from schema_rag import get_relevant_schema, get_relevant_schema_simple, format_schema_for_llm, get_full_schema
    from prompt_optimizer import (
        compress_rules_for_llm,
        create_easy_query_prompt,
        create_medium_query_prompt,
        create_hard_query_prompt,
        extract_sql_from_response,
        extract_columns_from_sql
    )
    from sql_validator import validate_sql, is_safe_query, fix_common_issues
    from llm_v2 import call_llm
    from db import run_sql
    
    if config is None:
        config = FlowConfig()
    
    result = QueryResult(sql="", results=None, success=False)
    start_time = time.time()
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 0: CHECK CACHE (if enabled)
        # ═══════════════════════════════════════════════════════════════════
        if config.enable_cache:
            try:
                from query_cache import QueryCache, compute_versions
                
                cache = QueryCache(
                    vector_engine=vector_engine,
                    enabled=True,
                    use_semantic=config.enable_semantic_cache
                )
                
                schema_version = QueryCache.compute_schema_version(
                    {t: [] for t in selected_tables} if selected_tables else {}
                )
                
                rules_for_version = []
                if config.enable_rule_rag:
                    try:
                        from vector_utils_v2 import get_relevant_context
                        context = get_relevant_context(vector_engine, question, enable_vector_search=True)
                        rules_for_version = context.get("rules", [])
                    except:
                        pass
                
                rules_version = QueryCache.compute_rules_version(rules_for_version)
                
                cached = cache.get(
                    question=question,
                    schema_version=schema_version,
                    rules_version=rules_version,
                    dialect=config.dialect
                )
                
                if cached:
                    print(f"[CACHE] Hit! Type: {cached.get('hit_type', 'unknown')}")
                    result.sql = cached["sql"]
                    result.cache_hit = True
                    result.cache_hit_type = cached.get("hit_type", "")
                    result.cache_key = cached.get("cache_key", "")
                    result.complexity = cached.get("complexity", "cached")
                    result.flow_path = f"CACHED ({cached.get('hit_type', 'exact')}): No LLM calls"
                    result.stages_completed.append("cache_hit")
                    
                    try:
                        query_result = run_sql(engine, result.sql)
                        result.results = query_result
                        result.success = True
                    except Exception as e:
                        result.error = str(e)
                        result.success = False
                        cache.invalidate(cache_key=result.cache_key)
                    
                    result.total_time_ms = int((time.time() - start_time) * 1000)
                    return result
                else:
                    print(f"[CACHE] Miss")
                    
            except ImportError:
                print("[CACHE] Module not available")
            except Exception as e:
                print(f"[CACHE] Error: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1: CLASSIFICATION
        # ═══════════════════════════════════════════════════════════════════
        stage_start = time.time()
        
        if config.enable_classification:
            classification = classify_query(
                question,
                use_llm=True,
                llm_provider=config.classification_provider
            )
            result.complexity = classification["complexity"]
            result.classification_reason = classification["reason"]
            result.tokens.classifier = classification["tokens"]
            result.llm_trace.classifier_input = classification.get("prompt", "")
            result.llm_trace.classifier_output = classification.get("response", "")
        else:
            result.complexity = "medium"
            result.classification_reason = "Classification disabled"
        
        result.stages_completed.append("classification")
        result.stage_times["classification"] = int((time.time() - stage_start) * 1000)
        
        flow_cfg = get_flow_config(result.complexity)
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2: BARE SCHEMA + RAG RULES
        # ─────────────────────────────────────────────────────────────────
        # Pass 1 needs: column names + types + business rules
        # Pass 1 does NOT need: Opus descriptions (deferred to Context Agent)
        # 
        # Why rules matter for Pass 1: user says "total sales" and DB has
        # both "Margin" and "Bottomline" columns. Business rule maps
        # "sales" → SUM("Margin"). Without rules, Pass 1 guesses wrong.
        #
        # What's deferred to Context Agent (after Pass 1):
        # - Opus descriptions (only for identified columns)
        # - Sample values  
        # - Entity resolution (live DB)
        # ═══════════════════════════════════════════════════════════════════
        stage_start = time.time()
        
        from context_agent import get_bare_schema, ContextAgent
        from context_cache import GLOBAL_CONTEXT_CACHE, ContextBundle, ContextCache

        rules_context = []
        rules_compressed = "[]"
        examples = []

        schema_version = ContextCache.compute_schema_version(engine, selected_tables)
        rules_version = ContextCache.compute_rules_version(vector_engine) if config.enable_rule_rag else "no_rules"
        context_cache_key = ContextCache.make_key(
            selected_tables=selected_tables,
            dialect=config.dialect,
            include_opus=config.enable_opus_descriptions,
            schema_version=schema_version,
            rules_version=rules_version,
        )
        result.context_cache_key = context_cache_key

        cached_bundle = GLOBAL_CONTEXT_CACHE.get(context_cache_key) if config.enable_context_cache else None

        if cached_bundle:
            bare_schema = cached_bundle.bare_schema
            schema_text = cached_bundle.schema_text
            rules_compressed = cached_bundle.rules_compressed
            result.rules_retrieved = cached_bundle.rules_retrieved
            result.context_cache_hit = True
            print(f"[STAGE2] Context cache hit: {context_cache_key}")
        else:
            bare_schema = get_bare_schema(engine, selected_tables, config.dialect)

            # Full schema still needed for SIMPLE flow (no Pass 1) and error recovery
            schema_text = get_full_schema_with_opus(
                engine,
                vector_engine,
                selected_tables,
                config.dialect,
                include_opus=config.enable_opus_descriptions
            )

            if config.enable_rule_rag:
                try:
                    if config.context_cache_use_static_rules:
                        rules_context = _get_all_active_rules(vector_engine)
                    else:
                        from vector_utils_v2 import get_relevant_context
                        context = get_relevant_context(
                            vector_engine,
                            question,
                            enable_vector_search=True,
                            similarity_threshold=config.rule_rag_threshold
                        )
                        rules_context = context.get("rules", [])

                    result.rules_retrieved = len(rules_context)
                    print(f"[STAGE2] Rules retrieved: {len(rules_context)}")
                    if rules_context:
                        print(f"[STAGE2] Rule names: {[r.get('rule_name', 'unknown') for r in rules_context[:5]]}")

                    examples = [r for r in rules_context if r.get("rule_type") in ["example", "query_example"]]

                    if config.compress_rules:
                        rules_compressed = compress_rules_for_llm(rules_context)
                    else:
                        from prompt_optimizer import safe_json_dumps
                        rules_compressed = safe_json_dumps(rules_context)

                except Exception as e:
                    print(f"[STAGE2] RAG Error: {e}")
                    import traceback
                    traceback.print_exc()
                    rules_compressed = "[]"
                    result.rules_retrieved = 0
            else:
                result.rules_retrieved = 0
                rules_compressed = "[]"

            # write context cache only for static-rule mode (question agnostic)
            if config.enable_context_cache and (config.context_cache_use_static_rules or not config.enable_rule_rag):
                GLOBAL_CONTEXT_CACHE.set(
                    context_cache_key,
                    ContextBundle(
                        schema_version=schema_version,
                        rules_version=rules_version,
                        bare_schema=bare_schema,
                        schema_text=schema_text,
                        rules_compressed=rules_compressed,
                        rules_retrieved=result.rules_retrieved,
                    ),
                )
                print(f"[STAGE2] Context cache stored: {context_cache_key}")

        result.schema_text = schema_text
        result.columns_retrieved = -1

        # ── Skip auto_apply date rules if user specified explicit period ──
        explicit_date_keywords = [
            "rolling", "last ", "past ", "previous ",
            "months", "weeks", "days", "since ", "between ",
            "year to date", "ytd", "quarter to date", "qtd",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan ", "feb ", "mar ", "apr ", "jun ", "jul ", "aug ",
            "sep ", "oct ", "nov ", "dec ", "fy2", "20"
        ]

        question_lower = question.lower()
        user_specified_date = any(kw in question_lower for kw in explicit_date_keywords)

        if user_specified_date:
            try:
                rules_list = json.loads(rules_compressed)
                filtered_rules = []
                skipped = []

                for rule in rules_list:
                    rule_data = rule.get("data", rule.get("rule_data", {}))
                    is_auto_date = (
                        isinstance(rule_data, dict)
                        and rule_data.get("auto_apply") is True
                        and any(kw in str(rule_data).lower() for kw in ["date", "month", "year", "fy", "financial", "period"])
                    )
                    if is_auto_date:
                        skipped.append(rule.get("name", rule.get("rule_name", "unknown")))
                    else:
                        filtered_rules.append(rule)

                if skipped:
                    rules_compressed = json.dumps(filtered_rules)
                    print(f"[STAGE2] Date override — skipped auto_apply: {skipped}")
            except Exception as e:
                print(f"[STAGE2] Date override filter error: {e}")
        
        result.rules_compressed = rules_compressed
        result.stages_completed.append("schema_and_rag")
        result.stage_times["schema_and_rag"] = int((time.time() - stage_start) * 1000)
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 3: GENERATE SQL — 4-level routing
        # ─────────────────────────────────────────────────────────────────
        # SIMPLE      → SQL Coder (Qwen) only
        # ANALYTICAL  → Reasoning Pass1 → Metadata → Reasoning Pass2 → SQL Coder
        # COMPARATIVE → Same as ANALYTICAL + Opus Review (handled in Stage 7)
        # COMPLEX     → Reasoning Pass1 → Metadata → Opus single call
        # Legacy easy/medium/hard mapped via get_flow_config()
        # ═══════════════════════════════════════════════════════════════════
        stage_start = time.time()

        from reasoning_prompts import (
            create_pass1_prompt,
            fetch_column_metadata,
            create_pass2_prompt,
            create_opus_complex_prompt,
            parse_pass1_output,
            parse_pass2_output
        )

        dialect_syntax = get_dialect_syntax_rules(config.dialect_info.get('dialect', 'postgresql'))
        dialect_name_upper = config.dialect_info.get('dialect', 'postgresql').upper()
        quote_char = config.dialect_info.get('quote_char', '"')

        # Normalise legacy complexity values
        complexity_norm = result.complexity
        legacy_map = {"easy": "simple", "medium": "analytical", "hard": "comparative", "analysis": "analysis"}
        complexity_norm = legacy_map.get(complexity_norm, complexity_norm)

        # If analysis complexity but Analyzer Agent is disabled, fall back to hard flow
        if complexity_norm == "analysis" and not config.enable_analyzer:
            print(f"[STAGE3] Analyzer Agent disabled — routing analysis query as comparative")
            complexity_norm = "comparative"

        if complexity_norm == "simple":
            # ── SIMPLE: SQL Coder only (schema + rules already fetched in Stage 2) ──
            result.flow_path = "SIMPLE: Question → Schema + RAG → SQL Coder → Execute"
            print(f"[STAGE3] SIMPLE flow")

            # Prompt caching: schema + rules + dialect → system; question → user
            _sql_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=schema_text,
                rules=rules_compressed,
                extra_instructions=f"You are an expert {dialect_name_upper} SQL generator. "
                                   f"Output only valid SQL — no explanation.",
            )
            _use_sql_sys = _is_claude_provider(config.sql_provider) and config.enable_prompt_caching
            sql_prompt = f"QUESTION: {question}\n\nOUTPUT: Only the SQL query. Start with SELECT or WITH."
            if _is_claude_provider(config.sql_provider) and not config.enable_prompt_caching:
                sql_prompt = f"{_sql_system}\n\n{sql_prompt}"
            sql_response, sql_tokens = call_llm(
                sql_prompt, config.sql_provider,
                system_prompt=_sql_system if _use_sql_sys else None,
            )
            result.tokens.sql_gen = sql_tokens
            _accumulate_cache_tokens(result.tokens.sql_gen, sql_tokens)
            result.sql = extract_sql_from_response(sql_response)
            result.llm_trace.sql_gen_input = sql_prompt
            result.llm_trace.sql_gen_output = sql_response

        elif complexity_norm in ("analytical", "comparative"):
            # ── ANALYTICAL / COMPARATIVE: Pass 1 (bare + rules) → Context Agent → Pass 2 → SQL Coder ─
            result.flow_path = f"{complexity_norm.upper()}: Question → Bare Schema + Rules → Pass 1 → Context Agent [descs + samples + resolver] → Pass 2 → SQL Coder → Execute"
            print(f"[STAGE3] {complexity_norm.upper()} flow — starting Pass 1 with bare schema + rules")

            # Pass 1: Column identification using BARE schema + business rules
            # Rules are critical — they map business terms to column names
            # (e.g., "sales" → SUM("Margin"), not "Bottomline")
            # Descriptions are NOT needed here — deferred to Context Agent
            # Prompt caching: schema + rules → system; question → user message
            _p1_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=bare_schema,
                rules=rules_compressed,
                extra_instructions=(
                    f"You are a SQL query planner for {dialect_name_upper}. "
                    f"Your task is to identify which tables and columns are needed.\n"
                    f"COLUMN FORMAT: {quote_char}column_name{quote_char}"
                ),
            )
            _p1_user = (
                f"QUESTION: {question}\n\n"
                f"Identify all tables and columns needed. Return JSON only:\n"
                f'{{"tables": [...], "columns": {{"table": ["col1",...]}}, '
                f'"string_filter_columns": [{{"table": "t", "column": "c", "user_value": "v", "filter_type": "include"}}], '
                f'"joins_needed": true/false}}\n\n'
                f"string_filter_columns: only columns where the user supplied a human-readable value "
                f"that may not match exactly (names, categories, codes). "
                f"Each entry MUST be a dict with keys: table, column, user_value, filter_type."
            )
            prefill = "{" if _is_claude_provider(config.reasoning_provider) else None
            if _is_claude_provider(config.reasoning_provider) and config.enable_prompt_caching:
                pass1_response, pass1_tokens = call_llm(
                    _p1_user, config.reasoning_provider,
                    prefill=prefill, system_prompt=_p1_system,
                )
                result.llm_trace.reasoning_pass1_input = _p1_user
            else:
                pass1_prompt = create_pass1_prompt(
                    question=question, schema=bare_schema,
                    rules=rules_compressed, dialect_info=config.dialect_info,
                )
                pass1_response, pass1_tokens = call_llm(
                    pass1_prompt, config.reasoning_provider, prefill=prefill,
                )
                result.llm_trace.reasoning_pass1_input = pass1_prompt
            result.tokens.reasoning_pass1 = pass1_tokens
            _accumulate_cache_tokens(result.tokens.reasoning_pass1, pass1_tokens)
            result.llm_trace.reasoning_pass1_output = pass1_response
            print(f"[STAGE3] Pass 1 complete — {pass1_tokens.get('input',0)+pass1_tokens.get('output',0)} tokens")

            # ── CONTEXT AGENT: Focused retrieval for ONLY identified columns ──
            pass1_data = parse_pass1_output(pass1_response)
            
            agent = ContextAgent(
                user_engine=engine,
                vector_engine=vector_engine,
                selected_tables=selected_tables,
                dialect_info=config.dialect_info,
                enable_resolver=config.enable_resolver,
            )
            
            bundle = agent.fetch_context(
                question=question,
                pass1_data=pass1_data,
                rules_compressed=rules_compressed
            )
            
            # Store agent results
            result.column_metadata = bundle.metadata
            result.rules_compressed = bundle.rules_compressed
            result.rules_retrieved = bundle.rules_retrieved
            result.resolver_result = bundle.resolver_result
            result.entities_resolved = bundle.entities_resolved
            result.resolver_time_ms = bundle.resolver_result.total_time_ms if bundle.resolver_result else 0
            result.llm_trace.resolver_summary = bundle.resolver_text
            result.stages_completed.append("context_agent")
            result.stage_times["context_agent"] = bundle.total_time_ms

            print(f"[STAGE3] Context Agent complete — {bundle.total_time_ms}ms | "
                  f"descs:{bundle.opus_descriptions_fetched} rules:{bundle.rules_retrieved} "
                  f"entities:{bundle.entities_resolved}")

            # Pass 2: Full plan with FOCUSED context
            # Prompt caching: rules + dialect → system; question + metadata → user
            _p2_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=bare_schema,
                rules=bundle.rules_compressed,
                extra_instructions=(
                    f"You are a SQL query planner for {dialect_name_upper}. "
                    f"Produce a detailed query plan — no SQL yet."
                ),
            )
            if _is_claude_provider(config.reasoning_provider) and config.enable_prompt_caching:
                pass2_prompt = create_pass2_prompt(
                    question=question,
                    pass1_output=pass1_response,
                    metadata=bundle.metadata,
                    dialect_info=config.dialect_info,
                    resolver_text=bundle.resolver_text,
                    rules="",  # already in system_prompt to avoid duplication
                )
                pass2_response, pass2_tokens = call_llm(
                    pass2_prompt, config.reasoning_provider,
                    prefill=prefill, system_prompt=_p2_system,
                )
            else:
                pass2_prompt = create_pass2_prompt(
                    question=question,
                    pass1_output=pass1_response,
                    metadata=bundle.metadata,
                    dialect_info=config.dialect_info,
                    resolver_text=bundle.resolver_text,
                    rules=bundle.rules_compressed,
                )
                pass2_response, pass2_tokens = call_llm(
                    pass2_prompt, config.reasoning_provider, prefill=prefill,
                )
            result.tokens.reasoning_pass2 = pass2_tokens
            _accumulate_cache_tokens(result.tokens.reasoning_pass2, pass2_tokens)
            result.llm_trace.reasoning_pass2_input = pass2_prompt
            result.llm_trace.reasoning_pass2_output = pass2_response
            result.pass2_plan = pass2_response
            print(f"[STAGE3] Pass 2 complete — {pass2_tokens.get('input',0)+pass2_tokens.get('output',0)} tokens")

            # SQL Coder: receives Pass 2 plan + FOCUSED schema (only relevant cols have descriptions)
            # Business rules are intentionally excluded — the Pass 2 plan already encodes every
            # filter condition explicitly, so repeating rules here wastes tokens with no benefit.
            sql_prompt = f"""Generate a {dialect_name_upper} SQL query from this plan.

{dialect_syntax}

SCHEMA:
{bundle.focused_schema}

QUERY PLAN (implement this exactly — filters are pre-decided, do not change them):
{pass2_response}

CRITICAL:
1. Use {quote_char} for ALL identifiers
2. Implement every filter from the plan EXACTLY as written
3. Do not substitute exact match for ILIKE or vice versa
4. Output SQL only — no explanation

OUTPUT: Only the SQL query. Start with SELECT or WITH."""

            sql_response, sql_tokens = call_llm(sql_prompt, config.sql_provider)
            result.tokens.sql_gen = sql_tokens
            result.sql = extract_sql_from_response(sql_response)
            result.llm_trace.sql_gen_input = sql_prompt
            result.llm_trace.sql_gen_output = sql_response

            # Keep legacy reasoning field populated for backward compat
            result.tokens.reasoning = pass1_tokens

        elif complexity_norm == "complex":
            # ── COMPLEX: Pass 1 (bare + rules) → Context Agent → Opus single call ─
            result.flow_path = "COMPLEX: Question → Bare Schema + Rules → Pass 1 → Context Agent [descs + samples + resolver] → Opus → Execute"
            print(f"[STAGE3] COMPLEX flow — Opus handles reasoning + SQL")

            # Quick Pass 1 using bare schema + rules (with caching for Claude)
            _cp1_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=bare_schema,
                rules=rules_compressed,
                extra_instructions=(
                    f"You are a SQL query planner for {dialect_name_upper}. "
                    f"Identify which tables and columns are needed.\n"
                    f"COLUMN FORMAT: {quote_char}column_name{quote_char}"
                ),
            )
            _cp1_user = (
                f"QUESTION: {question}\n\n"
                f"Identify all tables and columns needed. Return JSON only:\n"
                f'{{"tables": [...], "columns": {{"table": ["col1",...]}}, '
                f'"string_filter_columns": [{{"table": "t", "column": "c", "user_value": "v", "filter_type": "include"}}], '
                f'"joins_needed": true/false}}\n\n'
                f"string_filter_columns: only columns where the user supplied a human-readable value "
                f"that may not match exactly (names, categories, codes). "
                f"Each entry MUST be a dict with keys: table, column, user_value, filter_type."
            )
            prefill = "{" if _is_claude_provider(config.reasoning_provider) else None
            if _is_claude_provider(config.reasoning_provider) and config.enable_prompt_caching:
                pass1_response, pass1_tokens = call_llm(
                    _cp1_user, config.reasoning_provider,
                    prefill=prefill, system_prompt=_cp1_system,
                )
                result.llm_trace.reasoning_pass1_input = _cp1_user
            else:
                pass1_prompt = create_pass1_prompt(
                    question=question, schema=bare_schema,
                    rules=rules_compressed, dialect_info=config.dialect_info,
                )
                pass1_response, pass1_tokens = call_llm(
                    pass1_prompt, config.reasoning_provider, prefill=prefill,
                )
                result.llm_trace.reasoning_pass1_input = pass1_prompt
            result.tokens.reasoning_pass1 = pass1_tokens
            _accumulate_cache_tokens(result.tokens.reasoning_pass1, pass1_tokens)
            result.llm_trace.reasoning_pass1_output = pass1_response

            # ── CONTEXT AGENT ──
            pass1_data = parse_pass1_output(pass1_response)
            
            agent = ContextAgent(
                user_engine=engine,
                vector_engine=vector_engine,
                selected_tables=selected_tables,
                dialect_info=config.dialect_info,
                enable_resolver=config.enable_resolver,
            )
            
            bundle = agent.fetch_context(
                question=question,
                pass1_data=pass1_data,
                rules_compressed=rules_compressed
            )
            
            result.column_metadata = bundle.metadata
            result.rules_compressed = bundle.rules_compressed
            result.rules_retrieved = bundle.rules_retrieved
            result.resolver_result = bundle.resolver_result
            result.entities_resolved = bundle.entities_resolved
            result.resolver_time_ms = bundle.resolver_result.total_time_ms if bundle.resolver_result else 0
            result.llm_trace.resolver_summary = bundle.resolver_text
            result.stages_completed.append("context_agent")
            result.stage_times["context_agent"] = bundle.total_time_ms

            # Opus single call: FOCUSED schema + rules + metadata + resolutions → SQL
            _opus_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=bundle.focused_schema,
                rules=bundle.rules_compressed,
                extra_instructions=(
                    f"You are an expert {dialect_name_upper} SQL analyst. "
                    f"Analyze the question and generate a single correct SQL query."
                ),
            )
            _use_opus_cache = _is_claude_provider(config.opus_provider) and config.enable_prompt_caching
            opus_prompt = create_opus_complex_prompt(
                question=question,
                schema="" if _use_opus_cache else bundle.focused_schema,
                rules="" if _use_opus_cache else bundle.rules_compressed,
                metadata=bundle.metadata,
                dialect_info=config.dialect_info,
                resolver_text=bundle.resolver_text
            )
            opus_response, opus_complex_tokens = call_llm(
                opus_prompt, config.opus_provider,
                system_prompt=_opus_system if _use_opus_cache else None,
            )
            result.tokens.opus_complex = opus_complex_tokens
            _accumulate_cache_tokens(result.tokens.opus_complex, opus_complex_tokens)
            result.llm_trace.opus_complex_input = opus_prompt
            result.llm_trace.opus_complex_output = opus_response
            result.sql = extract_sql_from_response(opus_response)
            result.pass2_plan = opus_response
            print(f"[STAGE3] COMPLEX Opus call complete — {opus_complex_tokens.get('input',0)+opus_complex_tokens.get('output',0)} tokens")

            # Legacy reasoning token compat
            result.tokens.reasoning = pass1_tokens

        elif complexity_norm == "analysis":
            # ── ANALYSIS: multi-step Analyzer Agent ──────────────────────────
            # Decomposes the question into sub-queries, executes each through
            # the existing pipeline, and synthesizes a final answer.
            # If enable_analyzer is False the normalization block above already
            # remapped complexity_norm to "comparative", so this branch only
            # runs when the Analyzer Agent is enabled.
            result.flow_path = "ANALYSIS: Analyzer Agent → Sub-query decomposition → Synthesis → Opus Review"
            print(f"[STAGE3] ANALYSIS flow via SQLAnalyzer")
            from sql_analyzer import SQLAnalyzer
            analyzer = SQLAnalyzer(
                engine=engine,
                vector_engine=vector_engine,
                selected_tables=selected_tables,
                config=config,
                schema_text=schema_text,
                bare_schema=bare_schema,
                rules_compressed=rules_compressed,
                dialect_info=config.dialect_info,
            )
            analyzer_result = analyzer.analyze(question)
            result.sql = analyzer_result.synthesis_sql
            result.results = analyzer_result.final_results
            result.success = bool(analyzer_result.synthesis_sql and analyzer_result.final_results is not None)
            result.opus_verdict = analyzer_result.opus_verdict
            # Store full analyzer result for per-stage token display and debug tab
            result.analyzer_data = analyzer_result
            # Accumulate all analyzer stage tokens into the dedicated analyzer bucket
            for stage_tok in analyzer_result.total_tokens.values():
                result.tokens.analyzer["input"] += stage_tok.get("input", 0)
                result.tokens.analyzer["output"] += stage_tok.get("output", 0)

        else:
            # Fallback — treat as analytical
            result.flow_path = "ANALYTICAL (fallback): Question → RAG → Reasoning → SQL Coder → Execute"
            reasoning_prompt = create_hard_query_prompt(
                question, schema_text, rules_compressed,
                examples=examples[:3],
                dialect_info=config.dialect_info,
                output_type="analysis"
            )
            _fb_system = _build_static_system_prompt(
                dialect_syntax=dialect_syntax,
                schema=schema_text,
                rules=rules_compressed,
            ) if (_is_claude_provider(config.reasoning_provider) and config.enable_prompt_caching) else None
            prefill = "{" if _is_claude_provider(config.reasoning_provider) else None
            reasoning_response, reasoning_tokens = call_llm(
                reasoning_prompt, config.reasoning_provider,
                prefill=prefill,
                system_prompt=_fb_system,
            )
            result.tokens.reasoning = reasoning_tokens
            _accumulate_cache_tokens(result.tokens.reasoning, reasoning_tokens)
            result.llm_trace.reasoning_input = reasoning_prompt
            result.llm_trace.reasoning_output = reasoning_response

            sql_prompt = f"""Generate a {dialect_name_upper} SQL query based on this analysis.
ANALYSIS: {reasoning_response}
SCHEMA: {schema_text}
{dialect_syntax}
OUTPUT: Only the SQL query. Start with SELECT or WITH."""
            sql_response, sql_tokens = call_llm(sql_prompt, config.sql_provider)
            result.tokens.sql_gen = sql_tokens
            result.sql = extract_sql_from_response(sql_response)
            result.llm_trace.sql_gen_input = sql_prompt
            result.llm_trace.sql_gen_output = sql_response
        
        result.stages_completed.append("sql_generation")
        result.stage_times["sql_generation"] = int((time.time() - stage_start) * 1000)

        # ═══════════════════════════════════════════════════════════════════
        # TIER 1: STATIC VALIDATOR (pre-execution, no LLM)
        # ─────────────────────────────────────────────────────────────────
        # Checks SQL for type mismatches, GROUP BY issues, quote style,
        # date vs text comparisons, and missing mandatory filters.
        # If issues found → one mini-retry with SQL Coder (small prompt).
        # One attempt only — proceed to Stage 4 regardless of outcome.
        # ═══════════════════════════════════════════════════════════════════
        if result.sql:
            try:
                from sql_static_validator import validate_sql_static, build_tier1_mini_retry_prompt
                _tier1_col_meta = result.column_metadata or {}
                _dialect_quote_chars = {
                    "postgresql": '"', "mysql": "`", "mssql": "[",
                    "oracle": '"', "sqlite": '"',
                }
                _tier1_quote_char = _dialect_quote_chars.get(config.dialect, '"')

                tier1_issues = validate_sql_static(
                    sql=result.sql,
                    column_metadata=_tier1_col_meta,
                    dialect=config.dialect,
                    mandatory_columns=[],  # populated by rule_column_dependencies if available
                )

                if tier1_issues:
                    print(f"[TIER 1] Static validator found {len(tier1_issues)} issue(s). Mini-retry with SQL Coder.")
                    for _iss in tier1_issues:
                        print(f"  [{_iss.issue_type}] {_iss.detail}")

                    _t1_prompt = build_tier1_mini_retry_prompt(
                        original_sql=result.sql,
                        issues=tier1_issues,
                        dialect=config.dialect,
                        quote_char=_tier1_quote_char,
                    )
                    _t1_response, _t1_tokens = call_llm(_t1_prompt, config.sql_provider)
                    _t1_fixed = extract_sql_from_response(_t1_response)

                    result.tokens.tier1_fix = _t1_tokens
                    result.llm_trace.tier1_fix_input = _t1_prompt
                    result.llm_trace.tier1_fix_output = _t1_response

                    if _t1_fixed and _t1_fixed != result.sql:
                        result.sql = _t1_fixed
                        result.sql_fixed = True
                        result.fixes_applied.append("tier1_static_validator")
                        result.stages_completed.append("tier1_fix_applied")
                        print(f"[TIER 1] ✅ SQL Coder applied fix.")
                    else:
                        result.stages_completed.append("tier1_fix_unchanged")
                        print(f"[TIER 1] SQL Coder returned same SQL — proceeding anyway.")
                else:
                    print(f"[TIER 1] Static validator: no issues found.")
                    result.stages_completed.append("tier1_clean")
            except Exception as _t1_err:
                print(f"[TIER 1] Static validator error (non-fatal): {_t1_err}")

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 4: VALIDATE + AUTO-FIX SQL
        # ═══════════════════════════════════════════════════════════════════
        stage_start = time.time()
        
        if config.validate_sql and result.sql:
            validation = validate_sql(result.sql, config.dialect, available_tables=selected_tables)
            result.validation_result = {
                "is_valid": validation.is_valid,
                "issues": validation.issues,
                "warnings": validation.warnings,
                "severity": validation.severity
            }
            
            if config.auto_fix_sql and not validation.is_valid:
                fixed_sql, fixes = fix_common_issues(result.sql, config.dialect)
                if fixes:
                    result.sql = fixed_sql
                    result.sql_fixed = True
                    result.fixes_applied = fixes
        
        result.stages_completed.append("validation")
        result.stage_times["validation"] = int((time.time() - stage_start) * 1000)
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 5: EXECUTE SQL
        # ═══════════════════════════════════════════════════════════════════
        stage_start = time.time()
        
        is_safe, safety_reason = is_safe_query(result.sql)
        
        if not is_safe:
            result.success = False
            result.error = f"Query blocked: {safety_reason}"
            result.stages_completed.append("execution_blocked")
            result.stage_times["execution"] = int((time.time() - stage_start) * 1000)
        else:
            try:
                result.results = run_sql(engine, result.sql)
                result.success = True
                result.stages_completed.append("execution")
            except Exception as e:
                result.success = False
                result.error = str(e)
                result.original_error = str(e)
                result.stages_completed.append("execution_failed")
                print(f"[ERROR RECOVERY] SQL execution failed: {str(e)[:100]}")
                print(f"[ERROR RECOVERY] Starting recovery: Attempt 1 → Reasoning LLM")
        
        result.stage_times["execution"] = int((time.time() - stage_start) * 1000)
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 6: ERROR RECOVERY (only if execution failed)
        # ─────────────────────────────────────────────────────────────────
        # Tiered recovery strategy:
        #   Tier 2 → Error Classifier + SQL Coder mini-retry (no LLM classification)
        #   Tier 3 → Reasoning LLM (cheap, fast)  [only if Tier 2 didn't fix it]
        #   Tier 3 → Opus (powerful)               [only if Reasoning LLM failed]
        #   All fail → Return descriptive error to user
        # ═══════════════════════════════════════════════════════════════════
        if not result.success and result.error and "Query blocked" not in result.error:
            stage_start = time.time()
            result.error_recovery_attempted = True
            result.original_sql = result.sql  # Capture SQL before any recovery changes it

            # ── TIER 2: Error Classifier + SQL Coder mini-retry ─────────
            # Classify the database error with zero LLM cost, then send a
            # targeted mini-retry to SQL Coder (small prompt — SQL + diagnosis).
            # high/medium confidence → mini-retry → if fixed: done, else → Tier 3
            # low / unclassifiable  → skip mini-retry, go directly to Tier 3
            _tier2_fixed = False
            try:
                from sql_error_classifier import classify_sql_error, build_tier2_mini_retry_prompt
                _dialect_quote_chars = {
                    "postgresql": '"', "mysql": "`", "mssql": "[",
                    "oracle": '"', "sqlite": '"',
                }
                _t2_quote_char = _dialect_quote_chars.get(config.dialect, '"')

                diagnosis = classify_sql_error(result.error, result.sql, config.dialect)

                if diagnosis:
                    print(
                        f"[TIER 2] Classified error: {diagnosis.error_category} "
                        f"(confidence={diagnosis.confidence})"
                    )
                    if diagnosis.confidence in ("high", "medium"):
                        _t2_prompt = build_tier2_mini_retry_prompt(
                            original_sql=result.sql,
                            diagnosis=diagnosis,
                            dialect=config.dialect,
                            quote_char=_t2_quote_char,
                        )
                        _t2_response, _t2_tokens = call_llm(_t2_prompt, config.sql_provider)
                        _t2_fixed_sql = extract_sql_from_response(_t2_response)

                        result.tokens.tier2_fix = _t2_tokens
                        result.llm_trace.tier2_fix_input = _t2_prompt
                        result.llm_trace.tier2_fix_output = _t2_response

                        if _t2_fixed_sql:
                            try:
                                from db import run_sql as _run_sql
                                _t2_results = _run_sql(engine, _t2_fixed_sql)
                                # Tier 2 fixed it
                                result.sql = _t2_fixed_sql
                                result.results = _t2_results
                                result.success = True
                                result.error = None
                                result.error_recovery_method = "tier2_classifier"
                                result.sql_fixed = True
                                result.fixes_applied.append("tier2_error_classifier")
                                result.stages_completed.append("tier2_fix_success")
                                _tier2_fixed = True
                                print(f"[TIER 2] ✅ SQL Coder mini-retry succeeded.")
                            except Exception as _t2_exec_err:
                                result.error = str(_t2_exec_err)
                                result.stages_completed.append("tier2_fix_failed")
                                print(
                                    f"[TIER 2] Mini-retry failed: {str(_t2_exec_err)[:100]}. "
                                    f"Escalating to Tier 3."
                                )
                        else:
                            result.stages_completed.append("tier2_no_sql")
                            print(f"[TIER 2] SQL Coder returned no SQL. Escalating to Tier 3.")
                    else:
                        result.stages_completed.append("tier2_low_confidence_skip")
                        print(f"[TIER 2] Low confidence — skipping mini-retry, going to Tier 3.")
                else:
                    result.stages_completed.append("tier2_unclassified")
                    print(f"[TIER 2] Could not classify error. Going to Tier 3.")
            except Exception as _t2_err:
                print(f"[TIER 2] Classifier error (non-fatal): {_t2_err}. Going to Tier 3.")

            if not _tier2_fixed:
                # ── TIER 3: Reasoning LLM then Opus (existing flow) ─────
                # Only reached when Tier 2 couldn't fix the error.
                # Use stored metadata + plan if available (two-pass flow)
                has_two_pass_context = bool(result.pass2_plan and result.column_metadata)

                # Attempt 1: Reasoning LLM
                print(f"[TIER 3 / ERROR RECOVERY] Attempt 1: Reasoning LLM fixing error...")

                if has_two_pass_context:
                    from reasoning_prompts import create_error_retry_prompt
                    retry_prompt = create_error_retry_prompt(
                        question=question,
                        pass2_plan=result.pass2_plan,
                        metadata=result.column_metadata,
                        failed_sql=result.sql,
                        error_message=result.error,
                        dialect_info=config.dialect_info,
                        use_opus=False
                    )
                    prefill = "{" if "claude" in config.reasoning_provider else None
                    retry_response, retry_tokens = call_llm(
                        retry_prompt, config.reasoning_provider, prefill=prefill
                    )
                    result.tokens.error_fix_reasoning = retry_tokens
                    result.llm_trace.error_fix_reasoning_input = retry_prompt
                    result.llm_trace.error_fix_reasoning_output = retry_response

                    fixed_sql = extract_sql_from_response(retry_response)
                    reasoning_fix_result = {"fixed_sql": fixed_sql, "tokens": retry_tokens}

                    if fixed_sql and fixed_sql != result.sql:
                        try:
                            from db import run_sql
                            fix_results = run_sql(engine, fixed_sql)
                            reasoning_fix_result["success"] = True
                            reasoning_fix_result["results"] = fix_results
                            reasoning_fix_result["error"] = None
                        except Exception as fix_err:
                            reasoning_fix_result["success"] = False
                            reasoning_fix_result["error"] = str(fix_err)
                    else:
                        reasoning_fix_result["success"] = False
                        reasoning_fix_result["error"] = result.error
                else:
                    reasoning_fix_result = _run_reasoning_error_fix(
                        question=question,
                        sql=result.sql,
                        error=result.error,
                        schema_text=schema_text,
                        rules_compressed=rules_compressed,
                        config=config,
                        engine=engine
                    )
                    result.tokens.error_fix_reasoning = reasoning_fix_result.get("tokens", {"input": 0, "output": 0})
                    result.llm_trace.error_fix_reasoning_input = reasoning_fix_result.get("trace_input", "")
                    result.llm_trace.error_fix_reasoning_output = reasoning_fix_result.get("trace_output", "")

                if reasoning_fix_result.get("success"):
                    result.sql = reasoning_fix_result["fixed_sql"]
                    result.results = reasoning_fix_result["results"]
                    result.success = True
                    result.error = None
                    result.error_recovery_method = "reasoning_llm"
                    result.sql_fixed = True
                    result.fixes_applied.append("reasoning_llm_error_fix")
                    result.stages_completed.append("error_recovery_reasoning_success")
                    print(f"[TIER 3] ✅ Reasoning LLM fixed the error")
                else:
                    # Attempt 2: Opus
                    print(f"[TIER 3] Reasoning LLM failed. Attempt 2: Opus fixing error...")

                    if has_two_pass_context:
                        from reasoning_prompts import create_error_retry_prompt
                        opus_retry_prompt = create_error_retry_prompt(
                            question=question,
                            pass2_plan=result.pass2_plan,
                            metadata=result.column_metadata,
                            failed_sql=reasoning_fix_result.get("fixed_sql", result.sql),
                            error_message=reasoning_fix_result.get("error", result.error),
                            dialect_info=config.dialect_info,
                            use_opus=True
                        )
                        opus_retry_response, opus_retry_tokens = call_llm(
                            opus_retry_prompt, config.opus_provider
                        )
                        result.tokens.error_fix_opus = opus_retry_tokens
                        result.llm_trace.error_fix_opus_input = opus_retry_prompt
                        result.llm_trace.error_fix_opus_output = opus_retry_response

                        fixed_sql = extract_sql_from_response(opus_retry_response)
                        opus_fix_result = {"fixed_sql": fixed_sql, "tokens": opus_retry_tokens}

                        if fixed_sql:
                            try:
                                from db import run_sql
                                fix_results = run_sql(engine, fixed_sql)
                                opus_fix_result["success"] = True
                                opus_fix_result["results"] = fix_results
                            except Exception as fix_err:
                                opus_fix_result["success"] = False
                                opus_fix_result["error"] = str(fix_err)
                        else:
                            opus_fix_result["success"] = False
                            opus_fix_result["error"] = result.error
                    else:
                        opus_fix_result = _run_opus_error_fix(
                            question=question,
                            sql=reasoning_fix_result.get("fixed_sql", result.sql),
                            error=reasoning_fix_result.get("error", result.error),
                            schema_text=schema_text,
                            rules_compressed=rules_compressed,
                            config=config,
                            engine=engine
                        )
                        result.tokens.error_fix_opus = opus_fix_result.get("tokens", {"input": 0, "output": 0})
                        result.llm_trace.error_fix_opus_input = opus_fix_result.get("trace_opus_input", "")
                        result.llm_trace.error_fix_opus_output = opus_fix_result.get("trace_opus_output", "")

                    if opus_fix_result.get("success"):
                        result.sql = opus_fix_result["fixed_sql"]
                        result.results = opus_fix_result["results"]
                        result.success = True
                        result.error = None
                        result.error_recovery_method = "opus"
                        result.sql_fixed = True
                        result.fixes_applied.append("opus_error_fix")
                        result.stages_completed.append("error_recovery_opus_success")
                        print(f"[TIER 3] ✅ Opus fixed the error")
                    else:
                        result.error_recovery_method = "none"
                        result.error = _build_user_friendly_error(
                            original_error=result.original_error,
                            schema_text=schema_text
                        )
                        result.stages_completed.append("error_recovery_failed")
                        print(f"[TIER 3] ❌ All recovery attempts failed. Returning friendly error.")

            result.stage_times["error_recovery"] = int((time.time() - stage_start) * 1000)
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 7: OPUS REVIEW (only if enabled AND execution succeeded
        #          AND no error recovery was needed)
        # ─────────────────────────────────────────────────────────────────
        # NOTE: Opus review is now separate from error recovery.
        # Error recovery always uses the two-attempt strategy above.
        # Opus review here is optional SQL correctness validation.
        # ═══════════════════════════════════════════════════════════════════
        should_opus = _should_run_opus(config.enable_opus, result.complexity, result.success)
        
        # Skip Opus review if we already used Opus for error recovery
        # (it already reviewed the SQL implicitly by fixing it)
        if result.error_recovery_method == "opus":
            should_opus = False
            result.final_verdict = "FIXED_BY_OPUS"
            print(f"[DEBUG OPUS] Skipping review — Opus already handled error recovery")
        
        if should_opus:
            stage_start = time.time()
            
            opus_result = _run_opus_review(
                question=question,
                sql=result.sql,
                results=result.results,
                schema_text=schema_text,
                rules_compressed=rules_compressed,
                config=config,
                engine=engine,
                use_opus_refinement=(result.complexity == "hard")
            )
            
            result.llm_trace.opus_input = opus_result.get("trace_opus_input", "")
            result.llm_trace.opus_output = opus_result.get("trace_opus_output", "")
            result.llm_trace.refinement_input = opus_result.get("trace_refinement_input", "")
            result.llm_trace.refinement_output = opus_result.get("trace_refinement_output", "")

            result.opus_review = opus_result.get("final_review")
            result.opus_attempts = opus_result.get("attempts", 0)
            result.final_verdict = opus_result.get("verdict", "NOT_REVIEWED")
            result.tokens.opus = opus_result.get("tokens", {"input": 0, "output": 0})

            # Update flow path to show Opus ran
            result.flow_path = result.flow_path.rstrip() + " → Opus Review"

            # ── CRITICAL: Handle INCORRECT verdict ──────────────
            # If refinement produced a corrected SQL that actually executed,
            # use it (even if final verdict is FAILED_AFTER_RETRIES).
            # Then decide whether to block or warn.
            if result.final_verdict in ("INCORRECT", "FAILED_AFTER_RETRIES"):
                # First: apply corrected SQL if refinement succeeded
                if opus_result.get("corrected_sql"):
                    corrected_sql = opus_result["corrected_sql"]
                    corrected_results = opus_result.get("corrected_results")
                    if corrected_results is not None:
                        # Refinement produced SQL that actually executed
                        result.sql = corrected_sql
                        result.results = corrected_results
                        result.success = True
                        result.tokens.refinement = opus_result.get("refinement_tokens", {"input": 0, "output": 0})
                        result.sql_fixed = True
                        result.fixes_applied.append("opus_refinement")
                        # Still show warning since Opus wasn't fully satisfied
                        result.opus_blocked_soft = True
                        result.stages_completed.append("opus_refined_with_warning")
                        print(f"[OPUS] Refinement SQL applied — showing with warning")
                    else:
                        # Refinement SQL exists but didn't execute — fall through to block logic
                        pass

                # If no corrected results available, decide: block or warn
                if not (opus_result.get("corrected_sql") and opus_result.get("corrected_results") is not None):
                    opus_reasoning = ""
                    if result.opus_review:
                        opus_reasoning = (
                            result.opus_review.get("reasoning", "") +
                            " ".join(result.opus_review.get("issues", []))
                        ).lower()
                    
                    # Structural impossibility → hard block
                    hard_block_phrases = [
                        "column does not exist",
                        "table does not exist", 
                        "no direct way",
                        "cannot be attributed",
                        "no oem column",
                        "structurally impossible",
                        "data not available",
                        "no column exists"
                    ]
                    
                    # Fixable logic → soft warn
                    soft_block_phrases = [
                        "instead of rolling",
                        "financial year",
                        "date filter",
                        "wrong period",
                        "incorrect date range",
                        "should override",
                        "uses fy filter",
                        "calendar quarter",
                        "fiscal quarter"
                    ]
                    
                    is_hard_block = any(p in opus_reasoning for p in hard_block_phrases)
                    is_soft_block = any(p in opus_reasoning for p in soft_block_phrases)
                    
                    if is_hard_block and not is_soft_block:
                        # Genuine data unavailability — block results
                        result.results = None
                        result.success = False
                        opus_reasoning_text = (
                            result.opus_review.get("reasoning", "")
                            or result.opus_review.get("issues", ["The query result may not accurately answer your question."])[0]
                        )
                        result.error = f"OPUS_BLOCKED: {opus_reasoning_text}"
                        result.stages_completed.append("opus_blocked")
                        print(f"[OPUS] ❌ Result BLOCKED — structural impossibility")
                    else:
                        # Fixable logic error — show results with warning
                        result.opus_blocked_soft = True
                        opus_reasoning_text = (
                            result.opus_review.get("reasoning", "")
                            or result.opus_review.get("issues", ["The query result may not accurately answer your question."])[0]
                        )
                        result.error = f"OPUS_BLOCKED: {opus_reasoning_text}"
                        result.stages_completed.append("opus_warned")
                        print(f"[OPUS] ⚠️ Soft warning — fixable logic issue")
            else:
                if opus_result.get("corrected_sql") and result.success:
                    result.sql = opus_result["corrected_sql"]
                    result.results = opus_result.get("corrected_results", result.results)
                    result.tokens.refinement = opus_result.get("refinement_tokens", {"input": 0, "output": 0})

            result.stages_completed.append("opus_review")
            result.stage_times["opus_review"] = int((time.time() - stage_start) * 1000)
        else:
            if result.final_verdict == "NOT_REVIEWED":
                result.final_verdict = "NOT_REVIEWED"
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 8: STORE IN CACHE (if enabled and successful)
        # ═══════════════════════════════════════════════════════════════════
        if config.enable_cache and result.success and result.sql:
            try:
                from query_cache import QueryCache
                
                cache = QueryCache(
                    vector_engine=vector_engine,
                    enabled=True,
                    use_semantic=config.enable_semantic_cache
                )
                
                schema_version = QueryCache.compute_schema_version(
                    {t: [] for t in selected_tables} if selected_tables else {}
                )
                
                rules_for_version = []
                if rules_compressed and rules_compressed != "[]":
                    try:
                        rules_for_version = json.loads(rules_compressed)
                    except:
                        pass
                
                rules_version = QueryCache.compute_rules_version(rules_for_version)
                
                cache.set(
                    question=question,
                    sql=result.sql,
                    schema_version=schema_version,
                    rules_version=rules_version,
                    dialect=config.dialect,
                    complexity=result.complexity,
                    tokens_estimated=result.tokens.total_tokens()
                )
                
                result.cache_key = QueryCache.generate_cache_key(
                    QueryCache.normalize_question(question),
                    schema_version, rules_version, config.dialect
                )
                
                print(f"[CACHE] Stored: {result.cache_key[:16]}...")
                
            except Exception as e:
                print(f"[CACHE] Store error: {e}")
        
    except Exception as e:
        result.success = False
        result.error = f"Processing error: {str(e)}"
        result.stages_completed.append("error")
    
    result.total_time_ms = int((time.time() - start_time) * 1000)
    return result


# =============================================================================
# ERROR RECOVERY HELPERS
# =============================================================================

def _run_reasoning_error_fix(
    question: str,
    sql: str,
    error: str,
    schema_text: str,
    rules_compressed: str,
    config: FlowConfig,
    engine
) -> Dict[str, Any]:
    """
    Attempt 1 of error recovery: Use the Reasoning LLM to fix SQL.
    
    Handles common errors cheaply:
    - Type mismatches (text vs date, text vs integer)
    - Wrong column names
    - Missing schema prefix
    - Simple syntax errors
    
    Returns dict with: success, fixed_sql, results, error, tokens,
                       trace_input, trace_output
    """
    from llm_v2 import call_llm
    from db import run_sql
    from prompt_optimizer import extract_sql_from_response
    
    dialect_name = config.dialect.upper()

    # Prompt caching: put schema in system_prompt (stable) for Claude providers
    _fix_system = _build_static_system_prompt(
        dialect_syntax=get_dialect_syntax_rules(config.dialect),
        schema=schema_text,
        rules=rules_compressed,
        extra_instructions=f"You are a SQL expert. Fix failed {dialect_name} SQL queries.",
    ) if (_is_claude_provider(config.reasoning_provider) and config.enable_prompt_caching) else None

    if _fix_system:
        error_fix_prompt = (
            f"QUESTION: {question}\n\n"
            f"FAILED SQL:\n{sql}\n\n"
            f"ERROR:\n{error}\n\n"
            f"COMMON FIXES:\n"
            f"- text/date mismatch → CAST(col AS DATE) or col::date\n"
            f"- text/integer mismatch → CAST(col AS INTEGER)\n"
            f"- column not found → check schema for exact name and quoting\n"
            f"- ambiguous column → add table alias prefix\n\n"
            f"Return ONLY the corrected SQL. No explanation."
        )
    else:
        error_fix_prompt = (
            f"You are a SQL expert. Fix this failed {dialect_name} SQL query.\n\n"
            f"QUESTION: {question}\n\n"
            f"FAILED SQL:\n{sql}\n\n"
            f"ERROR:\n{error}\n\n"
            f"SCHEMA (check column names and data types carefully):\n{schema_text}\n\n"
            f"COMMON FIXES:\n"
            f"- text/date mismatch → CAST(col AS DATE) or DATE 'YYYY-MM-DD' format\n"
            f"- text/integer mismatch → CAST(col AS INTEGER)\n"
            f"- column not found → check schema for exact column name and quoting\n"
            f"- ambiguous column → add table alias prefix\n\n"
            f"Return ONLY the corrected SQL. No explanation."
        )

    response, tokens = call_llm(error_fix_prompt, config.reasoning_provider, system_prompt=_fix_system)
    fixed_sql = extract_sql_from_response(response)
    
    if not fixed_sql or fixed_sql == sql:
        print(f"[ERROR RECOVERY] Reasoning LLM produced no new SQL")
        return {
            "success": False,
            "fixed_sql": None,
            "results": None,
            "error": error,
            "tokens": tokens,
            "trace_input": error_fix_prompt,
            "trace_output": response
        }
    
    # Try executing the fixed SQL
    try:
        results = run_sql(engine, fixed_sql)
        return {
            "success": True,
            "fixed_sql": fixed_sql,
            "results": results,
            "error": None,
            "tokens": tokens,
            "trace_input": error_fix_prompt,
            "trace_output": response
        }
    except Exception as e:
        new_error = str(e)
        print(f"[ERROR RECOVERY] Reasoning LLM fix also failed: {new_error[:100]}")
        return {
            "success": False,
            "fixed_sql": fixed_sql,  # Pass to Opus so it can build on this attempt
            "results": None,
            "error": new_error,
            "tokens": tokens,
            "trace_input": error_fix_prompt,
            "trace_output": response
        }


def _build_user_friendly_error(
    original_error: str,
    schema_text: str
) -> str:
    """
    Convert a technical database error into a user-friendly message.
    Called when both Reasoning LLM and Opus fail to fix the SQL.
    
    Returns a message suitable for showing to a non-technical user.
    """
    error_lower = original_error.lower()
    
    if "does not exist" in error_lower and "operator" in error_lower:
        return (
            "There appears to be a data type mismatch in your database — "
            "a column being compared is stored as text instead of a date or number. "
            "Please ask your data team to verify the column data types."
        )
    elif "column" in error_lower and "does not exist" in error_lower:
        return (
            "A column referenced in the query could not be found in your database. "
            "This may be a schema change. Please verify your table structure."
        )
    elif "relation" in error_lower and "does not exist" in error_lower:
        return (
            "A table referenced in the query could not be found. "
            "Please verify the selected tables are accessible."
        )
    elif "permission" in error_lower or "access" in error_lower:
        return (
            "Database permission error. Your user account may not have "
            "read access to the required tables."
        )
    elif "timeout" in error_lower or "connection" in error_lower:
        return (
            "Database connection timed out. Please try again. "
            "If the problem persists, check your database connection."
        )
    elif "syntax" in error_lower:
        return (
            "A SQL syntax error occurred that could not be automatically resolved. "
            "Please try rephrasing your question."
        )
    else:
        # Generic but still helpful
        return (
            f"The query could not be executed after automatic repair attempts. "
            f"Technical detail: {original_error[:200]}"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _should_run_opus(enable_opus, complexity: str, execution_success: bool) -> bool:
    """Determine if Opus review should run."""
    print(f"[DEBUG OPUS] enable_opus={enable_opus} (type={type(enable_opus).__name__}), complexity={complexity}, success={execution_success}")
    
    if isinstance(enable_opus, str):
        if enable_opus.lower() == "true":
            enable_opus = True
        elif enable_opus.lower() == "false":
            enable_opus = False
    
    if enable_opus == True or enable_opus is True:
        print(f"[DEBUG OPUS] → Running: Always mode (success={execution_success})")
        return execution_success  # Only review if execution succeeded
    elif enable_opus == False or enable_opus is False:
        print(f"[DEBUG OPUS] → Skipping: Disabled")
        return False
    elif enable_opus == "auto":
        if not execution_success:
            print(f"[DEBUG OPUS] → Skipping: Auto mode + execution failed (handled by error recovery)")
            return False
        should_run = complexity == "hard"
        print(f"[DEBUG OPUS] → Auto mode: {'Running' if should_run else 'Skipping'} (complexity={complexity})")
        return should_run
    
    print(f"[DEBUG OPUS] → Skipping: Unknown value")
    return False


def _run_opus_review(
    question: str,
    sql: str,
    results: Any,
    schema_text: str,
    rules_compressed: str,
    config: FlowConfig,
    engine,
    use_opus_refinement: bool = False
) -> Dict[str, Any]:
    """
    Run Opus review with optional retry on INCORRECT verdict.

    If use_opus_refinement=True (recommended for hard queries),
    Opus also generates refinement SQL between review attempts.
    
    IMPORTANT: sql parameter must always be the FINAL executed SQL.

    Returns a dict that includes:
      - verdict, final_review, attempts, tokens
      - trace_opus_input / trace_opus_output
      - trace_refinement_input / trace_refinement_output
      - corrected_sql / corrected_results / refinement_tokens (if refined)
    """
    from llm_v2 import call_llm
    from db import run_sql
    from prompt_optimizer import (
        create_opus_review_prompt_optimized,
        create_refinement_prompt,
        extract_sql_from_response,
        extract_columns_from_sql
    )
    
    total_tokens = {"input": 0, "output": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    refinement_tokens = {"input": 0, "output": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    current_sql = sql
    current_results = results

    trace_opus_input = ""
    trace_opus_output = ""
    trace_refinement_input = ""
    trace_refinement_output = ""

    use_prefill = _is_claude_provider(config.opus_provider)
    use_cache = _is_claude_provider(config.opus_provider) and config.enable_prompt_caching

    # Build cacheable system prompt (schema + rules) for Opus review
    _review_system = _build_static_system_prompt(
        schema=schema_text,
        rules=rules_compressed,
    ) if use_cache else None

    review = {}

    for attempt in range(1, config.max_retries + 1):
        if hasattr(current_results, 'head'):
            results_preview = current_results.head(10).to_string()
        else:
            results_preview = str(current_results)[:1500]

        columns_used = extract_columns_from_sql(current_sql)

        # When caching, omit schema+rules from user message (already in system)
        opus_prompt = create_opus_review_prompt_optimized(
            question=question,
            sql=current_sql,
            results_preview=results_preview,
            columns_used=columns_used,
            schema_text="" if use_cache else schema_text,
            rules_compressed="" if use_cache else rules_compressed,
        )

        if use_prefill:
            opus_response, opus_tokens = call_llm(
                opus_prompt, config.opus_provider, prefill="{", system_prompt=_review_system,
            )
        else:
            opus_response, opus_tokens = call_llm(opus_prompt, config.opus_provider)

        total_tokens["input"] += opus_tokens["input"]
        total_tokens["output"] += opus_tokens["output"]
        total_tokens["cache_creation_input_tokens"] += opus_tokens.get("cache_creation_input_tokens", 0)
        total_tokens["cache_read_input_tokens"] += opus_tokens.get("cache_read_input_tokens", 0)

        if attempt == 1:
            trace_opus_input = opus_prompt
            trace_opus_output = opus_response
        
        try:
            response_text = opus_response
            if "```json" in response_text:
                response_text = response_text.split("```json")[-1].split("```")[0].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
            
            review = json.loads(response_text)
        except:
            review = {"verdict": "UNCERTAIN", "issues": ["Parse error"], "reasoning": opus_response[:200]}
        
        verdict = review.get("verdict", "UNCERTAIN")
        
        if verdict in ["CORRECT", "UNCERTAIN"]:
            return {
                "verdict": verdict,
                "final_review": review,
                "attempts": attempt,
                "tokens": total_tokens,
                # Propagate refined SQL when it changed (reviewed & approved on attempt 2+)
                "corrected_sql": current_sql if current_sql != sql else None,
                "corrected_results": current_results if current_sql != sql else None,
                "refinement_tokens": refinement_tokens,
                "trace_opus_input": trace_opus_input,
                "trace_opus_output": trace_opus_output,
                "trace_refinement_input": trace_refinement_input,
                "trace_refinement_output": trace_refinement_output,
            }
        
        if attempt < config.max_retries:
            refine_prompt = create_refinement_prompt(
                question, current_sql, review, schema_text, rules_compressed
            )
            
            refinement_provider = config.opus_provider if use_opus_refinement else config.reasoning_provider
            prefill = "{" if "claude" in refinement_provider else None
            refine_response, refine_tokens = call_llm(
                refine_prompt, refinement_provider, prefill=prefill
            )
            
            refinement_tokens["input"] += refine_tokens["input"]
            refinement_tokens["output"] += refine_tokens["output"]

            trace_refinement_input = refine_prompt
            trace_refinement_output = refine_response
            
            new_sql = extract_sql_from_response(refine_response)

            # If model repeats same SQL after INCORRECT verdict, force a full regeneration
            if (not new_sql or new_sql.strip() == current_sql.strip()) and verdict == "INCORRECT":
                force_prompt = f"""Previous refinement repeated the same incorrect SQL.
Generate a NEW corrected SQL from scratch that resolves all auditor failures.

QUESTION:
{question}

AUDITOR FEEDBACK JSON:
{json.dumps(review, ensure_ascii=False)}

SCHEMA:
{schema_text}

BUSINESS RULES:
{rules_compressed}

REJECTED SQL:
{current_sql}

Return ONLY SQL. No explanation."""
                force_provider = config.opus_provider if use_opus_refinement else config.reasoning_provider
                force_prefill = "{" if "claude" in force_provider else None
                force_response, force_tokens = call_llm(force_prompt, force_provider, prefill=force_prefill)
                refinement_tokens["input"] += force_tokens.get("input", 0)
                refinement_tokens["output"] += force_tokens.get("output", 0)
                trace_refinement_input = force_prompt
                trace_refinement_output = force_response
                new_sql = extract_sql_from_response(force_response)

            if new_sql and new_sql != current_sql:
                current_sql = new_sql
                try:
                    current_results = run_sql(engine, current_sql)
                except Exception:
                    pass
    
    return {
        "verdict": "FAILED_AFTER_RETRIES",
        "final_review": review,
        "attempts": config.max_retries,
        "tokens": total_tokens,
        "corrected_sql": current_sql if current_sql != sql else None,
        "corrected_results": current_results if current_sql != sql else None,
        "refinement_tokens": refinement_tokens,
        "trace_opus_input": trace_opus_input,
        "trace_opus_output": trace_opus_output,
        "trace_refinement_input": trace_refinement_input,
        "trace_refinement_output": trace_refinement_output,
    }


def _run_opus_error_fix(
    question: str, sql: str, error: str,
    schema_text: str, rules_compressed: str,
    config: FlowConfig, engine
) -> Dict[str, Any]:
    """
    Attempt 2 of error recovery: Use Opus to fix SQL that Reasoning LLM couldn't fix.
    
    Opus has full schema context and is better at:
    - Complex type inference
    - Multi-table JOIN errors
    - Subtle schema issues the reasoning LLM missed
    
    Returns dict with: success, fixed_sql, results, error, verdict, tokens,
                       trace_opus_input, trace_opus_output
    """
    from llm_v2 import call_llm
    from db import run_sql
    from prompt_optimizer import extract_sql_from_response
    
    total_tokens = {"input": 0, "output": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    errors_seen = [error]
    current_sql = sql
    trace_opus_input = ""
    trace_opus_output = ""

    # Prompt caching: schema + rules in system_prompt (stable across retry attempts)
    _fix_opus_system = _build_static_system_prompt(
        dialect_syntax=get_dialect_syntax_rules(config.dialect),
        schema=schema_text,
        rules=rules_compressed,
        extra_instructions="You are an expert SQL debugger. Analyze the error and produce a corrected SQL.",
    ) if (_is_claude_provider(config.opus_provider) and config.enable_prompt_caching) else None

    for attempt in range(1, config.max_retries + 1):
        print(f"[TIER 3] Opus attempt {attempt}/{config.max_retries}")

        # Dynamic part: only the failed SQL + error (schema/rules already in system)
        if _fix_opus_system:
            error_fix_prompt = (
                f"DATABASE: {config.dialect.upper()}\n\n"
                f"ORIGINAL QUESTION: {question}\n\n"
                f"FAILED SQL:\n{current_sql}\n\n"
                f"ERROR MESSAGE:\n{errors_seen[-1]}\n\n"
                f"ANALYSIS REQUIRED:\n"
                f"1. Exact error type (type mismatch, missing column, syntax, etc.)\n"
                f"2. Column/table causing the issue\n"
                f"3. Correct data type from the schema\n"
                f"4. Precise fix\n\n"
                f"Common fixes:\n"
                f"- text vs date → CAST(col AS DATE) or col::date\n"
                f"- text vs integer → CAST(col AS INTEGER)\n\n"
                f"Return ONLY the corrected SQL. No explanation."
            )
        else:
            error_fix_prompt = (
                f"You are a SQL expert. The following SQL query failed.\n\n"
                f"DATABASE: {config.dialect.upper()}\n\n"
                f"ORIGINAL QUESTION: {question}\n\n"
                f"FAILED SQL:\n{current_sql}\n\n"
                f"ERROR MESSAGE:\n{errors_seen[-1]}\n\n"
                f"SCHEMA (pay close attention to data types):\n{schema_text}\n\n"
                f"BUSINESS RULES:\n{rules_compressed}\n\n"
                f"Return ONLY the corrected SQL. No explanation, no markdown."
            )

        response, tokens = call_llm(
            error_fix_prompt, config.opus_provider, system_prompt=_fix_opus_system,
        )
        total_tokens["input"] += tokens["input"]
        total_tokens["output"] += tokens["output"]
        total_tokens["cache_creation_input_tokens"] += tokens.get("cache_creation_input_tokens", 0)
        total_tokens["cache_read_input_tokens"] += tokens.get("cache_read_input_tokens", 0)

        if attempt == 1:
            trace_opus_input = error_fix_prompt
            trace_opus_output = response
        
        fixed_sql = extract_sql_from_response(response)
        
        if not fixed_sql or fixed_sql == current_sql:
            print(f"[ERROR RECOVERY] Opus produced no new SQL on attempt {attempt}")
            continue
        
        try:
            results = run_sql(engine, fixed_sql)
            print(f"[ERROR RECOVERY] ✅ Opus fixed SQL on attempt {attempt}")
            
            return {
                "verdict": f"FIXED_BY_OPUS_ATTEMPT_{attempt}",
                "fixed_sql": fixed_sql,
                "results": results,
                "success": True,
                "error": None,
                "attempts": attempt,
                "tokens": total_tokens,
                "final_review": {
                    "original_error": errors_seen[0],
                    "fix_applied": True,
                    "attempts": attempt
                },
                "trace_opus_input": trace_opus_input,
                "trace_opus_output": trace_opus_output,
            }
            
        except Exception as e:
            new_error = str(e)
            print(f"[ERROR RECOVERY] Opus fix attempt {attempt} also failed: {new_error[:100]}")
            
            error_type = _extract_error_type(new_error)
            if any(_extract_error_type(e) == error_type for e in errors_seen):
                print(f"[ERROR RECOVERY] Same error type seen before, stopping Opus attempts")
                break
            
            errors_seen.append(new_error)
            current_sql = fixed_sql
    
    return {
        "verdict": "FIX_FAILED",
        "fixed_sql": None,
        "results": None,
        "success": False,
        "error": errors_seen[-1],
        "attempts": config.max_retries,
        "tokens": total_tokens,
        "final_review": {
            "original_error": errors_seen[0],
            "all_errors": errors_seen,
            "fix_applied": False
        },
        "trace_opus_input": trace_opus_input,
        "trace_opus_output": trace_opus_output,
    }


def _extract_error_type(error_message: str) -> str:
    """Extract error type for comparison (avoid infinite loops on same error)."""
    error_lower = error_message.lower()
    
    if "does not exist" in error_lower:
        return "not_found"
    elif "type" in error_lower or "cast" in error_lower or "operator" in error_lower:
        return "type_mismatch"
    elif "syntax" in error_lower:
        return "syntax"
    elif "ambiguous" in error_lower:
        return "ambiguous"
    elif "permission" in error_lower:
        return "permission"
    elif "connection" in error_lower or "timeout" in error_lower:
        return "connection"
    else:
        return error_message[:50]


def create_default_config(
    dialect: str = "postgresql",
    reasoning_provider: str = "claude_sonnet",
    sql_provider: str = "groq",
    enable_opus: str = "auto"
) -> FlowConfig:
    """Create a default configuration."""
    dialect_configs = {
        "postgresql": {"dialect": "postgresql", "quote_char": '"', "string_quote": "'"},
        "mysql": {"dialect": "mysql", "quote_char": '`', "string_quote": "'"},
        "mssql": {"dialect": "mssql", "quote_char": '[', "string_quote": "'"},
        "oracle": {"dialect": "oracle", "quote_char": '"', "string_quote": "'"},
        "sqlite": {"dialect": "sqlite", "quote_char": '"', "string_quote": "'"},
    }
    
    return FlowConfig(
        dialect=dialect,
        dialect_info=dialect_configs.get(dialect, dialect_configs["postgresql"]),
        reasoning_provider=reasoning_provider,
        sql_provider=sql_provider,
        enable_opus=enable_opus
    )


if __name__ == "__main__":
    print("=" * 70)
    print("FLOW ROUTER MODULE")
    print("=" * 70)
    print("\nFlow paths:")
    print("  EASY:   Question → RAG → SQL LLM → Execute → [Error Recovery if needed]")
    print("  MEDIUM: Question → RAG → Reasoning LLM → SQL Coder → Execute → [Error Recovery if needed]")
    print("  HARD:   Question → RAG → Reasoning LLM → SQL Coder → Execute → [Error Recovery if needed]")
    print("\nError Recovery:")
    print("  Attempt 1: Reasoning LLM (cheap, fast)")
    print("  Attempt 2: Opus (powerful, handles complex issues)")
    print("  Both fail: User-friendly error message")
    print("=" * 70)
