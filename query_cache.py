"""
Query Cache Module
==================
Intelligent caching for SQL queries based on question + schema + rules.
LLM-agnostic: Cache key does NOT include which LLM was used.

Features:
- Exact match caching (hash-based)
- Semantic similarity caching (embedding-based)
- Version tracking (schema, rules, opus)
- Toggle on/off
- Cache statistics

Usage:
    from query_cache import QueryCache
    
    cache = QueryCache(vector_engine, enabled=True)
    
    # Check cache
    result = cache.get(question, schema_version, rules_version, dialect)
    if result:
        return result['sql']
    
    # After generating SQL
    cache.set(question, sql, schema_version, rules_version, dialect)
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class CacheStats:
    """Track cache performance."""
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    tokens_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "tokens_saved": self.tokens_saved
        }


class QueryCache:
    """
    LLM-agnostic query cache.
    
    Cache key = hash(normalized_question + schema_version + rules_version + dialect)
    
    This means:
    - Same question with same schema/rules = cache hit (regardless of LLM)
    - Schema changes = cache miss
    - Rule changes = cache miss
    - LLM changes = still cache hit!
    """
    
    def __init__(
        self, 
        vector_engine=None, 
        enabled: bool = True,
        semantic_threshold: float = 0.95,
        use_semantic: bool = True
    ):
        """
        Initialize cache.
        
        Args:
            vector_engine: SQLAlchemy engine for Supabase (for persistent cache)
            enabled: Whether caching is enabled
            semantic_threshold: Similarity threshold for semantic matching
            use_semantic: Whether to use semantic (embedding) matching
        """
        self.vector_engine = vector_engine
        self.enabled = enabled
        self.semantic_threshold = semantic_threshold
        self.use_semantic = use_semantic
        self.stats = CacheStats()
        
        # In-memory cache for fast lookups (supplementary to DB)
        self._memory_cache: Dict[str, Dict] = {}
        
        # Ensure table exists
        if vector_engine and enabled:
            self._ensure_table()
    
    def _ensure_table(self):
        """Create cache table if not exists."""
        from sqlalchemy import text
        
        try:
            with self.vector_engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                        cache_key TEXT UNIQUE NOT NULL,
                        question_original TEXT NOT NULL,
                        question_normalized TEXT NOT NULL,
                        question_embedding vector(384),
                        sql_output TEXT NOT NULL,
                        schema_version TEXT,
                        rules_version TEXT,
                        opus_version TEXT,
                        dialect TEXT,
                        complexity TEXT,
                        tokens_estimated INT DEFAULT 0,
                        hit_count INT DEFAULT 0,
                        user_verified BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        last_hit_at TIMESTAMP DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_query_cache_key 
                    ON query_cache(cache_key);
                    
                    CREATE INDEX IF NOT EXISTS idx_query_cache_embedding 
                    ON query_cache USING ivfflat (question_embedding vector_cosine_ops)
                    WITH (lists = 100);
                """))
                conn.commit()
        except Exception as e:
            print(f"[CACHE] Warning: Could not create cache table: {e}")
    
    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Normalize question for consistent matching.
        
        - Lowercase
        - Remove extra whitespace
        - Remove punctuation
        - Sort words (optional, for bag-of-words matching)
        """
        import re
        
        # Lowercase
        q = question.lower().strip()
        
        # Remove extra whitespace
        q = re.sub(r'\s+', ' ', q)
        
        # Remove common filler words that don't change meaning
        fillers = ['please', 'can you', 'could you', 'show me', 'give me', 'i want', 'i need']
        for filler in fillers:
            q = q.replace(filler, '')
        
        # Remove punctuation except for important operators
        q = re.sub(r'[^\w\s<>=!]', '', q)
        
        # Clean up whitespace again
        q = re.sub(r'\s+', ' ', q).strip()
        
        return q
    
    @staticmethod
    def generate_cache_key(
        question_normalized: str,
        schema_version: str,
        rules_version: str,
        dialect: str
    ) -> str:
        """Generate deterministic cache key."""
        key_input = f"{question_normalized}|{schema_version}|{rules_version}|{dialect}"
        return hashlib.sha256(key_input.encode()).hexdigest()[:32]
    
    @staticmethod
    def compute_schema_version(schema_info: Dict) -> str:
        """
        Compute schema version hash.
        
        Args:
            schema_info: Dict of {table_name: [columns]}
        """
        if not schema_info:
            return "empty"
        
        # Sort for consistency
        items = []
        for table, columns in sorted(schema_info.items()):
            if isinstance(columns, list):
                for col in sorted(columns):
                    items.append(f"{table}.{col}")
            elif isinstance(columns, dict):
                for col, dtype in sorted(columns.items()):
                    items.append(f"{table}.{col}.{dtype}")
        
        content = "|".join(items)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @staticmethod
    def compute_rules_version(rules: List[Dict]) -> str:
        """Compute rules version hash."""
        if not rules:
            return "empty"
        
        items = []
        for rule in sorted(rules, key=lambda r: r.get('rule_name', '')):
            name = rule.get('rule_name', '')
            rtype = rule.get('rule_type', '')
            # Include rule_data hash for content changes
            data_str = json.dumps(rule.get('rule_data', {}), sort_keys=True)
            data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
            items.append(f"{name}.{rtype}.{data_hash}")
        
        content = "|".join(items)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get(
        self,
        question: str,
        schema_version: str,
        rules_version: str,
        dialect: str,
        opus_version: str = None
    ) -> Optional[Dict]:
        """
        Look up cached SQL for question.
        
        Returns:
            Dict with 'sql', 'cache_key', 'hit_type' if found, None otherwise
        """
        if not self.enabled:
            return None
        
        question_normalized = self.normalize_question(question)
        cache_key = self.generate_cache_key(
            question_normalized, schema_version, rules_version, dialect
        )
        
        # Try memory cache first (fastest)
        if cache_key in self._memory_cache:
            self.stats.hits += 1
            self._memory_cache[cache_key]['hit_count'] += 1
            return {
                **self._memory_cache[cache_key],
                'hit_type': 'memory'
            }
        
        # Try database exact match
        if self.vector_engine:
            result = self._db_exact_lookup(cache_key)
            if result:
                self.stats.hits += 1
                # Populate memory cache
                self._memory_cache[cache_key] = result
                return {**result, 'hit_type': 'exact'}
            
            # Try semantic match if enabled
            if self.use_semantic:
                result = self._db_semantic_lookup(
                    question_normalized, schema_version, rules_version, dialect
                )
                if result:
                    self.stats.hits += 1
                    self.stats.semantic_hits += 1
                    return {**result, 'hit_type': 'semantic'}
        
        # Cache miss
        self.stats.misses += 1
        return None
    
    def _db_exact_lookup(self, cache_key: str) -> Optional[Dict]:
        """Exact match lookup in database."""
        from sqlalchemy import text
        
        try:
            with self.vector_engine.connect() as conn:
                result = conn.execute(
                    text("""
                        UPDATE query_cache 
                        SET hit_count = hit_count + 1, last_hit_at = NOW()
                        WHERE cache_key = :key
                        RETURNING sql_output, complexity, tokens_estimated, 
                                  hit_count, user_verified, cache_key
                    """),
                    {"key": cache_key}
                ).fetchone()
                conn.commit()
                
                if result:
                    return {
                        "sql": result[0],
                        "complexity": result[1],
                        "tokens_estimated": result[2],
                        "hit_count": result[3],
                        "user_verified": result[4],
                        "cache_key": result[5]
                    }
        except Exception as e:
            print(f"[CACHE] DB lookup error: {e}")
        
        return None
    
    def _db_semantic_lookup(
        self,
        question_normalized: str,
        schema_version: str,
        rules_version: str,
        dialect: str
    ) -> Optional[Dict]:
        """Semantic similarity lookup using embeddings."""
        from sqlalchemy import text
        
        try:
            # Get embedding for question
            from vector_utils_v2 import get_embedding
            embedding = get_embedding(question_normalized)
            
            if embedding is None:
                return None
            
            with self.vector_engine.connect() as conn:
                # Find similar questions with same schema/rules versions
                result = conn.execute(
                    text("""
                        SELECT 
                            sql_output, complexity, tokens_estimated,
                            hit_count, user_verified, cache_key,
                            1 - (question_embedding <=> :embedding::vector) as similarity
                        FROM query_cache
                        WHERE schema_version = :schema_ver
                          AND rules_version = :rules_ver
                          AND dialect = :dialect
                          AND question_embedding IS NOT NULL
                        ORDER BY question_embedding <=> :embedding::vector
                        LIMIT 1
                    """),
                    {
                        "embedding": embedding,
                        "schema_ver": schema_version,
                        "rules_ver": rules_version,
                        "dialect": dialect
                    }
                ).fetchone()
                
                if result and result[6] >= self.semantic_threshold:
                    # Update hit count
                    conn.execute(
                        text("""
                            UPDATE query_cache 
                            SET hit_count = hit_count + 1, last_hit_at = NOW()
                            WHERE cache_key = :key
                        """),
                        {"key": result[5]}
                    )
                    conn.commit()
                    
                    return {
                        "sql": result[0],
                        "complexity": result[1],
                        "tokens_estimated": result[2],
                        "hit_count": result[3],
                        "user_verified": result[4],
                        "cache_key": result[5],
                        "similarity": result[6]
                    }
        except Exception as e:
            print(f"[CACHE] Semantic lookup error: {e}")
        
        return None
    
    def set(
        self,
        question: str,
        sql: str,
        schema_version: str,
        rules_version: str,
        dialect: str,
        complexity: str = None,
        tokens_estimated: int = 0,
        opus_version: str = None
    ) -> bool:
        """
        Store SQL in cache.
        
        Returns:
            True if stored successfully
        """
        if not self.enabled:
            return False
        
        question_normalized = self.normalize_question(question)
        cache_key = self.generate_cache_key(
            question_normalized, schema_version, rules_version, dialect
        )
        
        cache_entry = {
            "sql": sql,
            "complexity": complexity,
            "tokens_estimated": tokens_estimated,
            "hit_count": 0,
            "user_verified": False,
            "cache_key": cache_key
        }
        
        # Store in memory cache
        self._memory_cache[cache_key] = cache_entry
        
        # Store in database
        if self.vector_engine:
            return self._db_store(
                cache_key=cache_key,
                question_original=question,
                question_normalized=question_normalized,
                sql=sql,
                schema_version=schema_version,
                rules_version=rules_version,
                dialect=dialect,
                complexity=complexity,
                tokens_estimated=tokens_estimated,
                opus_version=opus_version
            )
        
        return True
    
    def _db_store(
        self,
        cache_key: str,
        question_original: str,
        question_normalized: str,
        sql: str,
        schema_version: str,
        rules_version: str,
        dialect: str,
        complexity: str = None,
        tokens_estimated: int = 0,
        opus_version: str = None
    ) -> bool:
        """Store in database with embedding."""
        from sqlalchemy import text
        
        try:
            # Get embedding for semantic matching
            embedding = None
            try:
                from vector_utils_v2 import get_embedding
                embedding = get_embedding(question_normalized)
            except:
                pass
            
            with self.vector_engine.connect() as conn:
                if embedding:
                    conn.execute(
                        text("""
                            INSERT INTO query_cache (
                                cache_key, question_original, question_normalized,
                                question_embedding, sql_output, schema_version,
                                rules_version, opus_version, dialect, complexity,
                                tokens_estimated
                            ) VALUES (
                                :cache_key, :question_original, :question_normalized,
                                :embedding::vector, :sql, :schema_version,
                                :rules_version, :opus_version, :dialect, :complexity,
                                :tokens_estimated
                            )
                            ON CONFLICT (cache_key) DO UPDATE SET
                                sql_output = EXCLUDED.sql_output,
                                tokens_estimated = EXCLUDED.tokens_estimated,
                                complexity = EXCLUDED.complexity
                        """),
                        {
                            "cache_key": cache_key,
                            "question_original": question_original,
                            "question_normalized": question_normalized,
                            "embedding": embedding,
                            "sql": sql,
                            "schema_version": schema_version,
                            "rules_version": rules_version,
                            "opus_version": opus_version,
                            "dialect": dialect,
                            "complexity": complexity,
                            "tokens_estimated": tokens_estimated
                        }
                    )
                else:
                    # Without embedding
                    conn.execute(
                        text("""
                            INSERT INTO query_cache (
                                cache_key, question_original, question_normalized,
                                sql_output, schema_version, rules_version,
                                opus_version, dialect, complexity, tokens_estimated
                            ) VALUES (
                                :cache_key, :question_original, :question_normalized,
                                :sql, :schema_version, :rules_version,
                                :opus_version, :dialect, :complexity, :tokens_estimated
                            )
                            ON CONFLICT (cache_key) DO UPDATE SET
                                sql_output = EXCLUDED.sql_output,
                                tokens_estimated = EXCLUDED.tokens_estimated,
                                complexity = EXCLUDED.complexity
                        """),
                        {
                            "cache_key": cache_key,
                            "question_original": question_original,
                            "question_normalized": question_normalized,
                            "sql": sql,
                            "schema_version": schema_version,
                            "rules_version": rules_version,
                            "opus_version": opus_version,
                            "dialect": dialect,
                            "complexity": complexity,
                            "tokens_estimated": tokens_estimated
                        }
                    )
                conn.commit()
                return True
                
        except Exception as e:
            print(f"[CACHE] DB store error: {e}")
            return False
    
    def invalidate(self, cache_key: str = None, question: str = None) -> bool:
        """
        Invalidate specific cache entry.
        
        Args:
            cache_key: Direct cache key
            question: Question to invalidate (will be normalized)
        """
        from sqlalchemy import text
        
        if question and not cache_key:
            # We need versions to compute cache key, so just delete by normalized question
            question_normalized = self.normalize_question(question)
            
            # Remove from memory
            keys_to_remove = [
                k for k, v in self._memory_cache.items()
                if v.get('question_normalized') == question_normalized
            ]
            for k in keys_to_remove:
                del self._memory_cache[k]
            
            # Remove from DB
            if self.vector_engine:
                try:
                    with self.vector_engine.connect() as conn:
                        conn.execute(
                            text("DELETE FROM query_cache WHERE question_normalized = :q"),
                            {"q": question_normalized}
                        )
                        conn.commit()
                except Exception as e:
                    print(f"[CACHE] Invalidate error: {e}")
                    return False
        
        elif cache_key:
            # Remove from memory
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            # Remove from DB
            if self.vector_engine:
                try:
                    with self.vector_engine.connect() as conn:
                        conn.execute(
                            text("DELETE FROM query_cache WHERE cache_key = :key"),
                            {"key": cache_key}
                        )
                        conn.commit()
                except Exception as e:
                    print(f"[CACHE] Invalidate error: {e}")
                    return False
        
        return True
    
    def invalidate_all(self) -> bool:
        """Clear entire cache."""
        from sqlalchemy import text
        
        self._memory_cache.clear()
        self.stats = CacheStats()
        
        if self.vector_engine:
            try:
                with self.vector_engine.connect() as conn:
                    conn.execute(text("TRUNCATE TABLE query_cache"))
                    conn.commit()
            except Exception as e:
                print(f"[CACHE] Invalidate all error: {e}")
                return False
        
        return True
    
    def mark_verified(self, cache_key: str, verified: bool = True) -> bool:
        """Mark a cached query as user-verified (correct/incorrect)."""
        from sqlalchemy import text
        
        if cache_key in self._memory_cache:
            self._memory_cache[cache_key]['user_verified'] = verified
        
        if self.vector_engine:
            try:
                with self.vector_engine.connect() as conn:
                    conn.execute(
                        text("""
                            UPDATE query_cache 
                            SET user_verified = :verified 
                            WHERE cache_key = :key
                        """),
                        {"verified": verified, "key": cache_key}
                    )
                    conn.commit()
                    return True
            except Exception as e:
                print(f"[CACHE] Mark verified error: {e}")
        
        return False
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats['memory_cache_size'] = len(self._memory_cache)
        stats['enabled'] = self.enabled
        stats['semantic_enabled'] = self.use_semantic
        
        # Get DB stats if available
        if self.vector_engine:
            from sqlalchemy import text
            try:
                with self.vector_engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(hit_count) as total_hits,
                            COUNT(*) FILTER (WHERE user_verified = true) as verified,
                            AVG(tokens_estimated) as avg_tokens
                        FROM query_cache
                    """)).fetchone()
                    
                    if result:
                        stats['db_entries'] = result[0]
                        stats['db_total_hits'] = result[1] or 0
                        stats['db_verified'] = result[2]
                        stats['db_avg_tokens'] = round(result[3] or 0)
            except:
                pass
        
        return stats
    
    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        """Get recent cache entries for display."""
        from sqlalchemy import text
        
        if not self.vector_engine:
            return list(self._memory_cache.values())[:limit]
        
        try:
            with self.vector_engine.connect() as conn:
                results = conn.execute(
                    text("""
                        SELECT 
                            cache_key, question_original, sql_output,
                            complexity, hit_count, user_verified, created_at
                        FROM query_cache
                        ORDER BY last_hit_at DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                ).fetchall()
                
                return [
                    {
                        "cache_key": r[0],
                        "question": r[1],
                        "sql": r[2],
                        "complexity": r[3],
                        "hit_count": r[4],
                        "user_verified": r[5],
                        "created_at": r[6].isoformat() if r[6] else None
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"[CACHE] Get recent error: {e}")
            return []


# Convenience function for computing versions
def compute_versions(
    schema_info: Dict = None,
    rules: List[Dict] = None,
    opus_descriptions: Dict = None
) -> Dict[str, str]:
    """
    Compute version hashes for schema, rules, and opus.
    
    Returns:
        {"schema": "abc123", "rules": "def456", "opus": "ghi789"}
    """
    return {
        "schema": QueryCache.compute_schema_version(schema_info or {}),
        "rules": QueryCache.compute_rules_version(rules or []),
        "opus": hashlib.md5(
            json.dumps(opus_descriptions or {}, sort_keys=True).encode()
        ).hexdigest()[:16] if opus_descriptions else "empty"
    }
