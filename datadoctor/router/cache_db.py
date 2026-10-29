from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Optional, Dict, Any, Union


class LLMCacheDB:
    """Persistent SQLite Cache & Token Ledger for LLM Judge queries.
    
    Ensures that identical text samples evaluated with identical medoid contexts
    reuse responses across random seeds and benchmark reruns.
    """

    def __init__(self, db_path: Union[str, Path] = ".cache/datadoctor/llm_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _execute(self, query: str, params: tuple = ()) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(query, params)
        finally:
            conn.close()

    def _init_db(self) -> None:
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                sample_sha1 TEXT NOT NULL,
                medoids_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_mode TEXT NOT NULL,
                prompt_text TEXT,
                raw_response TEXT,
                suggested_label TEXT,
                confidence REAL,
                rationale TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                created_at TEXT,
                PRIMARY KEY (sample_sha1, medoids_hash, model_id, prompt_mode)
            )
            """
        )

    def get(
        self,
        sample_sha1: str,
        medoids_hash: str,
        model_id: str,
        prompt_mode: str = "blind",
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached LLM response if present."""
        query = """
            SELECT suggested_label, confidence, rationale, input_tokens, output_tokens, raw_response
            FROM llm_cache
            WHERE sample_sha1 = ? AND medoids_hash = ? AND model_id = ? AND prompt_mode = ?
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(query, (sample_sha1, medoids_hash, model_id, prompt_mode))
            row = cursor.fetchone()
            if row is not None:
                return {
                    "suggested_label": row[0],
                    "confidence": float(row[1]),
                    "rationale": row[2],
                    "input_tokens": int(row[3]),
                    "output_tokens": int(row[4]),
                    "raw_response": row[5],
                }
        finally:
            conn.close()
        return None

    def put(
        self,
        sample_sha1: str,
        medoids_hash: str,
        model_id: str,
        prompt_mode: str,
        prompt_text: str,
        raw_response: str,
        suggested_label: str,
        confidence: float,
        rationale: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Stores LLM response and token ledger entry."""
        query = """
            INSERT OR REPLACE INTO llm_cache (
                sample_sha1, medoids_hash, model_id, prompt_mode, prompt_text,
                raw_response, suggested_label, confidence, rationale,
                input_tokens, output_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.utcnow().isoformat()
        self._execute(
            query,
            (
                sample_sha1,
                medoids_hash,
                model_id,
                prompt_mode,
                prompt_text,
                raw_response,
                suggested_label,
                confidence,
                rationale,
                input_tokens,
                output_tokens,
                now,
            ),
        )

    def get_token_summary(self) -> Dict[str, Any]:
        """Returns total cached input and output token metrics."""
        query = "SELECT SUM(input_tokens), SUM(output_tokens), COUNT(*) FROM llm_cache"
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return {
                "total_input_tokens": int(row[0] or 0),
                "total_output_tokens": int(row[1] or 0),
                "total_queries": int(row[2] or 0),
            }
        finally:
            conn.close()
