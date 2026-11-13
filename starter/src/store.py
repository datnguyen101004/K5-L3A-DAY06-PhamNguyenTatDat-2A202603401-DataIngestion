"""Store and search vectors using PostgreSQL + pgvector."""
import json
import re
from typing import Any
import psycopg
from rank_bm25 import BM25Okapi
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int = 1024) -> None:
        """Retain connection settings; validate model dimensions."""
        self.database_url = database_url
        self.dimensions = dimensions

    def get_connection(self):
        return psycopg.connect(self.database_url, autocommit=True)

    def create_schema(self) -> None:
        """Create table with vector column and source/page/filter metadata."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chunk_id, strategy)
                );
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_strategy 
                ON document_chunks (document_id, strategy);
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_metadata 
                ON document_chunks USING gin (metadata);
                """)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions and write stable IDs without duplicates.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")

        if not chunks:
            return

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                doc_ids = list({c.document_id for c in chunks})
                cur.execute(
                    "DELETE FROM document_chunks WHERE strategy = %s AND document_id = ANY(%s);",
                    (strategy, doc_ids),
                )

                insert_sql = """
                INSERT INTO document_chunks (
                    chunk_id, strategy, document_id, source, page, text, metadata, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                params = []
                for chunk, vector in zip(chunks, vectors):
                    if len(vector) != self.dimensions:
                        raise ValueError(
                            f"Vector dimension {len(vector)} does not match store dimension {self.dimensions}"
                        )
                    vec_str = "[" + ",".join(map(str, vector)) + "]"
                    meta_json = json.dumps(chunk.metadata) if isinstance(chunk.metadata, dict) else str(chunk.metadata)
                    params.append((
                        chunk.chunk_id,
                        strategy,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        meta_json,
                        vec_str,
                    ))

                cur.executemany(insert_sql, params)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k."""
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: {len(query_vector)} vs {self.dimensions}"
            )

        vec_str = "[" + ",".join(map(str, query_vector)) + "]"

        sql = """
        SELECT chunk_id, document_id, source, page, text, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        WHERE strategy = %s
        """
        params: list[Any] = [vec_str, strategy]

        if filters:
            for key, val in filters.items():
                if key in ["document_id", "source"]:
                    sql += f" AND {key} = %s"
                    params.append(val)
                elif key == "page":
                    sql += " AND page = %s"
                    params.append(int(val))
                else:
                    sql += " AND metadata->>%s = %s"
                    params.extend([key, str(val)])

        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([vec_str, top_k])

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "chunk_id": r[0],
                        "document_id": r[1],
                        "source": r[2],
                        "page": r[3],
                        "text": r[4],
                        "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                        "score": float(r[6]),
                    })
                return results

    def search_hybrid(
        self,
        strategy: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Combine dense vector similarity with BM25 keyword matching via Reciprocal Rank Fusion (RRF)."""
        # 1. Dense search across candidate pool
        dense_results = self.search(
            strategy=strategy,
            query_vector=query_vector,
            top_k=20,
            filters=filters,
        )
        dense_rank = {item["chunk_id"]: rank for rank, item in enumerate(dense_results, start=1)}

        # 2. Retrieve candidates matching strategy & filter for BM25
        sql = "SELECT chunk_id, document_id, source, page, text, metadata FROM document_chunks WHERE strategy = %s"
        params: list[Any] = [strategy]
        if filters:
            for key, val in filters.items():
                if key in ["document_id", "source"]:
                    sql += f" AND {key} = %s"
                    params.append(val)
                elif key == "page":
                    sql += " AND page = %s"
                    params.append(int(val))
                else:
                    sql += " AND metadata->>%s = %s"
                    params.extend([key, str(val)])

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        if not rows:
            return []

        candidates = [
            {
                "chunk_id": r[0],
                "document_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else json.loads(r[5]),
            }
            for r in rows
        ]

        # 3. BM25 calculation
        def tokenize(txt: str) -> list[str]:
            return re.findall(r"\w+", txt.lower())

        corpus_tokens = [tokenize(c["text"]) for c in candidates]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = tokenize(query_text)
        bm25_scores = bm25.get_scores(query_tokens)

        # Rank by BM25
        scored_candidates = sorted(zip(candidates, bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_rank = {cand["chunk_id"]: rank for rank, (cand, _) in enumerate(scored_candidates, start=1)}

        # 4. Reciprocal Rank Fusion (RRF)
        candidate_map = {c["chunk_id"]: c for c in candidates}
        rrf_scores = {}
        for cid in candidate_map:
            d_rank = dense_rank.get(cid, 999)
            b_rank = bm25_rank.get(cid, 999)
            score = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))
            rrf_scores[cid] = score

        # Top-k by RRF
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_results = []
        for cid, score in sorted_rrf:
            cand = dict(candidate_map[cid])
            cand["score"] = float(score)
            cand["dense_rank"] = dense_rank.get(cid, None)
            cand["bm25_rank"] = bm25_rank.get(cid, None)
            final_results.append(cand)

        return final_results
