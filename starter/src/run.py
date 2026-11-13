"""Pipeline entry point for Data Ingestion and Embedding."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.chunking import fixed_size, structure_based
from src.embeddings import AlibabaEmbedder
from src.ingestion import load_pdf
from src.store import PgVectorStore


def run_strategy(strategy: str = "structure_based") -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)

    db_url = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
    model_name = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    base_url = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://ws-b0ojoo4bja1v9457.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )
    api_key = os.getenv("OPENAI_API_KEY")

    print("=" * 60)
    print(f"INGESTION & EMBEDDING - Strategy: {strategy}")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Database: {db_url}")

    pdf_path = BASE_DIR / "data" / "WPS5288.pdf"
    if not pdf_path.exists():
        pdf_path = BASE_DIR.parent / "data" / "WPS5288.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"\n[1] Loading PDF from: {pdf_path}")
    pages = load_pdf(pdf_path, document_id="WPS5288", source="WPS5288.pdf")
    non_empty_pages = [p for p in pages if not p.metadata.get("is_empty", False)]
    print(f"Total physical pages: {len(pages)}")
    print(f"Content pages with text: {len(non_empty_pages)}")

    print(f"\n[2] Chunking with strategy '{strategy}'...")
    chunks = []
    for p in pages:
        if strategy == "fixed_128":
            page_chunks = fixed_size(p, size=128, overlap=16)
        elif strategy == "structure_based":
            page_chunks = structure_based(p, max_tokens=384)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        chunks.extend(page_chunks)

    total_chunks = len(chunks)
    total_tokens = sum(c.metadata.get("token_count", 0) for c in chunks)
    total_chars = sum(len(c.text) for c in chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks else 0

    print(f"Total chunks created: {total_chunks}")
    print(f"Total tokens: {total_tokens} (avg {avg_tokens:.1f} tokens/chunk)")
    print(f"Total characters: {total_chars}")

    print(f"\n[3] Generating embeddings via Alibaba Cloud model '{model_name}'...")
    embedder = AlibabaEmbedder(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        dimensions=1024,
        batch_size=16,
    )

    texts = [c.text for c in chunks]
    start_time = time.time()
    vectors = embedder.embed_documents(texts)
    elapsed = time.time() - start_time

    print(f"Generated {len(vectors)} embeddings in {elapsed:.2f}s ({len(vectors)/elapsed:.2f} chunks/s)")

    print(f"\n[4] Storing in PgVectorStore (PostgreSQL + pgvector)...")
    store = PgVectorStore(database_url=db_url, dimensions=embedder.dimensions)
    store.create_schema()
    store.upsert(strategy=strategy, chunks=chunks, vectors=vectors)
    print(f"Successfully stored {len(chunks)} chunks and vectors under strategy '{strategy}'.")

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{strategy}_summary.json"

    summary_data = {
        "strategy": strategy,
        "document_id": "WPS5288",
        "source": "WPS5288.pdf",
        "total_pages": len(pages),
        "content_pages": len(non_empty_pages),
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": round(avg_tokens, 2),
        "total_chars": total_chars,
        "embedding_model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "embedding_time_seconds": round(elapsed, 2),
        "sample_chunks": [
            {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "token_count": c.metadata.get("token_count"),
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
            }
            for c in chunks[:5]
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Also sync to root results dir
    root_results = BASE_DIR.parent / "results"
    root_results.mkdir(parents=True, exist_ok=True)
    with open(root_results / f"{strategy}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n[5] Saved summary to: {summary_path}")
    print("\n" + "=" * 60)
    print(f"STRATEGY '{strategy}' COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Ingestion and Embedding Pipeline")
    parser.add_argument(
        "--strategy",
        choices=["fixed_128", "structure_based", "all"],
        default="structure_based",
        help="Chunking strategy to run",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        run_strategy("fixed_128")
        run_strategy("structure_based")
    else:
        run_strategy(args.strategy)


if __name__ == "__main__":
    main()
