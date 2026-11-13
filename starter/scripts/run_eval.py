"""Run benchmark evaluation on stored vector embeddings."""
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.embeddings import AlibabaEmbedder
from src.evaluation import evaluate_query, summarize
from src.store import PgVectorStore


def run_evaluation(config_name: str = "D") -> dict:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    base_url = os.getenv("EMBEDDING_BASE_URL")

    is_hybrid = False
    # Map configuration
    if config_name in ["fixed_128", "A"]:
        table_strategy = "fixed_128"
        apply_filter = False
        config_label = "Config A: Fixed-size 128 (No Filter)"
        file_suffix = "fixed_128"
    elif config_name in ["structure_based", "B"]:
        table_strategy = "structure_based"
        apply_filter = False
        config_label = "Config B: Structure-based (No Filter)"
        file_suffix = "structure_based"
    elif config_name in ["structure_filtered", "C"]:
        table_strategy = "structure_based"
        apply_filter = True
        config_label = "Config C: Structure-based + Metadata Filter"
        file_suffix = "structure_filtered"
    elif config_name in ["hybrid", "D"]:
        table_strategy = "structure_based"
        apply_filter = True
        is_hybrid = True
        config_label = "Config D: Hybrid Retrieval (Dense Vector + BM25 RRF)"
        file_suffix = "hybrid"
    else:
        raise ValueError(f"Unknown config: {config_name}")

    questions_path = BASE_DIR.parent / "benchmark" / "questions.json"
    if not questions_path.exists():
        questions_path = BASE_DIR / "benchmark" / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 70)
    print(f"BENCHMARK RETRIEVAL EVALUATION - {config_label}")
    print(f"Total questions: {len(questions)} (Dev: 15, Test: 5)")
    print(f"Embedding Model: {model_name} | Hybrid Search: {'YES' if is_hybrid else 'NO'}")
    print("=" * 70)

    embedder = AlibabaEmbedder(model_name=model_name, base_url=base_url)
    store = PgVectorStore(database_url=db_url, dimensions=1024)

    all_traces = []

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        split = q["split"]
        filters = q.get("filter") if apply_filter else None

        # 1. Embed query
        q_vec = embedder.embed_query(q_text)

        # 2. Search top-3 (hybrid vs standard vector)
        if is_hybrid:
            hits = store.search_hybrid(
                strategy=table_strategy,
                query_text=q_text,
                query_vector=q_vec,
                top_k=3,
                filters=filters,
            )
        else:
            hits = store.search(
                strategy=table_strategy,
                query_vector=q_vec,
                top_k=3,
                filters=filters,
            )

        # 3. Evaluate hits
        trace = evaluate_query(q, hits)
        trace["type"] = q.get("type")
        trace["filter_applied"] = filters
        all_traces.append(trace)

        status_1 = "HIT" if trace["hit_at_1"] else "MISS"
        status_3 = "HIT" if trace["evidence_hit_at_3"] else "MISS"
        top_page = hits[0]["page"] if hits else "None"
        gold_page = q.get("gold_page")
        filter_str = f" [filter: {filters}]" if filters else ""
        print(f"[{split.upper()}] {q_id} | Gold: p{gold_page} | Top-1: p{top_page} | Hit@1: {status_1} | Hit@3: {status_3}{filter_str}")

    dev_traces = [t for t in all_traces if t["split"] == "dev"]
    test_traces = [t for t in all_traces if t["split"] == "test"]

    dev_summary = summarize(dev_traces)
    test_summary = summarize(test_traces)
    overall_summary = summarize(all_traces)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {config_label}")
    print("=" * 70)
    print(f"DEV  (15 questions) -> Hit@1: {dev_summary['hit_at_1_count']}/15 ({dev_summary['hit_at_1_rate']*100:.1f}%) | Evidence Hit@3: {dev_summary['evidence_hit_at_3_count']}/15 ({dev_summary['evidence_hit_at_3_rate']*100:.1f}%)")
    print(f"TEST ( 5 questions) -> Hit@1: {test_summary['hit_at_1_count']}/5  ({test_summary['hit_at_1_rate']*100:.1f}%) | Evidence Hit@3: {test_summary['evidence_hit_at_3_count']}/5  ({test_summary['evidence_hit_at_3_rate']*100:.1f}%)")
    print(f"ALL  (20 questions) -> Hit@1: {overall_summary['hit_at_1_count']}/20 ({overall_summary['hit_at_1_rate']*100:.1f}%) | Evidence Hit@3: {overall_summary['evidence_hit_at_3_count']}/20 ({overall_summary['evidence_hit_at_3_rate']*100:.1f}%)")

    # Save traces to results
    results_dir = BASE_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"trace_{file_suffix}.json"

    result_payload = {
        "config": config_name,
        "config_label": config_label,
        "is_hybrid": is_hybrid,
        "apply_filter": apply_filter,
        "embedding_model": model_name,
        "dev_metrics": dev_summary,
        "test_metrics": test_summary,
        "overall_metrics": overall_summary,
        "traces": all_traces,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed traces to: {out_file}\n")
    return result_payload


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "D"
    if arg in ["all", "--all"]:
        print("\nRUNNING ALL FOUR CONFIGURATIONS COMPARISON (A, B, C, D)...\n")
        res_a = run_evaluation("A")
        res_b = run_evaluation("B")
        res_c = run_evaluation("C")
        res_d = run_evaluation("D")

        print("=" * 85)
        print("FINAL COMPARISON TABLE ACROSS ALL 4 CONFIGURATIONS")
        print("=" * 85)
        print(f"{'Configuration':<40} | {'DEV Hit@1':<10} | {'DEV Hit@3':<10} | {'TEST Hit@1':<10} | {'TEST Hit@3':<10}")
        print("-" * 85)
        configs = [
            ("Config A: Fixed-size 128", res_a),
            ("Config B: Structure-based", res_b),
            ("Config C: Structure + Filter", res_c),
            ("Config D: Hybrid (Dense+BM25)", res_d),
        ]
        for label, r in configs:
            dev_1 = f"{r['dev_metrics']['hit_at_1_count']}/15"
            dev_3 = f"{r['dev_metrics']['evidence_hit_at_3_count']}/15"
            tst_1 = f"{r['test_metrics']['hit_at_1_count']}/5"
            tst_3 = f"{r['test_metrics']['evidence_hit_at_3_count']}/5"
            print(f"{label:<40} | {dev_1:<10} | {dev_3:<10} | {tst_1:<10} | {tst_3:<10}")
        print("=" * 85)
    else:
        run_evaluation(arg)


if __name__ == "__main__":
    main()
