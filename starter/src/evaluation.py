"""Evaluate retrieved evidence and preserve per-query traces."""
from typing import Any


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold_source = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote", "").strip().lower()

    hit_at_1 = False
    evidence_hit_at_3 = False
    retrieved_chunks = []

    for rank, hit in enumerate(hits, start=1):
        chunk_text = hit.get("text", "")
        chunk_page = hit.get("page")
        chunk_source = hit.get("document_id") or hit.get("source")

        source_match = (not gold_source) or (gold_source in str(chunk_source))
        page_match = (gold_page is None) or (chunk_page == gold_page)
        
        # Robust evidence matching: normalize internal newlines and spaces
        clean_evidence = " ".join(evidence_quote.lower().split())
        clean_chunk = " ".join(chunk_text.lower().split())
        evidence_match = (not clean_evidence) or (clean_evidence in clean_chunk)

        is_hit = source_match and page_match and evidence_match

        if rank == 1 and is_hit:
            hit_at_1 = True
        if rank <= 3 and is_hit:
            evidence_hit_at_3 = True

        retrieved_chunks.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "text": chunk_text,
            "metadata": hit.get("metadata"),
            "is_hit": is_hit,
        })

    return {
        "question_id": question.get("id"),
        "question": question.get("question"),
        "split": question.get("split", "dev"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "top_hits": retrieved_chunks,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    total = len(traces)
    if total == 0:
        return {"total": 0, "hit_at_1": 0, "evidence_hit_at_3": 0}

    hit_1_count = sum(1 for t in traces if t.get("hit_at_1"))
    hit_3_count = sum(1 for t in traces if t.get("evidence_hit_at_3"))

    return {
        "total": total,
        "hit_at_1_count": hit_1_count,
        "hit_at_1_rate": hit_1_count / total,
        "evidence_hit_at_3_count": hit_3_count,
        "evidence_hit_at_3_rate": hit_3_count / total,
    }
