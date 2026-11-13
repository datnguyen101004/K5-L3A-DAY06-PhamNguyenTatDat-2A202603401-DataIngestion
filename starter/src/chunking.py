"""Implement fixed-size and structure-aware chunking."""
import re
import tiktoken
from .models import Chunk, DocumentPage

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def fixed_size(page: DocumentPage, size: int = 128, overlap: int = 16) -> list[Chunk]:
    """Split text into overlapping windows of `size` tokens and retain page metadata."""
    text = page.text.strip()
    if not text:
        return []

    enc = get_tokenizer()
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    if step <= 0:
        step = size

    chunk_idx = 0
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()

        if chunk_text:
            chunk_id = f"{page.document_id}_p{page.page}_c{chunk_idx}"
            chunk_metadata = dict(page.metadata)
            chunk_metadata.update({
                "chunk_index": chunk_idx,
                "token_count": len(chunk_tokens),
                "strategy": "fixed_128",
                "char_length": len(chunk_text),
            })

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
            chunk_idx += 1

        if end == total_tokens:
            break
        start += step

    return chunks


def structure_based(page: DocumentPage, max_tokens: int = 384) -> list[Chunk]:
    """Structure-aware chunking by headings and paragraphs.

    - Binds short headings to the subsequent paragraph.
    - Strips isolated page numbers.
    - Groups paragraphs up to `max_tokens`.
    - Splits oversized paragraphs at sentence boundaries.
    """
    text = page.text.strip()
    if not text:
        return []

    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        return []

    # Bind headings to following paragraph, ignore solitary page digits
    cleaned_paras: list[str] = []
    pending_prefix = ""

    for p in raw_paras:
        # Ignore isolated page digits
        if len(p) <= 3 and p.isdigit():
            continue
        # Heading detection: short single-line heading without period
        if len(p.splitlines()) == 1 and len(p) < 65 and not p.endswith("."):
            if pending_prefix:
                pending_prefix += "\n" + p
            else:
                pending_prefix = p
            continue
        if pending_prefix:
            cleaned_paras.append(pending_prefix + "\n\n" + p)
            pending_prefix = ""
        else:
            cleaned_paras.append(p)

    if pending_prefix:
        cleaned_paras.append(pending_prefix)

    enc = get_tokenizer()
    chunks: list[Chunk] = []
    chunk_idx = 0

    curr_paras: list[str] = []
    curr_tokens = 0

    def add_chunk(chunk_content: str):
        nonlocal chunk_idx
        c_text = chunk_content.strip()
        if not c_text:
            return
        c_tokens = len(enc.encode(c_text))
        chunk_id = f"{page.document_id}_p{page.page}_s{chunk_idx}"
        chunk_metadata = dict(page.metadata)
        chunk_metadata.update({
            "chunk_index": chunk_idx,
            "strategy": "structure_based",
            "token_count": c_tokens,
            "char_length": len(c_text),
        })
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=c_text,
                metadata=chunk_metadata,
            )
        )
        chunk_idx += 1

    for para in cleaned_paras:
        p_tokens = len(enc.encode(para))
        if p_tokens > max_tokens:
            # First flush current accumulated paragraphs
            if curr_paras:
                add_chunk("\n\n".join(curr_paras))
                curr_paras = []
                curr_tokens = 0

            # Split oversized paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_batch = []
            sent_tokens = 0
            for sent in sentences:
                s_tok = len(enc.encode(sent))
                if sent_tokens + s_tok > max_tokens and sent_batch:
                    add_chunk(" ".join(sent_batch))
                    sent_batch = [sent]
                    sent_tokens = s_tok
                else:
                    sent_batch.append(sent)
                    sent_tokens += s_tok
            if sent_batch:
                add_chunk(" ".join(sent_batch))
        elif curr_tokens + p_tokens > max_tokens and curr_paras:
            add_chunk("\n\n".join(curr_paras))
            curr_paras = [para]
            curr_tokens = p_tokens
        else:
            curr_paras.append(para)
            curr_tokens += p_tokens

    if curr_paras:
        add_chunk("\n\n".join(curr_paras))

    return chunks


# Alias for compatibility with template
by_paragraph = structure_based
