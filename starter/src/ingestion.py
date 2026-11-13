"""Read and normalize source data."""
from pathlib import Path
import re
import unicodedata
import pypdf
from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    if not text:
        return ""
    # NFKC Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean up horizontal whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse excessive newlines to maximum 2 newlines (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = pypdf.PdfReader(str(path))
    pages: list[DocumentPage] = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        norm_text = normalize_text(raw_text)
        page_num = idx + 1

        if page_num <= 7:
            section = "front_matter"
        elif page_num <= 18:
            section = "overview"
        elif page_num <= 35:
            section = "product_regulation"
        elif page_num <= 48:
            section = "provider_regulation"
        else:
            section = "lessons"

        metadata = {
            "char_count": len(norm_text),
            "is_empty": len(norm_text.strip()) == 0,
            "document_id": document_id,
            "source": source,
            "year": 2010,
            "document_type": "working_paper",
            "section": section,
        }

        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=page_num,
            text=norm_text,
            metadata=metadata
        ))

    return pages
