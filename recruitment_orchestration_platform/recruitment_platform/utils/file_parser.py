import io
import re
from typing import Optional
from loguru import logger

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file given its bytes."""
    if not HAS_PYPDF:
        raise ImportError("pypdf is required for PDF parsing. Run: pip install pypdf")

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)
    logger.info(f"[PDFParser] Extracted {len(full_text)} characters from {len(reader.pages)} pages")
    return full_text


def clean_jd_text(raw_text: str) -> str:
    """Remove excessive whitespace, special characters, and normalize the text."""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """
    Route to the correct extractor based on file extension.
    Supports: .pdf, .txt
    """
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        raw = extract_text_from_pdf(file_bytes)
    elif ext in ("txt", "md"):
        raw = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Please upload a .pdf or .txt file.")

    return clean_jd_text(raw)
