from __future__ import annotations
import re
import fitz
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\u024F]", " ", text)
    return text.strip()

def _process_page(args) -> Dict | None:
    page_index, page_bytes = args
    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]
    text = page.get_text("text")
    doc.close()
    if not text.strip():
        return None
    return {"page": page_index + 1, "text": clean_text(text)}

def load_pdf_text(pdf_path: str, max_workers: int = 4) -> List[Dict]:
    """Returns list of {"page": int, "text": str} dicts."""
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        page_bytes_list = []
        for i in range(total_pages):
            single = fitz.open()
            single.insert_pdf(doc, from_page=i, to_page=i)
            page_bytes_list.append((i, single.tobytes()))
            single.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_page, page_bytes_list))

    pages = [r for r in results if r is not None]
    if not pages:
        raise ValueError("PDF contains no extractable text (may be image-based).")
    return pages