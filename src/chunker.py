from __future__ import annotations
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Tuple

for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[\s\.\)]+[A-Z]"
    r"|[A-Z][A-Z\s]{4,}$"
    r"|(?:Chapter|Section|Part)\s+\d+"
    r")"
)

def detect_section(sentence: str) -> str | None:
    s = sentence.strip()
    if len(s) > 120:
        return None
    if HEADING_RE.match(s):
        return s
    return None

def choose_chunk_params(total_words: int) -> Tuple[int, int]:
    if total_words < 500:       return 80, 1
    elif total_words < 2_000:   return 120, 2
    elif total_words < 10_000:  return 160, 2
    else:                       return 180, 3

def chunk_text(pages: List[Dict], max_words: int, overlap_sentences: int) -> List[Dict]:
    """Returns list of {"text", "pages", "chunk_id", "section"} dicts."""
    all_chunks: List[Dict] = []
    current_section: str | None = None

    for page_data in pages:
        page_num = page_data["page"]
        sentences = [s.strip() for s in sent_tokenize(page_data["text"]) if s.strip()]
        if not sentences:
            continue

        current_chunk: List[str] = []
        current_words = 0
        chunk_section: str | None = current_section

        for sentence in sentences:
            heading = detect_section(sentence)
            if heading:
                current_section = heading
                chunk_section = heading

            w = len(sentence.split())
            if current_words + w > max_words and current_chunk:
                _flush(all_chunks, current_chunk, page_num, chunk_section)
                tail = current_chunk[-overlap_sentences:] if overlap_sentences else []
                current_chunk = list(tail)
                current_words = sum(len(s.split()) for s in current_chunk)
                chunk_section = current_section

            current_chunk.append(sentence)
            current_words += w

        if current_chunk:
            _flush(all_chunks, current_chunk, page_num, chunk_section)

    final: List[Dict] = []
    for chunk in all_chunks:
        if len(chunk["text"].split()) >= 5:
            chunk["chunk_id"] = len(final)
            final.append(chunk)
    return final

def _flush(all_chunks, sentences, page_num, section):
    text = " ".join(sentences).strip()
    if text:
        all_chunks.append({"text": text, "pages": [page_num], "section": section})