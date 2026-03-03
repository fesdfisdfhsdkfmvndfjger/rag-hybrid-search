from __future__ import annotations
import json
import requests
from typing import Iterator

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

_ANSWER_PROMPT = """\
You are a precise document question-answering assistant.

Rules:
1. Answer ONLY using facts explicitly stated in the provided context.
2. Be concise and direct — no preamble.
3. Do NOT infer, speculate, or add external knowledge.
4. If the answer is not in the context, output exactly:
   "Answer not found in the document."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""

_VERIFY_PROMPT = """\
You are a strict fact-checker. Given a context and an answer, determine if every claim in the answer is directly supported by the context.

Context:
{context}

Answer to verify:
{answer}

Respond with ONLY one of:
- SUPPORTED: (if every claim is directly in the context)
- UNSUPPORTED: (if any claim is not in the context)
- PARTIAL: (if some claims are supported but others are not)

Verdict:"""


def _call_ollama(prompt: str, max_tokens: int, temperature: float = 0.1) -> str:
    """Non-streaming call, returns full text."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature,
                            "top_p": 0.9, "repeat_penalty": 1.1},
                "keep_alive": "15m",
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"[ERROR] {e}"


def generate_answer_full(
    context_chunks: list[str],
    question: str,
    max_tokens: int = 512,
) -> str:
    """Non-streaming version — returns complete answer string."""
    seen: set[str] = set()
    unique = [c for c in context_chunks if not (c.strip() in seen or seen.add(c.strip()))]
    context = "\n\n".join(unique)
    prompt = _ANSWER_PROMPT.format(context=context, question=question)
    return _call_ollama(prompt, max_tokens)


def verify_answer(context_chunks: list[str], answer: str) -> str:
    """
    Runs a second-pass verification.
    Returns one of: "SUPPORTED", "UNSUPPORTED", "PARTIAL"
    """
    if answer.startswith("[ERROR]") or "Answer not found" in answer:
        return "SUPPORTED"   # nothing to verify
    context = "\n\n".join(context_chunks[:5])  # limit tokens
    prompt = _VERIFY_PROMPT.format(context=context, answer=answer)
    verdict = _call_ollama(prompt, max_tokens=10, temperature=0.0).upper()
    for label in ("SUPPORTED", "UNSUPPORTED", "PARTIAL"):
        if label in verdict:
            return label
    return "SUPPORTED"


def generate_document_summary(context_chunks: list[str]) -> str:
    """Agentic generation of a 3-bullet summary upon document upload using local LLM."""
    context = "\n\n".join(context_chunks[:4]) # Use beginning of document
    prompt = f"Based on the following document introduction, provide a 3-bullet-point summary of what this document is about. Be extremely concise.\n\n{context}\n\nSummary:"
    return _call_ollama(prompt, max_tokens=150, temperature=0.3)