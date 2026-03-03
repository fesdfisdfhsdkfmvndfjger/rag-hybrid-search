from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from pdf_loader import load_pdf_text
from chunker import chunk_text, choose_chunk_params
from embedder import embed_chunks, embed_query, rerank_chunks, rerank_scores
from vector_store import VectorStore
import persistence as pr
from answer_generation import generate_answer_full, verify_answer, generate_document_summary
from confidence import is_confident, confidence_label, confidence_color
from hybrid_search import HybridSearch


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    page: int
    section: Optional[str]
    vector_score: float
    rerank_score: float = 0.0


@dataclass
class RAGResult:
    answer: str
    verification: str           
    retrieved: List[RetrievedChunk]
    confidence_score: float
    confidence_label: str
    confidence_color: str
    search_mode: str
    elapsed_s: float
    pdf_name: str
    page_scores: Dict[int, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        chunks: List[dict],
        vector_store: VectorStore,
        pdf_name: str,
        cfg: dict,
        summary: str = ""
    ):
        self.chunks = chunks
        self.vector_store = vector_store
        self.pdf_name = pdf_name
        self.cfg = cfg
        self.summary = summary
        self.hybrid = HybridSearch(chunks, vector_store)

    def query(self, question: str, search_mode: str = "hybrid") -> RAGResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        
        # HyDE: Hypothetical Document Embeddings 
        search_query = question
        if cfg.get("use_hyde", False):
            hyde_prompt = f"Write a brief, factual paragraph that answers this question: {question}\nAnswer:"
            hypothetical_ans = generate_answer_full([], hyde_prompt, max_tokens=100)
            search_query = f"{question} {hypothetical_ans}"

        q_emb = embed_query(search_query)

        
        if search_mode == "vector":
            indices, scores = self.vector_store.search(q_emb, top_k=cfg["candidate_k"])
            indices, scores = indices.tolist(), scores.tolist()
        elif search_mode == "bm25":
            indices, scores = self.hybrid.bm25_search(search_query, top_k=cfg["candidate_k"])
        else:  
            indices, scores = self.hybrid.hybrid_search(
                search_query, q_emb, top_k=cfg["candidate_k"]
            )

        
        conf_score = float(scores[0]) if scores else 0.0
        if not is_confident(scores, abs_threshold=cfg["abs_threshold"]):
            return RAGResult(
                answer="Answer not found in the document.",
                verification="SUPPORTED",
                retrieved=[],
                confidence_score=conf_score,
                confidence_label=confidence_label(conf_score),
                confidence_color=confidence_color(conf_score),
                search_mode=search_mode,
                elapsed_s=time.perf_counter() - t0,
                pdf_name=self.pdf_name,
            )

        candidate_chunks = [self.chunks[i] for i in indices]
        candidate_texts  = [c["text"] for c in candidate_chunks]

        # Reranking
        ranked_indices = rerank_chunks(question, candidate_texts, top_n=cfg["final_k"])
        top_chunks = [candidate_chunks[i] for i in ranked_indices]

        
        r_scores = rerank_scores(question, [c["text"] for c in top_chunks])

      
        expanded = self._expand_context(top_chunks, window=cfg["context_window"])
        context_texts = [c["text"] for c in expanded]

       
        answer = generate_answer_full(context_texts, question, cfg["max_tokens"])

       
        verification = verify_answer(context_texts, answer)
        if verification == "UNSUPPORTED":
            answer = generate_answer_full(
                context_texts,
                f"Answer ONLY using exact phrases from the context. Question: {question}",
                cfg["max_tokens"],
            )
            verification = verify_answer(context_texts, answer)

       
        page_scores: Dict[int, float] = {}
        for chunk, rs in zip(top_chunks, r_scores):
            p = chunk["pages"][0]
            page_scores[p] = max(page_scores.get(p, 0.0), float(rs))

        if page_scores:
            max_ps = max(page_scores.values())
            if max_ps > 0:
                page_scores = {p: v / max_ps for p, v in page_scores.items()}

        retrieved = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                page=c["pages"][0],
                section=c.get("section"),
                vector_score=float(scores[i]) if i < len(scores) else 0.0,
                rerank_score=float(r_scores[ri]) if ri < len(r_scores) else 0.0,
            )
            for ri, (i, c) in enumerate(zip(ranked_indices, top_chunks))
        ]

        return RAGResult(
            answer=answer,
            verification=verification,
            retrieved=retrieved,
            confidence_score=conf_score,
            confidence_label=confidence_label(conf_score),
            confidence_color=confidence_color(conf_score),
            search_mode=search_mode,
            elapsed_s=time.perf_counter() - t0,
            pdf_name=self.pdf_name,
            page_scores=page_scores,
        )

    def _expand_context(self, top_chunks, window: int = 1):
        seen: set[int] = set()
        expanded = []
        for chunk in top_chunks:
            cid = chunk["chunk_id"]
            for i in range(cid - window, cid + window + 1):
                if 0 <= i < len(self.chunks) and i not in seen:
                    expanded.append(self.chunks[i])
                    seen.add(i)
        expanded.sort(key=lambda c: c["chunk_id"])
        return expanded


def get_file_hash(pdf_bytes: bytes) -> str:
    """Calculates an MD5 hash of the file bytes to prevent cache collisions."""
    return hashlib.md5(pdf_bytes).hexdigest()


def ingest_pdf(pdf_bytes: bytes, pdf_name: str, cache_dir: Path, cfg: dict) -> tuple[list, VectorStore, str]:
    """Ingests a PDF and caches results based on file hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_hash = get_file_hash(pdf_bytes)
    chunks_path = cache_dir / f"{file_hash}_chunks.pkl"
    faiss_path  = cache_dir / f"{file_hash}_faiss.index"

    
    tmp_pdf = cache_dir / f"{file_hash}.pdf"
    tmp_pdf.write_bytes(pdf_bytes)
    
    pages = load_pdf_text(str(tmp_pdf))
    total_words = sum(len(p["text"].split()) for p in pages)
    max_words, overlap = choose_chunk_params(total_words)
    chunks = chunk_text(pages, max_words, overlap)
    
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(chunk_texts)

    vs = VectorStore(embeddings)
    vs.save(faiss_path)
    pr.save_chunks(chunks, chunks_path)
    pr.save_embeddings(embeddings, cache_dir / f"{file_hash}_emb.npy")
    
    
    summary = generate_document_summary(chunk_texts)
    
    return chunks, vs, summary


def load_pdf_cache(pdf_bytes: bytes, cache_dir: Path) -> tuple[list, VectorStore] | None:
    """Loads cached chunks + index if they exist based on hash."""
    file_hash = get_file_hash(pdf_bytes)
    chunks_path = cache_dir / f"{file_hash}_chunks.pkl"
    faiss_path  = cache_dir / f"{file_hash}_faiss.index"
    if chunks_path.exists() and faiss_path.exists():
        return pr.load_chunks(chunks_path), VectorStore.load(faiss_path)
    return None