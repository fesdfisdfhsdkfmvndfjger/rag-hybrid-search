
# 🔍 RAG Hybrid Search

> **A production-grade, 100% offline Retrieval-Augmented Generation (RAG) workspace.** > Engineered for forensic traceability, zero-data-leakage, and extreme retrieval precision using True Hybrid Search (FAISS + BM25) and Cross-Encoder Reranking.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama)](https://ollama.ai)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-5C5C5C)](https://github.com/facebookresearch/faiss)
[![Author](https://img.shields.io/badge/Author-Aditya_Vijay-blue?logo=linkedin)](https://www.linkedin.com/in/adityavijay21/)

## 📖 Overview

Standard RAG implementations suffer from critical flaws: they **hallucinate**, they **lose context**, and they **send sensitive documents to cloud APIs**. 

**`rag-hybrid-search`** solves this by providing an air-gapped, verifiable AI research engine. It doesn't just generate answers; it proves them. By fusing dense vector search with sparse lexical search, reranking candidates with a Cross-Encoder, and enforcing a strict dual-gate confidence check, this pipeline guarantees that every answer is factually grounded and forensically traceable to exact page numbers.


## ⚡ Performance Summary

Optimized for consumer hardware without sacrificing enterprise accuracy.

| Metric | Performance |
| :--- | :--- |
| **Retrieval & Rerank** | `< 1 second` (even on CPU) |
| **Prompt Processing** | Near-instant via Ollama `keep_alive` memory persistence |
| **Document Support** | Tested seamlessly with 100+ page technical PDFs |
| **Data Privacy** | `100% Offline` (Zero API calls to external servers) |

---

## 🏗️ The Workflow & Architecture

This project ditches naive approaches (like simple character splitters) in favor of a robust, multi-layer deterministic pipeline.

### 1. Ingestion Phase
* **PDF Parsing:** Extracts text using `PyMuPDF`, capturing exact page metadata for UI rendering.
* **Semantic Chunking:** `chunker.py` uses `NLTK` to split text at natural sentence boundaries and uses Regex to detect document headings/sections. Dynamic chunking intelligently adjusts sizes based on total document length.
* **Embedding & Indexing:** Chunks are embedded using HuggingFace `all-MiniLM-L6-v2` and pushed into a highly stable `FAISS` index.
* **MD5 Hashing & Caching:** The raw PDF bytes are hashed. This acts as the cache key, ensuring instant reloads if the same document is uploaded again, while preventing collisions from identically named files.
* **Agentic Summarization:** Upon upload, an LLM agent automatically reads the first few pages and greets the user with a concise 3-bullet-point executive summary.

### 2. Retrieval & Generation Phase (Multi-Layer)
* **Layer 0: Query Expansion (HyDE):** Optionally, the LLM generates a "hypothetical answer" to the user's query. The query and hypothetical answer are embedded together to drastically improve vector recall.
* **Layer 1: Hybrid Search Formulation:** * *Vector Search (FAISS):* Captures semantic meaning (e.g., "revenue" matches "earnings").
  * *Lexical Search (BM25):* Captures exact keywords (e.g., finding specific proper nouns or serial numbers like "ID: 994B").
  * *Fusion:* Scores from both algorithms are Min-Max normalized (0 to 1) and fused using a weighted average to retrieve the top `candidate_k` chunks.
* **Layer 2: Cross-Encoder Reranking:** The broad candidates are passed to a `TinyBERT` Cross-Encoder. It processes the Query and Document *simultaneously* to output a highly accurate relevance score, narrowing down to the `final_k` chunks.
* **Layer 3: Dual-Gate Confidence Check:** If the top chunk's similarity score falls below a strict configurable threshold (e.g., `0.25`), the pipeline immediately halts and returns *"Answer not found"*, deterministically preventing hallucinations.
* **Layer 4: Answer Verification:** The LLM generates the final answer. A secondary LLM agent then cross-references the answer against the retrieved context to label the response as `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.

---

## 🛠️ Tech Stack (The "Why")

Every technology in this stack was chosen to maximize local performance, hardware compatibility, and retrieval precision.

| Technology | Purpose | Why this specific tool? |
| :--- | :--- | :--- |
| **Ollama (`phi3`)** | LLM Engine | 100% offline inference. Microsoft's `phi-3` offers exceptional reasoning capabilities while running flawlessly on standard consumer GPUs and Apple M-Series chips. |
| **FAISS (`IndexFlatIP`)** | Vector Database | Upgrading to HNSW often causes segmentation faults on Apple Silicon due to C++ memory thread conflicts. `IndexFlatIP` requires contiguous NumPy arrays, ensuring 100% stable, mathematically exact inner-product (cosine) searches. |
| **`rank_bm25`** | Sparse Lexical Search | Standard embeddings fail at exact keyword matching. BM25 ensures critical exact-match terms are heavily weighted in the hybrid fusion. |
| **SentenceTransformers** | Embeddings & Reranking | `all-MiniLM-L6-v2` is exceptionally fast for dense vectors. `ms-marco-TinyBERT-L-2-v2` serves as a lightweight, lightning-fast Cross-Encoder for precision reranking. |
| **PyMuPDF (`fitz`)** | Document Parsing | Significantly faster than `PyPDF2` or `pdfminer`. It accurately preserves reading order and maps bounding boxes for the Streamlit split-screen UI. |
| **Streamlit** | Frontend UI | Allowed for rapid prototyping while supporting aggressive custom CSS to create a premium, Gemini-style floating chat interface. |

---

## ✨ Advanced User Features

* **🎯 Verifiable Citations:** Responses include interactive `📄 p.12` pills. Clicking them instantly jumps the native PDF viewer to the exact referenced page.
* **📊 Retrieval Transparency:** Users can view the exact Vector Scores and Cross-Encoder Rerank scores for every chunk used in the context window.
* **📥 Markdown Reports:** 1-click export of your entire research conversation, complete with source citations, into a clean Markdown file.
* **⚙️ Complete UI Control:** Adjust Search Mode (Hybrid/Vector/BM25), Candidate K, Final K, and Confidence Thresholds directly from the Streamlit sidebar.

---

## 🚀 Installation & Setup

> [!WARNING]
> This project requires **Ollama** to be installed and running on your system.
> 1. **Download Ollama**: [Ollama.ai](https://ollama.ai/)
> 2. **Pull the model**: Run `ollama pull phi3` in your terminal.

**1. Clone the repository**
```bash
git clone https://github.com/adityavijay21/rag-hybrid-search.git
cd rag-hybrid-search

```

**2. Set up a virtual environment (Python 3.11+ recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

---

## 💻 Usage Guide

Start the local workspace UI:

```bash
cd src
streamlit run app.py

```

1. Open the provided `localhost` URL in your browser.
2. Upload one or more PDFs via the drag-and-drop zone.
3. Open the **⚙️ Engine Settings** sidebar to tune the pipeline:
* **Search Mode:** Toggle between `Hybrid`, `Vector`, or `BM25`.
* **Use HyDE:** Enable query expansion for complex, conceptual questions.
* **Confidence Filter:** Lower this threshold (e.g., `0.10`) for keyword-dense documents like resumes; raise it (e.g., `0.30`) for strict legal/technical compliance.


4. Ask questions, click citations to verify context, and download your final report.

---

## 📂 Repository Structure

```text
rag-hybrid-search/
├── src/
│   ├── app.py                 # Streamlit UI, State Management, Custom CSS
│   ├── rag_pipeline.py        # Core RAG logic, HyDE, MD5 Hashing, Agentic Summary
│   ├── embedder.py            # Local HF Embeddings & Cross-Encoder Initialization
│   ├── vector_store.py        # Contiguous FAISS IndexFlatIP Abstraction
│   ├── hybrid_search.py       # BM25 + FAISS Normalized Fusion Math
│   ├── answer_generation.py   # Ollama API Integration & Verification loop
│   ├── chunker.py             # NLTK Sentence-aware & Regex-heading parsing
│   ├── confidence.py          # Dual-gate Cosine Similarity thresholds
│   ├── pdf_loader.py          # Fast PyMuPDF multithreaded extraction
│   └── persistence.py         # Atomic read/write cache handlers
├── requirements.txt
└── README.md

```

---

## 👨‍💻 Author

**Aditya Vijay**

* **GitHub:** [@adityavijay21](https://github.com/adityavijay21)
* **LinkedIn:** [Aditya Vijay](https://www.linkedin.com/in/adityavijay21/?skipRedirect=true)

If you find this project helpful, please consider giving it a ⭐ on GitHub! It helps the project grow and reach more developers.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/adityavijay21/rag-hybrid-search/issues).

## 📄 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.
