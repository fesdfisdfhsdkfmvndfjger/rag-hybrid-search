# Ask_my_docs: Advanced Local RAG System

A professional-grade Retrieval-Augmented Generation (RAG) system built with Python, emphasizing performance, accuracy, and user control. This project demonstrates a production-ready approach to local LLM knowledge retrieval.

## Key Features

- **Multi-Layer Retrieval**: Implements a two-stage retrieval process.
    - **Layer 1 (Vector Search)**: Broad retrieval of 15 candidate chunks using FAISS.
    - **Layer 2 (Re-ranking)**: Precision scoring using a Cross-Encoder (`TinyBERT`) to narrow down to the top 2 most relevant results.
- **Optimized for Speed**: 
    - **Dynamic Chunking**: Intelligently adjusts chunk sizes based on document length.
    - **Phi-3 Integration**: Uses Microsoft's lightweight Phi-3 model for near-instant responses on consumer hardware.
    - **Model Persistence**: Uses `keep_alive` to keep the model in memory.
- **Interactive Streaming**: Responses are streamed live to the terminal.
- **User Interrupt**: Stop AI generation immediately by pressing 'q' or 'x' (Windows).

## Technical Stack

- **LLM Engine**: Ollama (Phi-3)
- **Vectors**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Re-ranking**: Cross-Encoders (`ms-marco-TinyBERT-L-2-v2`)
- **PDF Extraction**: PyMuPDF

## Setup & Installation

> [!WARNING]
> This project requires **Ollama** to be installed and running on your system.
> 1. **Download Ollama**: [Ollama.ai](https://ollama.ai/)
> 2. **Pull the model**: Run `ollama pull phi3` in your terminal.

### Project Setup
1. **Clone the repository**:
   ```bash
   git clone <your-bitbucket-url>
   cd ask_my_docs
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Application**:
   ```bash
   python src/main.py
   ```

## Configuration

Settings can be adjusted in `config/param.xml`:
- `candidate_k`: Initial retrieval depth.
- `final_k`: Context sent to LLM after re-ranking.
- `max_tokens`: Answer size limit.
- `new_pdf`: Set to `True` to process a new PDF. It will automatically detect the first `.pdf` file in the `/data` directory.

## Performance Summary

| Metric | Performance |
| :--- | :--- |
| **Retrieval & Rerank** | < 1 second |
| **Prompt Processing** | Optimized for low-latency |
| **Document Support** | Tested with 80+ page technical PDFs |
