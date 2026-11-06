# llm-clg-project-

# Student Portfolio LLM — RAG & React Integration

A full-stack, retrieval-augmented conversational AI portfolio assistant for **Prem Kumar** (B.Tech in Artificial Intelligence and Data Science, K S Rangasamy College of Technology).

---

## 🚀 Key Features

- **Dual-Mode Inference & Auto-Switching:**
  - **Direct LLM**: Answers stable profile and identity questions directly using fine-tuned model weights.
  - **LLM + RAG**: Dynamically retrieves up-to-date facts (projects, skills, certifications, achievements) from the portfolio knowledge base (`prem_rag_knowledge_base.pdf`).
  - **Intelligent Auto-Router**: Automatically analyzes question intent and switches modes without manual selection.
- **Strict Anti-Hallucination Guardrails**:
  - Unconfirmed facts (e.g. CGPA, internships, salary expectations) are handled with strict refusal rather than invented claims.
- **Interactive Execution Trace ("Honing")**:
  - Expandable real-time drawer showing query intent parsing, FAISS vector retrieval, similarity score filtering ($\ge 0.25$), and prompt synthesis.
- **FastAPI Backend**:
  - High-performance API with endpoints for `/chat`, `/health`, and `/index/rebuild`.
- **Modern React & Vite Frontend**:
  - Sleek dark theme, live auto-switching pill indicators, markdown formatting, clickable project links, and one-click copy.
- **Automated Test Suite**:
  - 11 unit & integration test cases validating PDF extraction, vector search, grounded answering, refusal guardrails, and auto-routing.

---

## 🏗️ Architecture

```text
Visitor
  │
React Portfolio Frontend (Vite)
  │ POST /chat
FastAPI Backend
  │
  ├── Intent Auto-Router
  │     ├── Stable Profile Intent ──> Direct LLM (Qwen2.5-1.5B + LoRA)
  │     └── Dynamic Fact Intent  ──> RAG Pipeline
  │                                    │
  │                          prem_rag_knowledge_base.pdf
  │                                    │
  │                              PyMuPDF Chunks
  │                                    │
  │                           all-MiniLM-L6-v2 Embeddings
  │                                    │
  │                           FAISS Inner Product Search
  │                                    │
  │                          Threshold Filter (score >= 0.25)
  │                                    │
  │                         Grounded Prompt Assembly
  │                                    │
  │                          Qwen2.5-1.5B + LoRA Adapter
  │                                    │
  +──────────────────────── Grounded JSON Response
```

---

## 📁 Project Structure

```text
llm-mini-project/
├── backend/
│   ├── app.py                 # FastAPI backend & Auto-router
│   ├── rag.py                 # PDF parsing, embedding & FAISS vector store
│   ├── llm.py                 # Qwen2.5 + LoRA grounded inference engine
│   ├── requirements.txt       # Python dependencies
│   ├── data/
│   │   └── prem_rag_knowledge_base.pdf
│   └── vector_store/
│       ├── portfolio.faiss    # FAISS vector index
│       └── chunks.json        # Extracted text passages
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PortfolioChat.jsx  # Main chat interface
│   │   │   ├── ThinkingTrace.jsx  # Collapsible Honing/Process drawer
│   │   │   ├── ChatMessage.jsx    # Formatted chat bubble & citations
│   │   │   └── ModeToggle.jsx     # Auto-routing indicator
│   │   ├── App.jsx
│   │   └── index.css          # Dark-theme styling & resets
│   ├── package.json
│   └── vite.config.js
├── tests/
│   └── test_portfolio_rag.py  # 11 automated test cases
├── prem_portfolio_lora/       # Trained LoRA adapter files & tokenizer
├── prem_finetune_dataset.jsonl# Fine-tuning dataset
└── README.md
```

---

## 🛠️ Quick Start

### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build RAG FAISS index
python rag.py

# Start FastAPI server
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Backend docs available at: `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser.

### 3. Run Automated Tests

```bash
python -m unittest tests/test_portfolio_rag.py
```
