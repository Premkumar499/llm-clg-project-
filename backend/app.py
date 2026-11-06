import time
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import retrieve_context, build_index
from llm import generate_answer

app = FastAPI(
    title="Student Portfolio LLM API",
    description="Fine-tuned Qwen2.5-1.5B + LoRA + FAISS RAG Portfolio Assistant for Prem Kumar",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_route(question: str) -> Tuple[str, str]:
    """
    Intelligently inspects the input question and automatically switches between
    'direct' (stable identity / academic profile) and 'rag' (changing portfolio PDF facts).
    """
    q_norm = question.strip().lower()

    # RAG keywords (projects, skills, achievements, certifications, experience, links)
    rag_patterns = [
        "project", "built", "develop", "created", "deepfake", "kllm", "signbridge",
        "jarvis", "youtube", "theatre", "elite studio", "ai security", "audio",
        "skill", "programming", "language", "languages", "python", "java", "react", "fastapi",
        "machine learning", "tools", "database", "sql", "pytorch", "frontend", "backend",
        "achievement", "hackathon", "contest", "ambassador", "competition", "guvi", "league",
        "certification", "certificate", "course", "intern", "experience", "work",
        "cgpa", "marks", "gpa", "salary", "career", "goal", "target", "github", "link",
        "repository", "repo", "public", "portfolio"
    ]

    for p in rag_patterns:
        if p in q_norm:
            return "rag", f"Query detected dynamic portfolio topic ('{p}'). Routed to LLM + RAG."

    # Direct stable identity patterns
    direct_patterns = [
        "who are you", "introduce yourself", "tell me about yourself", "your name", "full name",
        "what are you called", "which college", "what college", "your college", "ksrct",
        "your degree", "which degree", "what are you studying", "what do you study", "your department",
        "where are you from", "where do you live", "your location", "graduation year",
        "when do you graduate", "expected graduation", "hello", "hi", "hey", "how are you"
    ]

    for p in direct_patterns:
        if p in q_norm:
            return "direct", f"Query matches stable profile/identity ('{p}'). Routed to Direct LLM."

    # General / ambiguous: evaluate FAISS vector similarity
    try:
        results = retrieve_context(question, top_k=1)
        if results and results[0]["score"] >= 0.28:
            return "rag", f"Vector similarity ({results[0]['score']:.2f}) indicates relevance in knowledge base. Routed to LLM + RAG."
    except Exception:
        pass

    return "direct", "Query answered directly using fine-tuned model parameters."


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    mode: Optional[str] = Field(default="auto", description="Execution mode: 'auto', 'rag', or 'direct'")
    top_k: Optional[int] = Field(default=6, ge=1, le=10)
    threshold: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)


class ProcessStep(BaseModel):
    title: str
    detail: Optional[str] = None
    status: str = "completed"
    badge: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: List[str]
    process: List[ProcessStep]
    latency_ms: float
    auto_routed: bool = True
    routing_reason: Optional[str] = None


class ReindexResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Student Portfolio LLM API",
        "author": "Prem Kumar",
        "version": "1.0.0"
    }


@app.post("/index/rebuild", response_model=ReindexResponse)
def reindex_portfolio():
    try:
        count = build_index()
        return ReindexResponse(
            status="ok",
            chunks_indexed=count,
            message="Knowledge base re-indexed successfully."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(exc)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    t_start = time.time()
    question = request.question.strip()
    requested_mode = (request.mode or "auto").lower()
    process_steps: List[ProcessStep] = []

    # Step 1: Input analysis & Intent-based Automatic Routing
    if requested_mode == "auto":
        active_mode, routing_reason = classify_route(question)
        auto_routed = True
        process_steps.append(ProcessStep(
            title=f"Auto-Router: Switched to {'LLM + RAG' if active_mode == 'rag' else 'Direct LLM'}",
            detail=routing_reason,
            status="completed",
            badge="Auto-Switched"
        ))
    else:
        active_mode = requested_mode
        auto_routed = False
        routing_reason = f"Mode explicitly requested as {active_mode.upper()}."
        process_steps.append(ProcessStep(
            title=f"Mode set to {'LLM + RAG' if active_mode == 'rag' else 'Direct LLM'}",
            detail=routing_reason,
            status="completed",
            badge="Configured"
        ))

    if active_mode == "direct":
        # Direct LLM execution trace
        process_steps.append(ProcessStep(
            title="Bypassing RAG vector retrieval (Direct LLM Mode)",
            detail="Operating purely on fine-tuned LoRA model weights without external PDF knowledge retrieval.",
            status="completed",
            badge="Direct"
        ))
        process_steps.append(ProcessStep(
            title="Formatting prompt with fine-tuned system instructions",
            detail="Using Qwen2.5 ChatML template with strict profile grounding guardrails.",
            status="completed",
            badge="Prompt"
        ))
        process_steps.append(ProcessStep(
            title="Running causal LM inference with LoRA adapter",
            detail="Sampling response with temperature=0.3, max_new_tokens=220.",
            status="completed",
            badge="Inference"
        ))

        answer = generate_answer(question, context=None, mode="direct")
        latency_ms = round((time.time() - t_start) * 1000, 2)

        return ChatResponse(
            answer=answer,
            mode="direct",
            sources=[],
            process=process_steps,
            latency_ms=latency_ms,
            auto_routed=auto_routed,
            routing_reason=routing_reason
        )

    # RAG Mode execution
    process_steps.append(ProcessStep(
        title="Searching portfolio knowledge base (FAISS index)",
        detail="Computing sentence embeddings with all-MiniLM-L6-v2 and querying Inner Product vector index.",
        status="completed",
        badge="FAISS"
    ))

    try:
        retrieved = retrieve_context(question, top_k=request.top_k or 4)
        top_score = max((x["score"] for x in retrieved), default=0.0)

        process_steps.append(ProcessStep(
            title=f"Retrieved {len(retrieved)} knowledge chunks (Top similarity: {top_score:.4f})",
            detail=f"Extracted candidate passages from prem_rag_knowledge_base.pdf.",
            status="completed",
            badge=f"Score: {top_score:.2f}"
        ))

        threshold = request.threshold if request.threshold is not None else 0.25
        relevant = [x for x in retrieved if x["score"] >= threshold]

        process_steps.append(ProcessStep(
            title=f"Relevance threshold check (similarity >= {threshold})",
            detail=f"{len(relevant)} of {len(retrieved)} chunks passed the confidence threshold.",
            status="completed",
            badge=f"{len(relevant)} Passed"
        ))

        if not relevant:
            process_steps.append(ProcessStep(
                title="Strict grounding check: Insufficient evidence in PDF",
                detail="No chunks satisfied the similarity threshold; preventing hallucination.",
                status="completed",
                badge="Ungrounded"
            ))
            latency_ms = round((time.time() - t_start) * 1000, 2)
            return ChatResponse(
                answer="I couldn't find that information in Prem's portfolio.",
                mode="rag",
                sources=[],
                process=process_steps,
                latency_ms=latency_ms,
                auto_routed=auto_routed,
                routing_reason=routing_reason
            )

        process_steps.append(ProcessStep(
            title="Synthesizing grounded context into prompt",
            detail=f"Injected {len(relevant)} validated reference chunks into Qwen system/user prompts.",
            status="completed",
            badge="Context Injected"
        ))

        context = "\n\n".join(x["text"] for x in relevant)
        answer = generate_answer(question, context=context, mode="rag")

        process_steps.append(ProcessStep(
            title="Generated grounded response with Qwen2.5 + LoRA",
            detail="Output constrained strictly to supported portfolio facts.",
            status="completed",
            badge="Generated"
        ))

        sources = [x["text"][:220] + ("..." if len(x["text"]) > 220 else "") for x in relevant]
        latency_ms = round((time.time() - t_start) * 1000, 2)

        return ChatResponse(
            answer=answer,
            mode="rag",
            sources=sources,
            process=process_steps,
            latency_ms=latency_ms,
            auto_routed=auto_routed,
            routing_reason=routing_reason
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
