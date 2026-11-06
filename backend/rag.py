import json
import os
import glob
import warnings

# Avoid hanging on DNS lookups if offline
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
import fitz
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
RAG_PDF_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "rag pdf"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(DATA_DIR, "prem_updated_portfolio_rag_knowledge_base.pdf"))
if not os.path.exists(PDF_PATH):
    PDF_PATH = os.path.join(DATA_DIR, "prem_rag_knowledge_base.pdf")

STORE_DIR = os.getenv("STORE_DIR", os.path.join(BASE_DIR, "vector_store"))
INDEX_PATH = os.path.join(STORE_DIR, "portfolio.faiss")
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.json")

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def get_all_pdf_paths():
    """Find all unique PDF knowledge base files from data/ and 'rag pdf/' folders."""
    seen_names = set()
    unique_paths = []
    
    # Priority: backend/data then rag pdf
    for search_dir in [DATA_DIR, RAG_PDF_DIR]:
        if os.path.exists(search_dir):
            for pdf_file in glob.glob(os.path.join(search_dir, "*.pdf")):
                bname = os.path.basename(pdf_file)
                if bname not in seen_names:
                    seen_names.add(bname)
                    unique_paths.append(os.path.abspath(pdf_file))
    
    if not unique_paths and os.path.exists(PDF_PATH):
        unique_paths.append(os.path.abspath(PDF_PATH))
    return sorted(unique_paths)


def extract_pdf_text(path=None):
    if path is None:
        path = PDF_PATH
    with fitz.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def split_into_chunks(text, chunk_size=180, overlap=30):
    # Split by words; overlap preserves context across chunk boundaries.
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap

    return chunks


def build_index():
    global _cached_index, _cached_chunks
    os.makedirs(STORE_DIR, exist_ok=True)
    
    pdf_files = get_all_pdf_paths()
    print(f"Reading {len(pdf_files)} PDF source(s):")
    for p in pdf_files:
        print(f"  - {os.path.basename(p)}")

    all_chunks = []
    for pdf_file in pdf_files:
        try:
            text = extract_pdf_text(pdf_file)
            file_chunks = split_into_chunks(text)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Warning: Failed to extract text from {pdf_file}: {e}")

    # Deduplicate identical chunks if any
    unique_chunks = []
    seen = set()
    for c in all_chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)

    if not unique_chunks:
        raise ValueError("No text was extracted from the portfolio PDFs.")

    model = get_embedding_model()
    vectors = model.encode(
        unique_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Inner product on normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, ensure_ascii=False, indent=2)

    _cached_index = index
    _cached_chunks = unique_chunks

    print(f"Indexed {len(unique_chunks)} chunks from {len(pdf_files)} PDF(s).")
    return len(unique_chunks)


_cached_index = None
_cached_chunks = None


def load_index():
    global _cached_index, _cached_chunks
    if _cached_index is not None and _cached_chunks is not None:
        return _cached_index, _cached_chunks

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        build_index()

    _cached_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _cached_chunks = json.load(f)

    return _cached_index, _cached_chunks


def retrieve_context(question, top_k=4):
    index, chunks = load_index()
    model = get_embedding_model()
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_vector, min(top_k, len(chunks))
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": chunks[int(idx)],
                "score": float(score)
            })

    return results


if __name__ == "__main__":
    build_index()
