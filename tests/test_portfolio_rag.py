import os
import sys
import unittest
import json

# Add backend directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rag import extract_pdf_text, split_into_chunks, load_index, retrieve_context, PDF_PATH
from llm import generate_answer, ADAPTER_PATH
from fastapi.testclient import TestClient
from app import app


class TestPortfolioRAGSystem(unittest.TestCase):
    """
    Test suite verifying all items in Section 10 of student_portfolio_rag_llm_integration.md.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Checklist 1: /health returns status ok"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("author", data)

    def test_02_pdf_text_extraction(self):
        """Checklist 2: PDF text extraction works and contains Prem Kumar's info"""
        self.assertTrue(os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}")
        text = extract_pdf_text(PDF_PATH)
        self.assertTrue(len(text) > 200, "Extracted text is too short")
        self.assertIn("Prem Kumar", text)
        self.assertIn("Projects", text)
        self.assertIn("Skills", text)

    def test_03_chunking_logic(self):
        """Checklist 3: Chunking splits text with word overlap"""
        sample_text = "word " * 500
        chunks = split_into_chunks(sample_text, chunk_size=100, overlap=20)
        self.assertTrue(len(chunks) > 3)
        self.assertTrue(all(len(c.split()) <= 100 for c in chunks))

    def test_04_faiss_index_loads(self):
        """Checklist 4: FAISS index is created and loads"""
        index, chunks = load_index()
        self.assertIsNotNone(index)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(index.ntotal, len(chunks))

    def test_05_known_project_retrieval(self):
        """Checklist 5: A known project question retrieves the correct chunk"""
        results = retrieve_context("What projects has Prem developed?", top_k=4)
        self.assertTrue(len(results) > 0)
        # Check that project-related chunks are retrieved
        all_text = " ".join(r["text"] for r in results)
        self.assertTrue(
            any(name in all_text for name in ["Deepfake", "kLLM", "SignBridge", "Projects"]),
            "Retrieved chunks should mention Prem's projects"
        )
        # Top score should exceed confidence threshold
        self.assertGreater(results[0]["score"], 0.25)

    def test_06_direct_llm_mode(self):
        """Checklist 6: Direct LLM mode returns stable identity without RAG retrieval"""
        response = self.client.post("/chat", json={
            "question": "Who are you?",
            "mode": "direct"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "direct")
        self.assertEqual(data["sources"], [])
        self.assertIn("Prem Kumar", data["answer"])
        self.assertIn("K S Rangasamy College of Technology", data["answer"])
        self.assertTrue(len(data["process"]) > 0)

    def test_07_rag_mode_grounded_answer(self):
        """Checklist 7: /chat returns an answer grounded in retrieved context"""
        response = self.client.post("/chat", json={
            "question": "What projects has Prem developed?",
            "mode": "rag"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "rag")
        self.assertTrue(len(data["sources"]) > 0)
        self.assertTrue(len(data["process"]) >= 4)
        self.assertTrue(
            any(p in data["answer"] for p in ["Deepfake", "kLLM", "SignBridge", "Voice Assistant", "YouTube Clone"]),
            "Answer should mention documented projects from the PDF"
        )

    def test_08_unknown_cgpa_not_invented(self):
        """Checklist 8: Unknown questions (CGPA) are not answered with invented facts"""
        response = self.client.post("/chat", json={
            "question": "What is Prem's CGPA?",
            "mode": "rag"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should state it's not available, not invent a number
        self.assertTrue(
            "couldn't find" in data["answer"].lower() or "not" in data["answer"].lower(),
            f"Expected refusal or not available, got: {data['answer']}"
        )

    def test_09_unknown_internship_not_invented(self):
        """Checklist 9: Unknown internship question does not invent an employer"""
        response = self.client.post("/chat", json={
            "question": "Which company did Prem intern at?",
            "mode": "rag"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(
            "not" in data["answer"].lower() or "couldn't" in data["answer"].lower(),
            f"Expected refusal for unconfirmed internship, got: {data['answer']}"
        )

    def test_10_process_trace_structure(self):
        """Checklist 10: Step-by-step process trace contains required animation and honing steps"""
        response = self.client.post("/chat", json={
            "question": "Which programming languages does Prem know?",
            "mode": "rag"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        process = data.get("process", [])
        self.assertTrue(len(process) >= 4)
        titles = [step["title"] for step in process]
        # Should have search, retrieval, threshold, and generation steps
        self.assertTrue(any("FAISS" in t or "Searching" in t for t in titles))
        self.assertTrue(any("Retrieved" in t for t in titles))
        self.assertTrue(any("threshold" in t.lower() for t in titles))
        self.assertTrue(any("response" in t.lower() or "synthesizing" in t.lower() or "generated" in t.lower() for t in titles))

    def test_11_auto_switching_router(self):
        """Checklist 11: Auto-switch between Direct LLM and LLM+RAG based on input without manual setting"""
        # Test stable identity -> Auto-switches to Direct LLM
        res_direct = self.client.post("/chat", json={"question": "Who are you?"})
        self.assertEqual(res_direct.status_code, 200)
        d_data = res_direct.json()
        self.assertEqual(d_data["mode"], "direct")
        self.assertTrue(d_data["auto_routed"])
        self.assertIn("Prem Kumar", d_data["answer"])

        # Test dynamic projects query -> Auto-switches to LLM + RAG
        res_rag = self.client.post("/chat", json={"question": "What projects has Prem developed?"})
        self.assertEqual(res_rag.status_code, 200)
        r_data = res_rag.json()
        self.assertEqual(r_data["mode"], "rag")
        self.assertTrue(r_data["auto_routed"])
        self.assertTrue(len(r_data["sources"]) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

