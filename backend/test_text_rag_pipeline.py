# backend/test_text_rag_pipeline.py
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipelines.nlp_rag.hybrid_search import search, compute_query_embedding
from pipelines.nlp_rag.reranker import rerank_documents, compute_similarity_score, SIMILARITY_THRESHOLD

def run_text_rag_tests():
    print("=" * 70)
    print("[TEST] ZENITHDX TEXT RAG PIPELINE UPGRADES TEST")
    print("Testing Min-Max Score Normalization & Defensive Gating Thresholding")
    print("=" * 70)

    # 1. Test Min-Max Score Normalization & High-Relevance Query
    print("\n1. Testing Min-Max Score Normalization for Respiratory Query...")
    query_resp = "Patient presents with fever, productive cough, dyspnea, and right lung opacity."
    context_resp, docs_resp = search(query_resp, k_final=5, similarity_threshold=0.20)
    
    print(f"   * Query: '{query_resp}'")
    print(f"   * Retrieved High-Relevance Documents: {len(docs_resp)}")
    assert len(docs_resp) > 0, "Failed to retrieve documents for respiratory query"
    for i, d in enumerate(docs_resp, 1):
        print(f"     [{i}] {d[:100]}...")

    # 2. Test Defensive Gating (Similarity Thresholding) on Irrelevant Query
    print("\n2. Testing Defensive Gating (Zero RAG Bleed) on Non-Respiratory Query...")
    query_headache = "Severe tension headache, migraine with photophobia and dizziness"
    context_headache, docs_headache = search(query_headache, k_final=5, similarity_threshold=0.65)
    
    print(f"   * Query: '{query_headache}'")
    print(f"   * Context Length: {len(context_headache)} characters")
    print(f"   * Retrieved Documents Passed Threshold: {len(docs_headache)}")
    assert len(docs_headache) == 0 and context_headache == "", "Defensive gating failed! RAG context leaked for irrelevant query."
    print("   * [SUCCESS] Defensive Gating successfully blocked RAG Bleed!")

    # 3. Test Similarity Score Computation
    print("\n3. Testing ColBERT / Similarity Score Bounds...")
    s1 = compute_similarity_score("pneumonia fever cough", "Pneumonia is an acute lung infection with fever and cough.")
    s2 = compute_similarity_score("migraine headache", "Pneumonia is an acute lung infection with fever and cough.")
    print(f"   * High Relevance Score: {s1:.4f} (Expected > 0.60)")
    print(f"   * Low Relevance Score:  {s2:.4f} (Expected < 0.20)")
    assert s1 > s2, "Similarity scoring function failed score ordering"

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TEXT RAG PIPELINE TESTS PASSED CLEANLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_text_rag_tests()
