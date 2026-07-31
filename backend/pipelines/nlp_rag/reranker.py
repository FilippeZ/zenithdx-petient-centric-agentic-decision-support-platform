# backend/pipelines/nlp_rag/reranker.py
from __future__ import annotations

import re
import sys
from typing import List, Tuple, Optional, Any

_COLBERT_MODEL = None
SIMILARITY_THRESHOLD = 0.25  # Clinical relevance threshold to prevent RAG Bleed

MEDICAL_CONCEPT_EXPANSION = {
    "pneumonia": ["fever", "cough", "dyspnea", "opacity", "infiltrate", "consolidation", "sputum", "breath", "lung", "respiratory"],
    "atelectasis": ["collapse", "opacity", "volume loss", "hypoventilation", "lung"],
    "edema": ["fluid", "heart failure", "swelling", "orthopnea", "congestion", "pulmonary"],
    "consolidation": ["exudate", "opacity", "dense", "infection", "lung"],
    "bronchitis": ["cough", "wheezing", "airway", "bronchospasm", "sputum"],
    "headache": ["migraine", "cephalea", "photophobia", "dizziness", "tension"],
}

def get_colbert_reranker():
    """Lazy loader for ColBERT late-interaction re-ranker model."""
    global _COLBERT_MODEL
    if _COLBERT_MODEL is None:
        try:
            from ragatouille import RAGPretrainedModel
            _COLBERT_MODEL = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            print("[ColBERT] [OK] Loaded colbertv2.0 model successfully.")
        except Exception as e:
            print(f"[ColBERT] Notice: ColBERT reranker offline fallback ({e}). Using semantic similarity ranker.", file=sys.stderr)
            _COLBERT_MODEL = False
    return _COLBERT_MODEL

def compute_similarity_score(query: str, doc: Any) -> float:
    """
    Computes query term coverage and Jaccard-Cosine semantic similarity score
    with medical concept expansion.
    Returns normalized score in range [0.0, 1.0].
    """
    if hasattr(doc, 'page_content'):
        doc_str = str(doc.page_content)
    elif isinstance(doc, dict) and 'page_content' in doc:
        doc_str = str(doc['page_content'])
    else:
        doc_str = str(doc)

    # Exclude procedural device insertion documents (e.g. G-tubes, J-tubes) for pure symptom queries
    device_terms = {"gastrostomy", "jejunostomy", "g-tube", "j-tube", "catheter placement"}
    q_lower = query.lower()
    is_device_query = any(t in q_lower for t in device_terms)
    if not is_device_query:
        doc_lower = doc_str.lower()
        if any(t in doc_lower for t in device_terms):
            return 0.0

    stopwords = {
        "a", "an", "the", "in", "of", "and", "or", "to", "for", "with", "on", "at",
        "by", "from", "is", "was", "were", "patient", "presents", "right", "left", "stay",
        "combined", "clinical", "representation"
    }

    q_words = {w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stopwords and len(w) > 2}
    d_words = {w.lower() for w in re.findall(r'\w+', doc_str) if w.lower() not in stopwords and len(w) > 2}

    if not q_words or not d_words:
        return 0.0

    # Expand medical concept synonyms in d_words
    expanded_d_words = set(d_words)
    for word in d_words:
        if word in MEDICAL_CONCEPT_EXPANSION:
            expanded_d_words.update(MEDICAL_CONCEPT_EXPANSION[word])

    intersection = q_words.intersection(expanded_d_words)
    coverage = len(intersection) / float(len(q_words))
    jaccard = len(intersection) / float(len(q_words.union(expanded_d_words)))

    if len(intersection) > 0:
        score = max(0.35, 0.7 * coverage + 0.3 * jaccard)
    else:
        score = 0.0
    return float(score)

def rerank_documents(
    query: str,
    documents: List[Any],
    k: int = 8,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Re-ranks candidate text documents using ColBERT MaxSim / semantic similarity scores
    and enforces Defensive Gating (similarity thresholding) to eliminate RAG Bleed.
    """
    if not documents:
        return []

    # Clean document text list
    clean_texts = [
        d.page_content if hasattr(d, 'page_content') else (d['page_content'] if isinstance(d, dict) and 'page_content' in d else str(d))
        for d in documents
    ]

    model = get_colbert_reranker()
    scored_results: List[Tuple[str, float]] = []

    if model:
        try:
            results = model.rerank(query=query, documents=clean_texts, k=k)
            for item in results:
                doc_text = item.get("content", "")
                raw_score = float(item.get("score", 0.0))
                # Normalize raw ColBERT score (typically 0-30) to range [0, 1]
                score = min(1.0, max(0.0, raw_score / 30.0))
                scored_results.append((doc_text, score))
        except Exception as e:
            print(f"[ColBERT] Reranking exception ({e}). Falling back to similarity scoring.", file=sys.stderr)
            for doc_text in clean_texts:
                score = compute_similarity_score(query, doc_text)
                scored_results.append((doc_text, score))
    else:
        for doc_text in clean_texts:
            score = compute_similarity_score(query, doc_text)
            scored_results.append((doc_text, score))

    # Defensive Gating Filter: Filter out documents below clinical relevance threshold
    filtered_docs = [
        doc for doc, score in scored_results
        if score >= similarity_threshold
    ]

    print(f"[ColBERT Defensive Gating] Query: '{query[:40]}...' | Input Docs: {len(documents)} | Passed Threshold (>= {similarity_threshold}): {len(filtered_docs)}")
    return filtered_docs[:k]
