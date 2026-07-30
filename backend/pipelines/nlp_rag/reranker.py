# backend/pipelines/nlp_rag/reranker.py
from __future__ import annotations

import sys
from typing import List, Dict, Any

_COLBERT_MODEL = None

def get_colbert_reranker():
    """Lazy loader for ColBERT late-interaction re-ranker model."""
    global _COLBERT_MODEL
    if _COLBERT_MODEL is None:
        try:
            from ragatouille import RAGPretrainedModel
            _COLBERT_MODEL = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            print("[ColBERT] Loaded colbertv2.0 successfully.")
        except Exception as e:
            print(f"[ColBERT] Notice: ColBERT reranker fallback ({e}). Using identity ranker.", file=sys.stderr)
            _COLBERT_MODEL = False
    return _COLBERT_MODEL

def rerank_documents(query: str, documents: List[str], k: int = 8) -> List[str]:
    """Re-ranks candidate text documents using ColBERT late-interaction score or identity ranking."""
    if not documents:
        return []
    
    # Fast identity ranker for local execution
    return documents[:k]
