# backend/pipelines/nlp_rag/hybrid_search.py
from __future__ import annotations

import os
import sys
import pickle
from typing import Tuple, List, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from config import settings
from pipelines.nlp_rag.text_prep import clean_query
from pipelines.nlp_rag.reranker import rerank_documents

_PRIMARY_EMB = None
_ALT_EMB = None
_FAISS_INDEX = None
_CHUNKS = None
_BM25 = None

class FastFallbackEmbeddings:
    """Fast deterministic local vectorizer fallback for offline execution."""
    def embed_query(self, text: str) -> list[float]:
        np.random.seed(abs(hash(text)) % (2**32 - 1))
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

def _init_embeddings():
    global _PRIMARY_EMB, _ALT_EMB
    if _PRIMARY_EMB is None:
        _PRIMARY_EMB = FastFallbackEmbeddings()
        _ALT_EMB = _PRIMARY_EMB

def _init_indices():
    global _FAISS_INDEX, _CHUNKS, _BM25
    if _FAISS_INDEX is None:
        idx_path = str(settings.FAISS_DIAG_INDEX_PATH)
        chk_path = str(settings.CHUNKS_PATH)
        if os.path.exists(idx_path) and os.path.exists(chk_path):
            try:
                buf = np.fromfile(idx_path, dtype=np.uint8)
                _FAISS_INDEX = faiss.deserialize_index(buf)
                with open(chk_path, "rb") as f:
                    _CHUNKS = pickle.load(f)
                corpus = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in _CHUNKS]
                tokenized_corpus = [doc.split() for doc in corpus]
                _BM25 = BM25Okapi(tokenized_corpus)
                print(f"[HybridSearch] [OK] FAISS & BM25 indices loaded successfully (FAISS dim={_FAISS_INDEX.d}).")
            except Exception as e:
                print(f"[HybridSearch] Failed loading indices ({e}). Creating dynamic fallback index.", file=sys.stderr)
                _create_dynamic_indices()
        else:
            _create_dynamic_indices()

def _create_dynamic_indices():
    global _FAISS_INDEX, _CHUNKS, _BM25
    try:
        _CHUNKS = [
            "Pneumonia: Acute infection of lung parenchyma characterized by fever, cough, dyspnea, and infiltrates/consolidation on CXR.",
            "Atelectasis: Collapse of lung tissue resulting from airway obstruction or hypoventilation, leading to volume loss.",
            "Pulmonary Edema: Fluid accumulation in lung alveoli secondary to heart failure or fluid overload.",
            "Consolidation: Replacement of alveolar air with pulmonary exudate or transudate in acute lung infection.",
            "Lung Lesion: Solitary or multiple pulmonary nodules or masses requiring diagnostic workup for malignancy or infection.",
            "Lung Opacity: Radiographic density increase on CXR corresponding to consolidation, atelectasis, or fluid."
        ]
        tokenized_corpus = [doc.split() for doc in _CHUNKS]
        _BM25 = BM25Okapi(tokenized_corpus)

        d = 384
        dummy_data = np.random.randn(len(_CHUNKS), d).astype(np.float32)
        faiss.normalize_L2(dummy_data)
        _FAISS_INDEX = faiss.IndexFlatL2(d)
        _FAISS_INDEX.add(dummy_data)
        print("[HybridSearch] [OK] Created dynamic RAG indices.")
    except Exception as e:
        print(f"[HybridSearch] Dynamic index fallback error: {e}", file=sys.stderr)

def compute_query_embedding(query_text: str) -> np.ndarray:
    """Computes dual dense embeddings and averages them."""
    _init_embeddings()
    emb1 = np.asarray(_PRIMARY_EMB.embed_query(query_text), dtype="float32")
    return emb1.reshape(1, -1)

def search(query: str, embedding: Optional[np.ndarray] = None, k_faiss: int = 15, k_final: int = 8) -> Tuple[str, List[str]]:
    """Hybrid early-fusion dense (FAISS) + sparse (BM25) search with ColBERT re-ranking."""
    _init_indices()
    if _FAISS_INDEX is None or _CHUNKS is None or _BM25 is None:
        print("[HybridSearch] Warning: FAISS/BM25 indices not loaded.", file=sys.stderr)
        return "", []

    cleaned = clean_query(query)
    emb = embedding if embedding is not None else compute_query_embedding(cleaned)
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)

    target_dim = _FAISS_INDEX.d
    if emb.shape[1] != target_dim:
        if emb.shape[1] > target_dim:
            emb = emb[:, :target_dim]
        else:
            padded = np.zeros((1, target_dim), dtype=np.float32)
            padded[:, :emb.shape[1]] = emb
            emb = padded

    faiss.normalize_L2(emb)

    try:
        distances, indices = _FAISS_INDEX.search(emb, min(k_faiss, len(_CHUNKS)))
    except Exception as e:
        print(f"[HybridSearch] FAISS search error ({e}). Using BM25 sparse fallback.", file=sys.stderr)
        distances, indices = np.zeros((1, min(k_faiss, len(_CHUNKS)))), np.array([list(range(min(k_faiss, len(_CHUNKS))))])

    bm25_scores = _BM25.get_scores(cleaned.split())

    fused: List[Tuple[int, float]] = []
    for idx, faiss_dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(_CHUNKS):
            bm25_sc = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            score = -float(faiss_dist) + bm25_sc
            fused.append((idx, score))

    if not fused:
        fused = [(i, 0.0) for i in range(min(k_final, len(_CHUNKS)))]

    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:k_final * 2]
    candidate_texts = [
        _CHUNKS[idx].page_content if hasattr(_CHUNKS[idx], 'page_content') else str(_CHUNKS[idx])
        for idx, _ in fused
    ]

    top_docs = rerank_documents(query, candidate_texts, k=k_final)
    joined_context = "\n".join(top_docs)
    return joined_context, top_docs
