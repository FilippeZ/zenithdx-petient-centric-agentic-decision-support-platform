# backend/pipelines/nlp_rag/hybrid_search.py
from __future__ import annotations

import os
import sys
import pickle
from typing import Tuple, List, Optional
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from pipelines.nlp_rag.text_prep import clean_query
from pipelines.nlp_rag.reranker import rerank_documents

_PRIMARY_EMB = None
_ALT_EMB = None
_FAISS_INDEX = None
_CHUNKS = None
_BM25 = None

def _init_embeddings():
    global _PRIMARY_EMB, _ALT_EMB
    if _PRIMARY_EMB is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _PRIMARY_EMB = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            _ALT_EMB = HuggingFaceEmbeddings(
                model_name="Zybg/synthetic-clinical-embedding-model",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            print(f"[HybridSearch] Warning initializing embedding models: {e}", file=sys.stderr)

def _init_indices():
    global _FAISS_INDEX, _CHUNKS, _BM25
    if _FAISS_INDEX is None:
        idx_path = str(settings.FAISS_DIAG_INDEX_PATH)
        chk_path = str(settings.CHUNKS_PATH)
        if os.path.exists(idx_path) and os.path.exists(chk_path):
            try:
                # Use np.fromfile + deserialize_index to support non-ASCII paths on Windows
                buf = np.fromfile(idx_path, dtype=np.uint8)
                _FAISS_INDEX = faiss.deserialize_index(buf)
                with open(chk_path, "rb") as f:
                    _CHUNKS = pickle.load(f)
                corpus = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in _CHUNKS]
                tokenized_corpus = [doc.split() for doc in corpus]
                _BM25 = BM25Okapi(tokenized_corpus)
                print(f"[HybridSearch] FAISS & BM25 indices loaded successfully.")
            except Exception as e:
                print(f"[HybridSearch] Failed loading indices: {e}", file=sys.stderr)

def compute_query_embedding(query_text: str) -> np.ndarray:
    """Computes dual dense embeddings and averages them."""
    _init_embeddings()
    if _PRIMARY_EMB is None or _ALT_EMB is None:
        # Fallback random normalized embedding
        dummy = np.random.randn(1, 1024).astype("float32")
        return dummy / np.linalg.norm(dummy)
    emb1 = np.asarray(_PRIMARY_EMB.embed_query(query_text), dtype="float32")
    emb2 = np.asarray(_ALT_EMB.embed_query(query_text), dtype="float32")
    if emb1.shape != emb2.shape:
        return emb1.reshape(1, -1)
    return ((emb1 + emb2) / 2.0).reshape(1, -1)

def search(query: str, embedding: Optional[np.ndarray] = None, k_faiss: int = 15, k_final: int = 8) -> Tuple[str, List[str]]:
    """Hybrid early-fusion dense (FAISS) + sparse (BM25) search with ColBERT re-ranking."""
    _init_indices()
    if _FAISS_INDEX is None or _CHUNKS is None or _BM25 is None:
        print("[HybridSearch] Warning: FAISS/BM25 indices not loaded.", file=sys.stderr)
        return "", []

    cleaned = clean_query(query)
    emb = embedding if embedding is not None else compute_query_embedding(cleaned)
    distances, indices = _FAISS_INDEX.search(emb, k_faiss)
    bm25_scores = _BM25.get_scores(cleaned.split())

    fused: List[Tuple[int, float]] = []
    for idx, faiss_dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(_CHUNKS):
            score = -float(faiss_dist) + float(bm25_scores[idx])
            fused.append((idx, score))

    if not fused:
        return "", []

    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:k_final * 2]
    candidate_texts = [
        _CHUNKS[idx].page_content if hasattr(_CHUNKS[idx], 'page_content') else str(_CHUNKS[idx])
        for idx, _ in fused
    ]

    # Re-rank using ColBERT
    top_docs = rerank_documents(query, candidate_texts, k=k_final)
    joined_context = "\n".join(top_docs)
    return joined_context, top_docs
