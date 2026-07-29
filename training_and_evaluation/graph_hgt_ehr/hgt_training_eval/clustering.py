# backend/pipelines/graph_ehr/clustering.py
from __future__ import annotations

import os
import sys
import json
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import faiss

from config import settings
from pipelines.graph_ehr.hgt_model import get_visit_metadata_by_index, get_visit_note_by_index

_FAISS_PATIENT_INDEX = None
_INDEX_MAP = None
_ALL_EMB = None

def _init_patient_faiss():
    global _FAISS_PATIENT_INDEX, _INDEX_MAP, _ALL_EMB
    if _FAISS_PATIENT_INDEX is None:
        idx_path = str(settings.FAISS_PATIENT_INDEX_PATH)
        map_path = str(settings.FAISS_PATIENT_MAPPING_PATH)
        if os.path.exists(idx_path) and os.path.exists(map_path):
            try:
                buf = np.fromfile(idx_path, dtype=np.uint8)
                _FAISS_PATIENT_INDEX = faiss.deserialize_index(buf)
                with open(map_path, "rb") as f:
                    _INDEX_MAP = pickle.load(f)
                _ALL_EMB = _FAISS_PATIENT_INDEX.reconstruct_n(0, _FAISS_PATIENT_INDEX.ntotal)
                print("[Clustering] FAISS patient index loaded OK.")
            except Exception as e:
                print(f"[Clustering] Warning: Error loading patient FAISS index ({e})", file=sys.stderr)

def fuse_embeddings(query_emb: np.ndarray, label_emb: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    return alpha * query_emb + (1 - alpha) * label_emb

def search_patient_visits_faiss_v2(
    fused_emb: np.ndarray,
    patient_id: Any,
    k: int = 5,
) -> Tuple[List[int], List[float]]:
    """Efficient FAISS patient-specific search without full copy."""
    _init_patient_faiss()
    if _FAISS_PATIENT_INDEX is None or _INDEX_MAP is None or _ALL_EMB is None:
        print(f"[Clustering] Patient FAISS index not available.", file=sys.stderr)
        return [], []

    if isinstance(patient_id, str) and patient_id.isdigit():
        patient_id = int(patient_id)

    indices = [i for i, pid in _INDEX_MAP.items() if pid == patient_id]
    if not indices:
        print(f"[Clustering] No embeddings found for patient_id={patient_id}", file=sys.stderr)
        return [], []

    sub_emb = _ALL_EMB[indices]
    d = sub_emb.shape[1]

    fused_emb = np.asarray(fused_emb).reshape(1, -1)
    if fused_emb.shape[1] != d:
        raise ValueError(
            f"[ERROR] Dimension mismatch: fused_emb shape={fused_emb.shape}, index dim={d}."
        )

    sub_index = faiss.IndexFlatL2(d)
    sub_index.add(sub_emb)
    D, I = sub_index.search(fused_emb, min(k, len(indices)))
    neighbor_global_idx = [indices[i] for i in I[0] if i < len(indices)]
    return neighbor_global_idx, D[0].tolist()

def collect_visit_metadata_and_texts(global_indices: List[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    visit_metadata_list = []
    visit_texts_list = []
    for idx in global_indices:
        meta = get_visit_metadata_by_index(idx)
        note = get_visit_note_by_index(idx)
        visit_metadata_list.append(meta)
        visit_texts_list.append(note)
    return visit_metadata_list, visit_texts_list

def create_personalized_llm_context(
    query: str,
    image_labels: List[str],
    relevant_visits: List[Dict[str, Any]],
    extra_notes: str = ""
) -> str:
    context = (
        f"User Query: {query}\n"
        f"Image Predicted Labels: {', '.join(image_labels)}\n"
        f"{extra_notes}\n"
        "Most Relevant Patient History Visits (with metadata):\n"
    )
    for i, visit in enumerate(relevant_visits, 1):
        context += f"\nVisit {i}: {json.dumps(visit, ensure_ascii=False)}"
    context += "\n\nGenerate a personalized, structured clinical report (assessment, reasoning, diagnosis)."
    return context

def personalized_patient_history_workflow_with_texts(
    query_text: str,
    image_labels: List[str],
    patient_id: Any,
    embeddings_model: Any,
    k: int = 5,
    alpha: float = 0.6,
    extra_notes: str = "",
):
    query_emb = np.asarray(embeddings_model.embed_query(query_text), dtype=np.float32)
    labels_text = " ".join(image_labels)
    label_emb = np.asarray(embeddings_model.embed_query(labels_text), dtype=np.float32)

    fused_emb = fuse_embeddings(query_emb, label_emb, alpha=alpha)

    top_global_indices, dists = search_patient_visits_faiss_v2(
        fused_emb, patient_id, k
    )

    relevant_visits, visit_texts = collect_visit_metadata_and_texts(top_global_indices)
    llm_context = create_personalized_llm_context(
        query_text, image_labels, relevant_visits, extra_notes=extra_notes
    )
    return relevant_visits, llm_context, visit_texts, dists
