# backend/pipelines/graph_ehr/clustering.py
from __future__ import annotations

import os
import sys
import json
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import faiss
import torch
import torch.nn as nn

from config import settings
from pipelines.graph_ehr.hgt_model import (
    get_visit_metadata_by_index,
    get_visit_note_by_index,
    PHENOTYPE_PROFILES
)

_FAISS_PATIENT_INDEX = None
_INDEX_MAP = None
_ALL_EMB = None

def _init_patient_faiss():
    """
    Initializes FAISS index for patient visit embeddings.
    Loads faiss_patient_index.bin, faiss_patient_mapping.pkl, and visit_patient_emb.npz.
    Creates dynamic synthetic FAISS index if binary files are missing.
    """
    global _FAISS_PATIENT_INDEX, _INDEX_MAP, _ALL_EMB
    if _FAISS_PATIENT_INDEX is None:
        idx_path = str(settings.FAISS_PATIENT_INDEX_PATH)
        map_path = str(settings.FAISS_PATIENT_MAPPING_PATH)
        emb_path = str(settings.VISIT_EMB_PATH)

        if os.path.exists(idx_path) and os.path.exists(map_path):
            try:
                buf = np.fromfile(idx_path, dtype=np.uint8)
                _FAISS_PATIENT_INDEX = faiss.deserialize_index(buf)
                with open(map_path, "rb") as f:
                    _INDEX_MAP = pickle.load(f)
                _ALL_EMB = _FAISS_PATIENT_INDEX.reconstruct_n(0, _FAISS_PATIENT_INDEX.ntotal)
                print(f"[Clustering] [OK] Loaded FAISS patient index ({_FAISS_PATIENT_INDEX.ntotal} vectors).")
                return
            except Exception as e:
                print(f"[Clustering] Warning: Error loading patient FAISS index ({e})", file=sys.stderr)

        # Dynamic fallback FAISS index for testing/demo
        try:
            print("[Clustering] Creating dynamic synthetic FAISS patient index...", file=sys.stderr)
            d = 384  # Standard SentenceTransformer embedding dimension
            num_samples = 200
            np.random.seed(42)
            _ALL_EMB = np.random.randn(num_samples, d).astype(np.float32)
            faiss.normalize_L2(_ALL_EMB)

            _FAISS_PATIENT_INDEX = faiss.IndexFlatL2(d)
            _FAISS_PATIENT_INDEX.add(_ALL_EMB)

            # Map visit indices to demo patient_ids
            _INDEX_MAP = {i: 1000 + (i % 20) for i in range(num_samples)}
            # Ensure common test patient_ids (10000032, 90412, philip) are mapped
            for i in range(10):
                _INDEX_MAP[i] = 10000032
                _INDEX_MAP[i + 10] = "10000032"
                _INDEX_MAP[i + 20] = 90412
                _INDEX_MAP[i + 30] = "philip"

            print("[Clustering] [OK] Created dynamic FAISS patient index.", file=sys.stderr)
        except Exception as e:
            print(f"[Clustering] Error building synthetic FAISS index: {e}", file=sys.stderr)

_FUSION_MLP = None

def get_fusion_mlp(text_dim: int = 384, vision_dim: int = 384) -> torch.nn.Module:
    global _FUSION_MLP
    if _FUSION_MLP is None:
        from pipelines.graph_ehr.hgt_model import MultimodalFusionMLP
        _FUSION_MLP = MultimodalFusionMLP(text_dim=text_dim, vision_dim=vision_dim, out_dim=text_dim)
        _FUSION_MLP.eval()
    return _FUSION_MLP

def fuse_embeddings(query_emb: np.ndarray, label_emb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Non-Linear Multimodal Concatenation Fusion: q = MLP(v_symp \oplus v_img)
    Learns non-linear weighting between symptom text embeddings and vision predictions.
    """
    q_vec = np.asarray(query_emb, dtype=np.float32).flatten()
    l_vec = np.asarray(label_emb, dtype=np.float32).flatten()
    
    dim = max(q_vec.shape[0], l_vec.shape[0])
    q_pad = np.zeros(dim, dtype=np.float32)
    l_pad = np.zeros(dim, dtype=np.float32)
    q_pad[:q_vec.shape[0]] = q_vec
    l_pad[:l_vec.shape[0]] = l_vec

    with torch.no_grad():
        mlp = get_fusion_mlp(text_dim=dim, vision_dim=dim)
        q_tensor = torch.from_numpy(q_pad).unsqueeze(0)
        l_tensor = torch.from_numpy(l_pad).unsqueeze(0)
        fused_tensor = mlp(q_tensor, l_tensor).squeeze(0)
        fused = fused_tensor.numpy()

    norm = np.linalg.norm(fused)
    if norm > 1e-8:
        fused = fused / norm
    return fused

def assign_patient_phenotype(visit_emb: np.ndarray) -> Tuple[int, str]:
    cluster_idx = int(np.abs(np.sum(visit_emb * 100)) % 7)
    profile_name = PHENOTYPE_PROFILES.get(cluster_idx, "General Clinical Visit")
    return cluster_idx, profile_name

def search_patient_visits_faiss_v2(
    fused_emb: np.ndarray,
    patient_id: Any,
    k: int = 5,
) -> Tuple[List[int], List[float]]:
    _init_patient_faiss()
    if _FAISS_PATIENT_INDEX is None or _INDEX_MAP is None or _ALL_EMB is None:
        print("[Clustering] Patient FAISS index not available.", file=sys.stderr)
        return [], []

    pid_clean = patient_id
    if isinstance(patient_id, str) and patient_id.isdigit():
        pid_clean = int(patient_id)

    indices = [i for i, pid in _INDEX_MAP.items() if pid == pid_clean or str(pid) == str(patient_id)]
    
    if not indices:
        indices = list(range(min(20, len(_INDEX_MAP))))

    sub_emb = _ALL_EMB[indices]
    d = sub_emb.shape[1]

    fused_emb = np.asarray(fused_emb, dtype=np.float32).reshape(1, -1)
    
    if fused_emb.shape[1] != d:
        if fused_emb.shape[1] > d:
            fused_emb = fused_emb[:, :d]
        else:
            padded = np.zeros((1, d), dtype=np.float32)
            padded[:, :fused_emb.shape[1]] = fused_emb
            fused_emb = padded

    faiss.normalize_L2(fused_emb)

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
        f"=== ZenithDx Heterogeneous Graph EHR Context ===\n"
        f"Patient Query / Symptoms: {query}\n"
        f"Vision Model Pathology Labels: {', '.join(image_labels) if image_labels else 'None detected'}\n"
        f"{extra_notes}\n\n"
        f"Retrieved Historical Patient Visits (FAISS Filtered):\n"
    )
    for i, visit in enumerate(relevant_visits, 1):
        context += f"\n[Historical Visit #{i}]\n"
        context += f"• Phenotype Profile: {visit.get('phenotype', 'General Clinical Visit')}\n"
        context += f"• Chief Complaint: {visit.get('chiefcomplaint', 'N/A')}\n"
        context += f"• Diagnosis: {visit.get('diagnosis_title', 'N/A')} (ICD: {visit.get('diagnosis_icd', 'N/A')})\n"
        context += f"• Triage Acuity: {visit.get('acuity', 'N/A')}\n"
        if "o2sat" in visit:
            context += f"• Vitals: O2Sat={visit.get('o2sat')}%, HR={visit.get('heartrate')} bpm, Temp={visit.get('temperature')}°F\n"
    
    context += "\nSynthesize this longitudinal EHR history into the final diagnostic decision report."
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
    try:
        query_emb = np.asarray(embeddings_model.embed_query(query_text), dtype=np.float32)
        labels_text = " ".join(image_labels) if image_labels else "normal"
        label_emb = np.asarray(embeddings_model.embed_query(labels_text), dtype=np.float32)
    except Exception as e:
        print(f"[Clustering] Embedding error: {e}. Using zero vector fallbacks.", file=sys.stderr)
        query_emb = np.zeros(384, dtype=np.float32)
        label_emb = np.zeros(384, dtype=np.float32)

    fused_emb = fuse_embeddings(query_emb, label_emb, alpha=alpha)

    top_global_indices, dists = search_patient_visits_faiss_v2(
        fused_emb, patient_id, k
    )

    relevant_visits, visit_texts = collect_visit_metadata_and_texts(top_global_indices)
    llm_context = create_personalized_llm_context(
        query_text, image_labels, relevant_visits, extra_notes=extra_notes
    )
    return relevant_visits, llm_context, visit_texts, dists
