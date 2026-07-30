# backend/agentic_core/tools/ehr_tool.py
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple
from pipelines.graph_ehr.clustering import personalized_patient_history_workflow_with_texts
import pipelines.nlp_rag.hybrid_search as hs

def run_ehr_analysis(
    query_text: str,
    image_labels: List[str],
    patient_id: Any,
    k: int = 5,
    alpha: float = 0.6,
    extra_notes: str = ""
) -> Tuple[List[Dict[str, Any]], str, List[str], List[float]]:
    """Runs EHR graph search and patient history workflow."""
    try:
        hs._init_embeddings()
        emb_model = hs._PRIMARY_EMB
    except Exception as e:
        print(f"[EHRTool] Warning initializing embeddings ({e}). Using fallback.", file=sys.stderr)
        emb_model = hs.FastFallbackEmbeddings()

    if emb_model is None:
        emb_model = hs.FastFallbackEmbeddings()

    return personalized_patient_history_workflow_with_texts(
        query_text=query_text,
        image_labels=image_labels,
        patient_id=patient_id,
        embeddings_model=emb_model,
        k=k,
        alpha=alpha,
        extra_notes=extra_notes
    )
