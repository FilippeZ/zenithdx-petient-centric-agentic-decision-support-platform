# ai_agent_runner.py
# ------------------
# Backward-compatibility shim for LangGraph agent execution.

from __future__ import annotations

import os
import time
import traceback
from typing import Optional, Dict, Any

os.environ.setdefault("AGENT_IMPORT_ONLY", "1")

try:
    from agentic_core.agent_loop import runnable
except Exception as exc:
    raise RuntimeError(
        f"❌ Failed to import runnable from agentic_core.agent_loop: {exc}\n{traceback.format_exc()}"
    ) from exc

_GRAPH = runnable

def run_agent(
    user_query: str,
    image_path: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the LangGraph agent and returns diagnosis & XAI result artifacts.
    """
    metadata: Dict[str, Any] = {}
    if image_path:
        metadata["image_path"] = image_path
    if patient_id:
        metadata["patient_id"] = patient_id

    agent_state: Dict[str, Any] = {
        "input":              user_query,
        "chat_history":       [],
        "intermediate_steps": [],
        "agent_outcome":      "",
        "plan":               [],
        "metadata":           metadata,
        "self_refine_iter":   0,
        "context_cache":      {},
    }

    t0 = time.time()
    output = _GRAPH.invoke(agent_state, config={"recursion_limit": 50})
    elapsed = round(time.time() - t0, 2)

    agent_out = output.get("agent_outcome", {}) or {}
    captum_keys = [k for k in agent_out.keys() if k.startswith("captum_")]
    captum_images = {k: agent_out[k] for k in captum_keys}

    return {
        "diagnosis":              agent_out.get("diagnosis") or agent_out.get("report") or "",
        "xai_report":             agent_out.get("xai_report") or agent_out.get("explain_text") or "",
        "classification_results": agent_out.get("classification_results", []),
        "classification_labels":  agent_out.get("classification_labels", []),
        "original_xray":          agent_out.get("original_xray"),
        "gradcam_overlay":        agent_out.get("gradcam_overlay"),
        "captum_image":           agent_out.get("captum_image"),
        **captum_images,
        "top_words":              agent_out.get("top_words", {}),
        "elapsed_sec":            elapsed,
        "agent_outcome":          agent_out,
    }
