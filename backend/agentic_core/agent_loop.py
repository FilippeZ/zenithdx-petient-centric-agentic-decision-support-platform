# backend/agentic_core/agent_loop.py
from __future__ import annotations

import os
import sys
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from langgraph.graph import StateGraph, END

from config import settings
from agentic_core.graph_state import AgentState, AgentAction, CustomMemorySaver
from agentic_core.tools import (
    run_vision_analysis,
    llm,
    llm_generate,
    web_search_tool,
    extract_keywords_llm,
    generate_initial_prompt,
    generate_rag_web_enrichment_prompt,
    generate_consistency_prompt,
    generate_image_enrichment_prompt,
    generate_image_rag_web_enrichment_prompt,
    generate_history_enrichment_prompt,
    generate_history_rag_web_enrichment_prompt,
    run_ehr_analysis,
)
from pipelines.nlp_rag.hybrid_search import search
from xai.text_explainer import enrich_with_captum_xai

MAX_SELF_REFINE = 2
MAX_STEPS_GUARD = 7  # Maximum steps allowed before forcing exit to prevent infinite loops

def parse_tool_action_json(raw_text: str) -> Tuple[Optional[AgentAction], Optional[str]]:
    """
    Strict Structured Output & Auto-Fix JSON Parser:
    Safely extracts and validates JSON tool call objects from local LLM outputs.
    Supports Markdown ```json ... ``` blocks, raw JSON, or conversational prefixes.
    Returns (AgentAction, error_message).
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty LLM response"

    # Extract JSON string substring
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', raw_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        brace_match = re.search(r'({.*})', raw_text, re.DOTALL)
        json_str = brace_match.group(1) if brace_match else raw_text

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return None, "Parsed JSON output is not a JSON object"

        tool_name = data.get("tool_name") or data.get("tool")
        tool_input = data.get("tool_input") or data.get("input") or {}
        reasoning = data.get("reasoning") or data.get("thought")

        if not tool_name:
            return None, "Missing 'tool_name' property in tool call JSON"

        action = AgentAction(
            tool_name=str(tool_name),
            tool_input=tool_input if isinstance(tool_input, dict) else {},
            reasoning=str(reasoning) if reasoning else None
        )
        return action, None
    except Exception as e:
        error_msg = (
            f"Auto-Fix Notice: Your last tool call output was invalid JSON ({str(e)}). "
            f"Please format your output strictly as a JSON object: "
            f'{{"tool_name": "<name>", "tool_input": {{...}}, "reasoning": "..."}}'
        )
        return None, error_msg

def prune_state_context(cache: Dict[str, Any], max_text_len: int = 500) -> Dict[str, Any]:
    """
    State Pruning Mechanism (Context Window Control):
    Prunes heavy observations (1024d SciBERT RAG texts, HGT EHR visit history)
    to keep Llama 3.2 3B context window compact (<4k tokens) and prevent state bloat.
    """
    pruned = cache.copy()
    for k in ["rag_query", "rag_image", "web_context_query", "web_context_image"]:
        if k in pruned and isinstance(pruned[k], str) and len(pruned[k]) > max_text_len:
            pruned[k] = pruned[k][:max_text_len] + "... [Pruned for Context Window]"

    if "history_text" in pruned:
        ht = pruned["history_text"]
        if isinstance(ht, list):
            pruned["history_text"] = [str(t)[:300] for t in ht[:3]]
        elif isinstance(ht, str) and len(ht) > max_text_len:
            pruned["history_text"] = ht[:max_text_len] + "... [Pruned]"

    return pruned

def planner_node(state: AgentState) -> dict:
    plan = ["query_diag", "query_rag_web_enrichment"]
    if state.metadata.get("image_path"):
        plan.extend(["image_diag", "image_rag_web_enrichment"])
    if state.metadata.get("patient_id"):
        plan.extend(["history_diag", "history_rag_web_enrichment"])
    plan.append("consistency_check")
    return {"plan": plan, "intermediate_steps": [], "context_cache": {}, "step_count": 0}

def get_last_report(state: AgentState) -> str:
    for step in reversed(state.intermediate_steps):
        out = getattr(step, "tool_output", None)
        if out and len(str(out).strip()) > 30:
            return str(out)
    return ""

def react_agent_node(state: AgentState) -> dict:
    current_step_count = getattr(state, "step_count", 0) + 1
    plan = state.plan
    steps_taken = [step.tool_name for step in state.intermediate_steps]
    next_tool = next((step for step in plan if step not in steps_taken), None)

    if not next_tool:
        return {"step_count": current_step_count}

    cache = prune_state_context(state.context_cache)
    user_query = state.input
    metadata = state.metadata
    xray_image = str(metadata.get("image_path", ""))
    patient_id = metadata.get("patient_id")

    # Pre-fetch patient history if patient_id is present and history not yet cached
    if patient_id and "history_text" not in cache:
        try:
            rel_visits, _, visit_texts, _ = run_ehr_analysis(user_query, cache.get("xray_findings", []), patient_id)
            cache["history_text"] = visit_texts
            cache["history_retrieved"] = len(visit_texts) > 0
            cache["relevant_visits"] = rel_visits
            cache = prune_state_context(cache)
        except Exception as e:
            print(f"[AgentLoop] Warning pre-fetching patient history: {e}", file=sys.stderr)

    history_texts = cache.get("history_text")

    print(f"\n[Agent Debug] [react_agent_node] Step #{current_step_count}: {next_tool}")

    if next_tool == "query_diag":
        prompt = generate_initial_prompt(user_query, history_text=history_texts)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input={"query": user_query},
            tool_output=diagnosis_report
        )
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "query_rag_web_enrichment":
        prev_diag = get_last_report(state)
        keywords = extract_keywords_llm(user_query)
        keywords_query = ", ".join(keywords) if keywords else user_query
        web_results = web_search_tool(keywords_query, num_results=4)
        cache["web_context_query"] = "\n".join(web_results)
        rag, _ = search(user_query)
        cache["rag_query"] = str(rag)[:500]
        tool_input = {"prev_diag": prev_diag, "rag": cache["rag_query"], "web_context": cache.get("web_context_query"), "query": user_query}
        prompt = generate_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=cache["rag_query"], web_context=cache["web_context_query"], user_query=user_query, history_text=history_texts)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "image_diag":
        if not xray_image:
            completed_action = AgentAction(tool_name=next_tool, tool_input={}, tool_output=None, reasoning="Skipped image_diag")
            return {"intermediate_steps": state.intermediate_steps + [completed_action], "context_cache": cache, "step_count": current_step_count}

        result = run_vision_analysis(xray_image, user_id=str(patient_id or "default"))
        findings_raw = result.get("findings", [])
        xray_findings = [f[0] for f in findings_raw if isinstance(f, tuple) and str(f[0]).lower() not in ["no finding", "none", "", " "]]
        cache["xray_findings"] = xray_findings
        cache["xray_classification"] = findings_raw
        paths = result.get("paths", {})
        cache["original_xray_path"] = paths.get("original")
        cache["gradcam_overlay_path"] = paths.get("gradcam_overlay")
        cache["gradcam_segmented_path"] = paths.get("gradcam_segmented")
        cache["captum_image_path"] = paths.get("captum_image")
        if result.get("captum"):
            cache["cloud_top_words"] = result["captum"].get("top_words", [])

        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "xray_findings": xray_findings, "user_query": user_query}
        prompt = generate_image_enrichment_prompt(prev_diag=prev_diag, xray_findings=xray_findings, user_query=user_query, history_text=history_texts)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "image_rag_web_enrichment":
        xray_findings = cache.get("xray_findings", [])
        findings_str = ", ".join(xray_findings) if xray_findings else ""
        web_results = web_search_tool(f"chest X-ray {findings_str}", num_results=4) if findings_str else []
        cache["web_context_image"] = "\n".join(web_results)
        rag, _ = search(findings_str) if findings_str else ("", None)
        cache["rag_image"] = str(rag)[:500] if rag else ""
        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "rag": cache["rag_image"], "web_context": cache["web_context_image"], "xray_findings": xray_findings, "user_query": user_query}
        prompt = generate_image_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=cache["rag_image"], web_context=cache["web_context_image"], xray_findings=xray_findings, user_query=user_query, history_text=history_texts)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "history_diag":
        if not patient_id:
            completed_action = AgentAction(tool_name=next_tool, tool_input={}, tool_output=None, reasoning="Skipped history_diag")
            return {"intermediate_steps": state.intermediate_steps + [completed_action], "context_cache": cache, "step_count": current_step_count}

        image_labels = cache.get("xray_findings", [])
        if "history_text" not in cache:
            rel_visits, _, visit_texts, _ = run_ehr_analysis(user_query, image_labels, patient_id)
            cache["history_text"] = visit_texts
            cache["history_retrieved"] = len(visit_texts) > 0
            cache["relevant_visits"] = rel_visits
            cache = prune_state_context(cache)
        else:
            visit_texts = cache["history_text"]
        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "history": visit_texts, "user_query": user_query, "xray_findings": image_labels}
        prompt = generate_history_enrichment_prompt(prev_diag=prev_diag, history_text=visit_texts, user_query=user_query, xray_findings=image_labels)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "history_rag_web_enrichment":
        history_texts = cache.get("history_text", "")
        rag, _ = search(user_query)
        prev_diag = get_last_report(state)
        cache["rag_query"] = str(rag)[:500]
        tool_input = {"prev_diag": prev_diag, "rag": cache["rag_query"], "web_context": "", "history_text": history_texts, "user_query": user_query}
        prompt = generate_history_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=cache["rag_query"], web_context="", history_text=history_texts, user_query=user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    if next_tool == "consistency_check":
        prev_diag = get_last_report(state)
        xray_findings = cache.get("xray_findings", [])
        history = cache.get("history_text", "")
        tool_input = {"final_diag": prev_diag, "query": user_query, "xray_findings": xray_findings, "history": history}
        prompt = generate_consistency_prompt(final_diag=prev_diag, user_query=user_query, xray_findings=xray_findings, history_text=history)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache, "step_count": current_step_count}

    return {"step_count": current_step_count}

def run_tool_node(state: AgentState) -> dict:
    if not state.intermediate_steps:
        return {}
    action = state.intermediate_steps[-1]
    completed_action = AgentAction(tool_name=action.tool_name, tool_input=action.tool_input, tool_output=action.tool_output)
    return {"intermediate_steps": state.intermediate_steps[:-1] + [completed_action]}

def reflector_node(state: AgentState) -> dict:
    intermediate_steps = state.intermediate_steps
    if not intermediate_steps:
        return {}
    steps_taken = [s.tool_name for s in intermediate_steps]
    all_steps_done = all(tool in steps_taken for tool in state.plan)
    last_action = intermediate_steps[-1]
    just_did_consistency = (last_action.tool_name == "consistency_check")
    decision = "FINAL" if (all_steps_done or just_did_consistency) else "CONTINUE"
    last_action.reasoning = f"Decision: {decision}"
    return {"reflect_decision": decision, "intermediate_steps": intermediate_steps[:-1] + [last_action]}

def should_continue_reflector(state: AgentState) -> str:
    """
    Infinite Loop Protection Guard & Self-Refine Routing:
    - If step_count >= 7, force routing to final_answer.
    - If decision is REVISE, route back to planner for plan revision.
    - If decision is CONTINUE, route back to react_agent.
    - Otherwise, route to final_answer.
    """
    step_cnt = getattr(state, "step_count", 0)
    if step_cnt >= MAX_STEPS_GUARD:
        print(f"[AgentLoop Guard] Forced exit to final_answer: step_count={step_cnt} >= {MAX_STEPS_GUARD} (Infinite loop protection).", file=sys.stderr)
        return "final_answer"

    decision = getattr(state, "reflect_decision", None)
    if decision == "REVISE":
        return "planner"
    elif decision == "CONTINUE":
        return "react_agent"
    return "final_answer"

def final_answer_node(state: AgentState) -> dict:
    answer = get_last_report(state)
    context_cache = getattr(state, "context_cache", {}) or {}
    user_id = str(state.metadata.get("patient_id") or state.metadata.get("user_id") or "default")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = str(settings.OUTPUT_DIR / user_id / run_ts)

    history_texts = context_cache.get("history_text", [])
    history_retrieved = bool(context_cache.get("history_retrieved", False))
    if not history_texts or (isinstance(history_texts, list) and len(history_texts) == 0):
        history_retrieved = False
        history_str = None
    else:
        history_str = "\n".join(history_texts) if isinstance(history_texts, list) else str(history_texts)

    xray_findings = context_cache.get("xray_findings", [])
    has_image = bool(state.metadata.get("image_path")) and len(xray_findings) > 0

    try:
        xai_dict = enrich_with_captum_xai(
            answer=answer,
            query=state.input,
            xray_findings=", ".join(xray_findings) if has_image else None,
            history=history_str,
            history_retrieved=history_retrieved,
            user_id=user_id,
            out_dir=session_dir
        )
    except Exception as e:
        print(f"[AgentLoop] XAI enrichment warning: {e}", file=sys.stderr)
        xai_dict = {}

    cls_results = context_cache.get("xray_classification") or xai_dict.get("classification_results", [])

    agent_output = {
        "diagnosis": answer or "[No diagnosis generated]",
        "xai_report": xai_dict.get("explain_text") or "No XAI explanation generated.",
        "classification_results": cls_results if has_image else [],
        "original_xray": context_cache.get("original_xray_path") if has_image else None,
        "gradcam_overlay": context_cache.get("gradcam_overlay_path") if has_image else None,
        "gradcam_segmented": context_cache.get("gradcam_segmented_path") if has_image else None,
        "captum_image": context_cache.get("captum_image_path") or xai_dict.get("captum_image"),
        "top_words": context_cache.get("cloud_top_words") or xai_dict.get("top_words", {}),
        "history_retrieved": history_retrieved,
        "history_text": history_str if history_retrieved else None,
    }
    for k, v in xai_dict.items():
        if k.startswith("captum_"):
            agent_output[k] = v

    return {"agent_outcome": agent_output}

# --- BUILD & COMPILE LANGGRAPH WORKFLOW WITH LOOP & STATE PROTECTION ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("react_agent", react_agent_node)
workflow.add_node("run_tool", run_tool_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("final_answer", final_answer_node)

workflow.add_edge("planner", "react_agent")
workflow.add_edge("react_agent", "run_tool")
workflow.add_edge("run_tool", "reflector")
workflow.add_conditional_edges(
    "reflector",
    path=should_continue_reflector
)
workflow.add_edge("final_answer", END)
workflow.set_entry_point("planner")

runnable = workflow.compile()
