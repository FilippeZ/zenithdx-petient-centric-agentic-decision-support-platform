# backend/agentic_core/agent_loop.py
from __future__ import annotations

import os
import sys
import re
from datetime import datetime
from typing import Dict, Any, List

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

def planner_node(state: AgentState) -> dict:
    plan = ["query_diag", "query_rag_web_enrichment"]
    if state.metadata.get("image_path"):
        plan.extend(["image_diag", "image_rag_web_enrichment"])
    if state.metadata.get("patient_id"):
        plan.extend(["history_diag", "history_rag_web_enrichment"])
    plan.append("consistency_check")
    return {"plan": plan}

def get_last_report(state: AgentState) -> str:
    for step in reversed(state.intermediate_steps):
        out = getattr(step, "tool_output", None)
        if out and len(str(out).strip()) > 30:
            return str(out)
    return ""

def react_agent_node(state: AgentState) -> dict:
    plan = state.plan
    steps_taken = [step.tool_name for step in state.intermediate_steps]
    next_tool = next((step for step in plan if step not in steps_taken), None)
    if not next_tool:
        return {}

    cache = state.context_cache.copy()
    user_query = state.input
    metadata = state.metadata
    xray_image = str(metadata.get("image_path", ""))
    patient_id = metadata.get("patient_id")

    print(f"\n[Agent Debug] [react_agent_node] Step: {next_tool}")

    if next_tool == "query_diag":
        prompt = generate_initial_prompt(user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(
            tool_name=next_tool,
            tool_input={"query": user_query},
            tool_output=diagnosis_report
        )
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "query_rag_web_enrichment":
        prev_diag = get_last_report(state)
        keywords = extract_keywords_llm(user_query)
        keywords_query = ", ".join(keywords) if keywords else user_query
        web_results = web_search_tool(keywords_query, num_results=4)
        cache["web_context_query"] = "\n".join(web_results)
        rag, _ = search(user_query)
        rag = str(rag)[:1000]
        cache["rag_query"] = rag
        tool_input = {"prev_diag": prev_diag, "rag": rag, "web_context": cache.get("web_context_query"), "query": user_query}
        prompt = generate_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=rag, web_context=cache["web_context_query"], user_query=user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "image_diag":
        if not xray_image:
            completed_action = AgentAction(tool_name=next_tool, tool_input={}, tool_output=None, reasoning="Skipped image_diag")
            return {"intermediate_steps": state.intermediate_steps + [completed_action], "context_cache": cache}
        
        result = run_vision_analysis(xray_image, user_id=str(patient_id or "default"))
        findings_raw = result.get("findings", [])
        xray_findings = [f[0] for f in findings_raw if isinstance(f, tuple) and str(f[0]).lower() not in ["no finding", "none", "", " "]]
        cache["xray_findings"] = xray_findings
        cache["xray_classification"] = findings_raw
        paths = result.get("paths", {})
        cache["original_xray_path"] = paths.get("original")
        cache["gradcam_overlay_path"] = paths.get("gradcam_overlay")

        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "xray_findings": xray_findings, "user_query": user_query}
        prompt = generate_image_enrichment_prompt(prev_diag=prev_diag, xray_findings=xray_findings, user_query=user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "image_rag_web_enrichment":
        xray_findings = cache.get("xray_findings", [])
        findings_str = ", ".join(xray_findings) if xray_findings else ""
        web_results = web_search_tool(f"chest X-ray {findings_str}", num_results=4) if findings_str else []
        cache["web_context_image"] = "\n".join(web_results)
        rag, _ = search(findings_str) if findings_str else ("", None)
        cache["rag_image"] = str(rag)[:1000] if rag else ""
        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "rag": cache["rag_image"], "web_context": cache["web_context_image"], "xray_findings": xray_findings, "user_query": user_query}
        prompt = generate_image_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=cache["rag_image"], web_context=cache["web_context_image"], xray_findings=xray_findings, user_query=user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "history_diag":
        if not patient_id:
            completed_action = AgentAction(tool_name=next_tool, tool_input={}, tool_output=None, reasoning="Skipped history_diag")
            return {"intermediate_steps": state.intermediate_steps + [completed_action], "context_cache": cache}
        
        image_labels = cache.get("xray_findings", [])
        rel_visits, _, visit_texts, _ = run_ehr_analysis(user_query, image_labels, patient_id)
        cache["history_text"] = visit_texts
        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "history": visit_texts, "user_query": user_query, "xray_findings": image_labels}
        prompt = generate_history_enrichment_prompt(prev_diag=prev_diag, history_text=visit_texts, user_query=user_query, xray_findings=image_labels)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "history_rag_web_enrichment":
        history_texts = cache.get("history_text", "")
        rag, _ = search(user_query)
        prev_diag = get_last_report(state)
        tool_input = {"prev_diag": prev_diag, "rag": str(rag)[:1000], "web_context": "", "history_text": history_texts, "user_query": user_query}
        prompt = generate_history_rag_web_enrichment_prompt(prev_diag=prev_diag, rag=str(rag)[:1000], web_context="", history_text=history_texts, user_query=user_query)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    if next_tool == "consistency_check":
        prev_diag = get_last_report(state)
        xray_findings = cache.get("xray_findings", [])
        history = cache.get("history_text", "")
        tool_input = {"final_diag": prev_diag, "query": user_query, "xray_findings": xray_findings, "history": history}
        prompt = generate_consistency_prompt(final_diag=prev_diag, query=user_query, xray_findings=xray_findings, history=history)
        diagnosis_report = llm_generate(prompt)
        new_action = AgentAction(tool_name=next_tool, tool_input=tool_input, tool_output=diagnosis_report)
        return {"intermediate_steps": state.intermediate_steps + [new_action], "context_cache": cache}

    return {}

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

def final_answer_node(state: AgentState) -> dict:
    answer = get_last_report(state)
    context_cache = getattr(state, "context_cache", {}) or {}
    user_id = str(state.metadata.get("patient_id") or state.metadata.get("user_id") or "default")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = str(settings.OUTPUT_DIR / user_id / run_ts)

    try:
        xai_dict = enrich_with_captum_xai(
            answer=answer,
            query=state.input,
            xray_findings=", ".join(context_cache.get("xray_findings", [])),
            history=str(context_cache.get("history_text", "")),
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
        "classification_results": cls_results,
        "original_xray": context_cache.get("original_xray_path"),
        "gradcam_overlay": context_cache.get("gradcam_overlay_path"),
        "top_words": xai_dict.get("top_words", {}),
    }
    for k, v in xai_dict.items():
        if k.startswith("captum_"):
            agent_output[k] = v

    return {"agent_outcome": agent_output}

# --- BUILD & COMPILE LANGGRAPH WORKFLOW ---
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
    path=lambda state: "react_agent" if getattr(state, "reflect_decision", None) in ("REVISE", "CONTINUE") else "final_answer"
)
workflow.add_edge("final_answer", END)
workflow.set_entry_point("planner")

runnable = workflow.compile()
