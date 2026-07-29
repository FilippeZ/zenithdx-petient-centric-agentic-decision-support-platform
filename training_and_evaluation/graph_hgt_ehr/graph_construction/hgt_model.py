# backend/pipelines/graph_ehr/hgt_model.py
from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, Any, List
import pandas as pd

from config import settings

_NODES_LIST = None
_EDSTAYS_DF = None
_DIAGNOSIS_DF = None
_TRIAGE_DF = None
_VITALSIGN_DF = None

_VISIT_STAY_IDS = []
_VISIT_SUBJECT_IDS = []
_VISIT_DATETIMES = []

def _load_data_tables():
    global _NODES_LIST, _EDSTAYS_DF, _DIAGNOSIS_DF, _TRIAGE_DF, _VITALSIGN_DF
    global _VISIT_STAY_IDS, _VISIT_SUBJECT_IDS, _VISIT_DATETIMES
    
    if _NODES_LIST is None:
        data_dir = settings.DATA_DIR
        nodes_path = data_dir / "nodes_200k.pkl"
        edstays_path = data_dir / "cleaned_edstays.pkl"
        diag_path = data_dir / "cleaned_diagnosis.pkl"
        triage_path = data_dir / "cleaned_triage.pkl"
        vitals_path = data_dir / "cleaned_vitalsign.pkl"

        if nodes_path.exists():
            try:
                with open(nodes_path, "rb") as f:
                    _NODES_LIST = pickle.load(f)
                _VISIT_STAY_IDS = [n["stay_id"] for n in _NODES_LIST if n.get("type") == "Visit"]
                _VISIT_SUBJECT_IDS = [n["subject_id"] for n in _NODES_LIST if n.get("type") == "Visit"]
                _VISIT_DATETIMES = [n.get("intime", None) for n in _NODES_LIST if n.get("type") == "Visit"]
            except Exception as e:
                print(f"[GraphEHR] Error loading nodes_200k.pkl: {e}", file=sys.stderr)

        if edstays_path.exists():
            _EDSTAYS_DF = pd.read_pickle(edstays_path)
        if diag_path.exists():
            _DIAGNOSIS_DF = pd.read_pickle(diag_path)
        if triage_path.exists():
            _TRIAGE_DF = pd.read_pickle(triage_path)
        if vitals_path.exists():
            _VITALSIGN_DF = pd.read_pickle(vitals_path)

def get_visit_metadata_by_index(idx: int) -> Dict[str, Any]:
    """Retrieves metadata for a visit index in nodes_list."""
    _load_data_tables()
    if not _VISIT_STAY_IDS or idx >= len(_VISIT_STAY_IDS):
        return {"index": idx, "error": "Invalid index or data not loaded"}

    stay_id = _VISIT_STAY_IDS[idx]
    subject_id = _VISIT_SUBJECT_IDS[idx]
    intime = _VISIT_DATETIMES[idx]

    meta = {
        "index": idx,
        "stay_id": stay_id,
        "subject_id": subject_id,
        "intime": intime
    }

    if _EDSTAYS_DF is not None:
        stay_row = _EDSTAYS_DF[_EDSTAYS_DF["stay_id"] == stay_id]
        if not stay_row.empty:
            s = stay_row.iloc[0]
            meta.update({
                "gender": str(s.get("gender")),
                "race": str(s.get("race")),
                "disposition": str(s.get("disposition")),
                "arrival_transport": str(s.get("arrival_transport")),
                "outtime": str(s.get("outtime")) if "outtime" in s else None,
                "age": float(s.get("anchor_age")) if "anchor_age" in s else None,
                "insurance": str(s.get("insurance")) if "insurance" in s else None
            })

    if _TRIAGE_DF is not None:
        triage_row = _TRIAGE_DF[_TRIAGE_DF["stay_id"] == stay_id]
        if not triage_row.empty:
            tri = triage_row.iloc[0]
            meta["chiefcomplaint"] = tri.get("chiefcomplaint", None)
            for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp", "pain"]:
                if v in tri and not pd.isna(tri[v]):
                    meta[v] = tri[v]
            if "acuity" in tri and not pd.isna(tri["acuity"]):
                meta["acuity"] = tri["acuity"]

    if _DIAGNOSIS_DF is not None:
        diag_rows = _DIAGNOSIS_DF[_DIAGNOSIS_DF["stay_id"] == stay_id].sort_values("seq_num")
        if not diag_rows.empty:
            first_diag = diag_rows.iloc[0]
            meta["diagnosis_icd"] = first_diag["icd_code"]
            meta["diagnosis_title"] = first_diag["icd_title"]

    if _VITALSIGN_DF is not None:
        vs_rows = _VITALSIGN_DF[_VITALSIGN_DF["stay_id"] == stay_id]
        if not vs_rows.empty:
            vs = vs_rows.iloc[-1]
            for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp"]:
                if v in vs and not pd.isna(vs[v]):
                    meta[f"last_{v}"] = vs[v]
            if "rhythm" in vs and pd.notna(vs["rhythm"]):
                meta["last_rhythm"] = vs["rhythm"]

    return meta

def get_visit_note_by_index(idx: int) -> str:
    """Generates structured clinical visit note string by index."""
    _load_data_tables()
    if not _VISIT_STAY_IDS or idx >= len(_VISIT_STAY_IDS):
        return f"No clinical note found for index {idx}"

    stay_id = _VISIT_STAY_IDS[idx]
    note_lines = []

    if _TRIAGE_DF is not None:
        triage_row = _TRIAGE_DF[_TRIAGE_DF["stay_id"] == stay_id]
        chiefcomplaint = triage_row["chiefcomplaint"].values[0] if not triage_row.empty else None
        if chiefcomplaint:
            note_lines.append(f"Chief Complaint: {chiefcomplaint}")
        if not triage_row.empty:
            tri = triage_row.iloc[0]
            vitals = []
            for v in ["heartrate", "temperature", "resprate", "o2sat", "sbp", "dbp"]:
                if v in tri and not pd.isna(tri[v]):
                    vitals.append(f"{v.capitalize()}: {tri[v]}")
            if vitals:
                note_lines.append("Triage Vitals: " + ", ".join(vitals))

    if _DIAGNOSIS_DF is not None:
        diag_rows = _DIAGNOSIS_DF[_DIAGNOSIS_DF["stay_id"] == stay_id].sort_values("seq_num")
        if not diag_rows.empty:
            first_diag = diag_rows.iloc[0]
            note_lines.append(f"Diagnosis: {first_diag['icd_title']} (ICD: {first_diag['icd_code']})")

    if _EDSTAYS_DF is not None:
        stay_row = _EDSTAYS_DF[_EDSTAYS_DF["stay_id"] == stay_id]
        if not stay_row.empty:
            s = stay_row.iloc[0]
            note_lines.append(f"Admitted: {s.get('intime')} - Discharged: {s.get('outtime')} (Disposition: {s.get('disposition')})")

    return "\n".join(note_lines) if note_lines else f"No clinical note found for stay_id {stay_id}"
