# backend/api/v1/graph.py
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from pipelines.graph_ehr.graph_visualizer import ZenithGraphVisualizer

router = APIRouter(prefix="/graph", tags=["Heterogeneous Graph EHR Visualizer"])

_VISUALIZER = ZenithGraphVisualizer()

@router.get("/patient/{patient_id}/html", response_class=HTMLResponse)
async def get_patient_graph_html(patient_id: str) -> HTMLResponse:
    """
    Dynamically generates interactive dark-mode PyVis HTML ego-graph for a patient.
    Embedded via <iframe> in the React/Vite doctor workstation dashboard.
    """
    try:
        html_content = _VISUALIZER.generate_patient_subgraph_html_string(patient_id)
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate patient graph HTML: {str(e)}")

@router.post("/export/gephi")
async def export_gephi_graphml(output_path: str = Query("zenithdx_full_graph.graphml")) -> Dict[str, Any]:
    """
    Exports full heterogeneous clinical graph to GraphML for Gephi ForceAtlas2 thesis visualization.
    """
    try:
        exported_file = _VISUALIZER.export_to_gephi(output_path=output_path)
        return {
            "status": "success",
            "message": "Full heterogeneous graph exported to GraphML format successfully.",
            "file_path": exported_file,
            "instructions": "Open file in Gephi, apply 'ForceAtlas2' layout, and color partition by 'type'."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export GraphML for Gephi: {str(e)}")
