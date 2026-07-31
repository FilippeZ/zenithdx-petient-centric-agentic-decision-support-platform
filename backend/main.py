# backend/main.py
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import settings
from api.v1.router import api_v1_router

app = FastAPI(
    title=settings.APP_NAME,
    description="ZenithDx Clinical AI Backend & Agentic Orchestration System",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 routes
app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(api_v1_router)  # Also mount at root level for legacy frontend compatibility

# Mount static file directories for outputs/uploads
if settings.OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")
if settings.UPLOAD_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

@app.get("/outputs/{path:path}")
def serve_outputs(path: str):
    p = settings.OUTPUT_DIR / path
    if p.exists() and p.is_file():
        return FileResponse(str(p))
    p_up = settings.UPLOAD_DIR / path
    if p_up.exists() and p_up.is_file():
        return FileResponse(str(p_up))
    fname = os.path.basename(path)
    for root, _, files in os.walk(str(settings.OUTPUT_DIR)):
        if fname in files:
            return FileResponse(os.path.join(root, fname))
    for root, _, files in os.walk(str(settings.UPLOAD_DIR)):
        if fname in files:
            return FileResponse(os.path.join(root, fname))
    raise HTTPException(404, f"Image file not found: {path}")

@app.get("/uploads/{path:path}")
def serve_uploads(path: str):
    p = settings.UPLOAD_DIR / path
    if p.exists() and p.is_file():
        return FileResponse(str(p))
    p_out = settings.OUTPUT_DIR / path
    if p_out.exists() and p_out.is_file():
        return FileResponse(str(p_out))
    fname = os.path.basename(path)
    for root, _, files in os.walk(str(settings.OUTPUT_DIR)):
        if fname in files:
            return FileResponse(os.path.join(root, fname))
    for root, _, files in os.walk(str(settings.UPLOAD_DIR)):
        if fname in files:
            return FileResponse(os.path.join(root, fname))
    raise HTTPException(404, f"Image file not found: {path}")

@app.on_event("startup")
def startup_event():
    import sys, requests
    print(f"[ZenithDx] Server starting. OLLAMA_HOST={settings.OLLAMA_HOST}, OLLAMA_MODEL={settings.OLLAMA_MODEL}", file=sys.stderr, flush=True)
    try:
        r = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=1)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            print(f"[ZenithDx] Ollama reachable. Installed models: {models}", file=sys.stderr, flush=True)
        else:
            print(f"[ZenithDx] Ollama status check returned code {r.status_code}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[ZenithDx] Ollama status check note: {e}", file=sys.stderr, flush=True)

@app.get("/")
def root():
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "docs_url": "/docs",
        "api_v1": "/api/v1",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Docker healthcheck endpoint — returns 200 when the server is ready."""
    return {"status": "healthy", "service": settings.APP_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=settings.DEBUG)
