# backend/main.py
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

@app.on_event("startup")
def startup_event():
    import sys, requests
    print(f"[ZenithDx] Server starting. OLLAMA_HOST={settings.OLLAMA_HOST}, OLLAMA_MODEL={settings.OLLAMA_MODEL}", file=sys.stderr)
    try:
        r = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            print(f"[ZenithDx] ✅ Ollama reachable. Installed models: {models}", file=sys.stderr)
        else:
            print(f"[ZenithDx] ⚠️ Ollama status check returned code {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"[ZenithDx] ⚠️ Ollama server unreachable at {settings.OLLAMA_HOST}: {e}", file=sys.stderr)

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
