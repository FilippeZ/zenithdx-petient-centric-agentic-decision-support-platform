# backend/pipelines/vision/cloud_vision_client.py
from __future__ import annotations

import os
import requests
import base64
from typing import Dict, Any, Optional

class CloudVisionClient:
    """Hybrid Architecture Client for Cloud Vision & XAI Microservices (Hugging Face / Azure / Custom FastAPI)."""

    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("CLOUD_VISION_ENDPOINT_URL")
        self.api_key = api_key or os.getenv("CLOUD_VISION_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.endpoint_url)

    def analyze_xray(self, image_path: str, query: str = "") -> Dict[str, Any]:
        """Sends X-ray image to remote Cloud Vision API and receives predictions and Base64 Grad-CAM overlays."""
        if not self.is_configured():
            raise ValueError("Cloud Vision endpoint URL is not configured.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"query": query}
            response = requests.post(self.endpoint_url, files=files, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
