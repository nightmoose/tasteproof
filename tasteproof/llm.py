"""Minimal xAI (Grok) chat-completions client with tool calling."""
from __future__ import annotations

import json
import os
import urllib.request

XAI_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("XAI_MODEL", "grok-4.20-0309-non-reasoning")


class GrokLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is not set")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0.3}
        req = urllib.request.Request(XAI_URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]
