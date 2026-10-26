"""Qloo adapter. Every Qloo call in TasteProof goes through QlooClient.

mode="live": GET https://hackathon.api.qloo.com with X-Api-Key from env QLOO_API_KEY.
mode="mock": serves fixtures from fixtures/ that follow the documented response
shapes. Mock results are labeled (source="mock") and are NOT real Qloo data.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://hackathon.api.qloo.com"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def default_mode() -> str:
    return "live" if os.environ.get("QLOO_API_KEY") else "mock"


class QlooError(RuntimeError):
    pass


class QlooClient:
    def __init__(self, mode: str | None = None, api_key: str | None = None, timeout: float = 15.0):
        self.mode = mode or default_mode()
        if self.mode not in ("mock", "live"):
            raise ValueError("mode must be 'mock' or 'live'")
        self.api_key = api_key or os.environ.get("QLOO_API_KEY")
        if self.mode == "live" and not self.api_key:
            raise QlooError("live mode needs QLOO_API_KEY")
        self.timeout = timeout
        self.request_log: list[dict] = []

    # ---- public API -------------------------------------------------
    def search(self, query: str, types: str | None = None) -> dict:
        params = {"query": query}
        if types:
            params["types"] = types
        return self._get("/search", params)

    def insights(self, params: dict) -> dict:
        if not params.get("filter.type"):
            raise QlooError("filter.type is required for /v2/insights")
        return self._get("/v2/insights", params)

    def tags(self, params: dict) -> dict:
        return self._get("/v2/tags", params)

    # ---- internals --------------------------------------------------
    def _get(self, path: str, params: dict) -> dict:
        params = {k: v for k, v in params.items() if v not in (None, "")}
        t0 = time.time()
        entry = {"mode": self.mode, "method": "GET", "path": path, "params": params}
        try:
            data = self._mock(path, params) if self.mode == "mock" else self._live(path, params)
            entry["status"] = 200
            entry["result_count"] = len(extract_entities(data))
            return data
        except Exception as e:  # logged, then re-raised
            entry["status"] = getattr(e, "code", "error")
            entry["error"] = str(e)[:300]
            raise
        finally:
            entry["ms"] = round((time.time() - t0) * 1000)
            self.request_log.append(entry)

    def _live(self, path: str, params: dict) -> dict:
        url = BASE_URL + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"X-Api-Key": self.api_key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _mock(self, path: str, params: dict) -> dict:
        if path == "/search":
            data = _load("search.json")
            q = params.get("query", "").lower()
            hits = [r for r in data["results"] if q and (q in r["name"].lower() or r["name"].lower() in q)]
            return {"source": "mock", "results": hits or data["results"][:1]}
        if path == "/v2/tags":
            data = _load("tags.json")
            q = params.get("filter.query", "").lower()
            tags = [t for t in data["results"]["tags"] if q in t["name"].lower()] if q else data["results"]["tags"]
            return {"source": "mock", "success": True, "results": {"tags": tags}}
        if path == "/v2/insights":
            t = params["filter.type"]
            name = {"urn:entity:place": "insights_place.json", "urn:entity:artist": "insights_artist.json"}.get(t)
            if not name:
                return {"source": "mock", "success": True, "results": {"entities": []}}
            data = _load(name)
            take = int(params.get("take", 10))
            ents = data["results"]["entities"][:take]
            return {"source": "mock", "success": True, "results": {"entities": ents}}
        raise QlooError(f"unknown path {path}")


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def extract_entities(data: dict) -> list[dict]:
    """Normalize search / insights responses to [{entity_id, name, type, score, tags}]."""
    if not isinstance(data, dict):
        return []
    res = data.get("results")
    raw = res.get("entities", []) if isinstance(res, dict) else (res or [])
    out = []
    for e in raw:
        if not isinstance(e, dict) or "entity_id" not in e:
            continue
        q = e.get("query") or {}
        score = q.get("affinity", e.get("popularity"))
        out.append({
            "entity_id": e["entity_id"],
            "name": e.get("name", ""),
            "type": e.get("subtype") or (e.get("types") or [e.get("type")])[0],
            "score": round(float(score), 4) if score is not None else None,
            "tags": [t.get("name") for t in (e.get("tags") or []) if isinstance(t, dict)][:4],
        })
    return out
