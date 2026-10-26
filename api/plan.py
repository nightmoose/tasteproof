"""Vercel Python function: POST /api/plan {"query": str, "inject_fault": bool}; GET returns status."""
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasteproof.agent import Agent  # noqa: E402
from tasteproof.llm import GrokLLM, DEFAULT_MODEL  # noqa: E402
from tasteproof.qloo import QlooClient, default_mode  # noqa: E402
from tasteproof.ratelimit import LIMIT_MSG, RateLimiter, client_ip  # noqa: E402

LIMITER = RateLimiter()


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"qloo_mode": default_mode(), "model": DEFAULT_MODEL, "llm_configured": bool(os.environ.get("XAI_API_KEY"))})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(min(n, 10000)) or b"{}")
            q = str(req.get("query", "")).strip()[:300]
            if not q:
                return self._send(400, {"error": "query is required"})
            ip = client_ip(self.headers)
            ok, why = LIMITER.check(ip)
            if not ok:
                return self._send(429, {"error": LIMIT_MSG, "limit": why})
            result = Agent(GrokLLM(), QlooClient(), inject_fault=bool(req.get("inject_fault"))).run(q)
            self._send(200, result, {"X-RateLimit-Remaining": LIMITER.remaining(ip)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})
