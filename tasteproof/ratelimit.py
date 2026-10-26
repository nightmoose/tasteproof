"""Simple abuse limits for the public /api/plan demo.

Per-IP: PLAN_PER_IP_HOUR (default 10) requests per rolling hour.
Global: PLAN_DAILY_CAP (default 200) requests per UTC day.

Storage: in-memory per serverless instance (free, no setup). If
UPSTASH_REDIS_REST_URL/TOKEN are set, the global daily cap and per-IP counts
are also enforced via Upstash REST (fixed windows) so they hold across instances.
"""
import json
import os
import threading
import time
import urllib.request
from collections import deque

LIMIT_MSG = "Demo limit reached — this public demo caps usage to protect API credits. Please try again later."


class RateLimiter:
    def __init__(self, per_ip=None, daily=None, clock=time.time):
        self.per_ip = int(per_ip if per_ip is not None else os.environ.get("PLAN_PER_IP_HOUR", 10))
        self.daily = int(daily if daily is not None else os.environ.get("PLAN_DAILY_CAP", 200))
        self.clock = clock
        self._ip = {}
        self._day = None
        self._count = 0
        self._lock = threading.Lock()

    def check(self, ip):
        """Return (allowed, reason). Counts the request if allowed."""
        now = self.clock()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self._lock:
            if day != self._day:
                self._day, self._count = day, 0
            if self._count >= self.daily:
                return False, "daily"
            q = self._ip.setdefault(ip, deque())
            while q and q[0] <= now - 3600:
                q.popleft()
            if len(q) >= self.per_ip:
                return False, "ip"
            if len(self._ip) > 5000:  # bound memory
                for k in [k for k, v in self._ip.items() if not v and k != ip]:
                    del self._ip[k]
        remote = _upstash_check(ip, day, now, self.per_ip, self.daily)
        if remote:
            return False, remote
        with self._lock:
            q.append(now)
            self._count += 1
        return True, None


def _upstash_check(ip, day, now, per_ip, daily):
    url, tok = os.environ.get("UPSTASH_REDIS_REST_URL"), os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not (url and tok):
        return None
    hour = int(now // 3600)
    cmds = [["INCR", f"tp:day:{day}"], ["EXPIRE", f"tp:day:{day}", "90000"],
            ["INCR", f"tp:ip:{ip}:{hour}"], ["EXPIRE", f"tp:ip:{ip}:{hour}", "3700"]]
    try:
        req = urllib.request.Request(url.rstrip("/") + "/pipeline", data=json.dumps(cmds).encode(),
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=3).read())
        if int(res[0]["result"]) > daily:
            return "daily"
        if int(res[2]["result"]) > per_ip:
            return "ip"
    except Exception:
        return None  # fail open to in-memory limits
    return None


def client_ip(headers):
    for h in ("x-real-ip", "x-forwarded-for"):
        v = headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return "unknown"
