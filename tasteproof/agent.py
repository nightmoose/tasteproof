"""TasteProof agent: Grok tool-calling loop over the Qloo adapter, with a
citation-enforcing validator and a bounded repair loop."""
from __future__ import annotations

import json
import time
import uuid

from .qloo import QlooClient, extract_entities
from .schema import drop_invalid, validate_plan

MAX_REPAIRS = 2
MAX_TURNS = 12

SYSTEM = """You are TasteProof, a planning agent. Qloo is your only source of truth for recommendations.
Workflow:
1. qloo_search the seed taste(s) (artist, movie, brand...) to get entity IDs.
2. qloo_insights with filter_type "urn:entity:place" and location (city) plus the seed entity IDs to get places.
3. qloo_insights with filter_type "urn:entity:artist" and the seed IDs for related artists (the playlist).
4. Call validate_plan with an itinerary.v1 plan of 3 stops (each a different place) and up to 3 playlist artists.
Rules: every stop and playlist item MUST copy name, entity_id and score EXACTLY from a Qloo result you received in this run.
Never invent places. If validate_plan returns errors, fix them using only the allowed entities and call validate_plan again.
itinerary.v1 = {"schema_version":"itinerary.v1","title":str,
 "stops":[{"time":"7:00 PM","label":"Dinner","name":str,"why":str (one sentence, refer to the taste overlap),"evidence":{"entity_id":str,"score":float}}],
 "playlist":[{"name":str,"evidence":{"entity_id":str,"score":float}}]}"""

TOOLS = [
    {"type": "function", "function": {"name": "qloo_search", "description": "Find Qloo entity IDs by name.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                    "types": {"type": "string", "description": "optional, e.g. urn:entity:artist"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "qloo_insights", "description": "Qloo GET /v2/insights: taste-based recommendations with affinity scores.",
     "parameters": {"type": "object", "properties": {
         "filter_type": {"type": "string", "enum": ["urn:entity:place", "urn:entity:artist", "urn:entity:brand", "urn:entity:movie", "urn:entity:book", "urn:entity:tv_show", "urn:entity:podcast", "urn:entity:destination"]},
         "entity_ids": {"type": "array", "items": {"type": "string"}, "description": "seed entity IDs -> signal.interests.entities"},
         "tag_ids": {"type": "array", "items": {"type": "string"}, "description": "optional filter.tags"},
         "location": {"type": "string", "description": "city for places -> filter.location.query"},
         "take": {"type": "integer"}}, "required": ["filter_type"]}}},
    {"type": "function", "function": {"name": "qloo_tags", "description": "Qloo GET /v2/tags: find tag IDs by keyword.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "validate_plan", "description": "Submit an itinerary.v1 plan for schema + citation validation.",
     "parameters": {"type": "object", "properties": {"plan": {"type": "object"}}, "required": ["plan"]}}},
]


class Agent:
    def __init__(self, llm, qloo: QlooClient | None = None, inject_fault: bool = False):
        self.llm = llm
        self.qloo = qloo or QlooClient()
        # Demo option: corrupt the FIRST draft (strip one citation, swap in an
        # uncited place) so the repair loop is visible. Clearly logged in the trace.
        self.inject_fault = inject_fault

    def run(self, query: str) -> dict:
        t0 = time.time()
        run_id = uuid.uuid4().hex[:4]
        index: dict[str, dict] = {}
        trace: list[dict] = []
        attempts: list[dict] = []
        counts: dict[str, int] = {}
        steps: list[str] = []
        last_plan: dict | None = None
        final = None

        def log(kind, **kw):
            trace.append({"t_ms": round((time.time() - t0) * 1000), "kind": kind, **kw})

        def count(k):
            counts[k] = counts.get(k, 0) + 1

        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        log("user_query", query=query, qloo_mode=self.qloo.mode)
        for turn in range(MAX_TURNS):
            count("llm.call")
            msg = self.llm.chat(msgs, TOOLS)
            calls = msg.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                log("llm_text", content=(msg.get("content") or "")[:500])
                msgs.append({"role": "user", "content": "Use the tools and finish by calling validate_plan."})
                continue
            done = False
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._tool(name, args, index, attempts, counts, steps, log)
                if name == "validate_plan":
                    last_plan = args.get("plan")
                    if result.get("ok") or result.get("final"):
                        done = True
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)[:12000]})
            if done:
                break

        # finalize: drop anything that still fails
        if last_plan is None:
            last_plan = {"stops": []}
            log("no_plan", note="agent never submitted a plan")
        itin, dropped = drop_invalid(last_plan, index)
        final = {"ok": not dropped and bool(itin.stops), "dropped": dropped}
        log("final_check", **final, stops=len(itin.stops))
        steps.append("final check " + ("passed" if final["ok"] else f"dropped {len(dropped)}"))
        repairs = sum(1 for a in attempts if a["kind"] == "repair")
        return {
            "run_id": run_id, "query": query, "qloo_mode": self.qloo.mode, "model": getattr(self.llm, "model", "?"),
            "itinerary": itin.model_dump(), "attempts": attempts, "dropped": dropped, "repairs": repairs,
            "steps": steps, "tool_counts": counts, "elapsed_s": round(time.time() - t0, 2),
            "evidence_index": index, "qloo_requests": self.qloo.request_log, "trace": trace,
        }

    def _tool(self, name, args, index, attempts, counts, steps, log):
        try:
            if name == "qloo_search":
                count_key = "qloo.search"
                data = self.qloo.search(args.get("query", ""), args.get("types"))
                steps.append(f'search "{args.get("query", "")}"')
            elif name == "qloo_insights":
                count_key = "qloo.insights"
                p = {"filter.type": args.get("filter_type"), "take": args.get("take") or 8}
                if args.get("entity_ids"):
                    p["signal.interests.entities"] = ",".join(args["entity_ids"])
                if args.get("tag_ids"):
                    p["filter.tags"] = ",".join(args["tag_ids"])
                if args.get("location"):
                    p["filter.location.query"] = args["location"]
                data = self.qloo.insights(p)
                steps.append(f'insights {args.get("filter_type", "").split(":")[-1]}' + (f' in {args["location"]}' if args.get("location") else ""))
            elif name == "qloo_tags":
                count_key = "qloo.tags"
                data = self.qloo.tags({"filter.query": args.get("query", "")})
                steps.append(f'tags "{args.get("query", "")}"')
                counts[count_key] = counts.get(count_key, 0) + 1
                tags = (data.get("results") or {}).get("tags", [])
                log("tool", name=name, args=args, n=len(tags))
                return {"tags": [{"id": t.get("id"), "name": t.get("name")} for t in tags[:15]]}
            elif name == "validate_plan":
                counts["validate"] = counts.get("validate", 0) + 1
                n = len(attempts)
                kind = "draft" if n == 0 else "repair"
                if kind == "draft" and self.inject_fault:
                    args["plan"] = _inject_fault(args.get("plan"))
                    log("fault_injected", note="demo: stops[1] citation removed, stops[2] replaced with an uncited place")
                res = validate_plan(args.get("plan"), index)
                attempts.append({"attempt": n + 1, "kind": kind, "ok": res["ok"], "errors": res["errors"], "plan": args.get("plan")})
                log("validate", attempt=n + 1, attempt_kind=kind, ok=res["ok"], errors=res["errors"])
                steps.append(("LLM draft" if kind == "draft" else "repair") + (" → schema check passed" if res["ok"] else f" → {len(res['errors'])} errors"))
                if res["ok"]:
                    return {"ok": True}
                if n >= MAX_REPAIRS:  # draft + 2 repairs used up
                    return {"ok": False, "final": True, "errors": res["errors"], "note": "repair budget exhausted; invalid items will be dropped"}
                allowed = [{"entity_id": k, **{kk: v[kk] for kk in ("name", "type", "score")}} for k, v in index.items()]
                return {"ok": False, "errors": res["errors"], "repairs_left": MAX_REPAIRS - n,
                        "instruction": "Fix every error using ONLY these allowed entities (copy name, entity_id, score exactly), then call validate_plan again.",
                        "allowed_entities": allowed}
            else:
                return {"error": f"unknown tool {name}"}
        except Exception as e:
            log("tool_error", name=name, error=str(e)[:300])
            return {"error": str(e)[:300]}
        counts[count_key] = counts.get(count_key, 0) + 1
        ents = extract_entities(data)
        for e in ents:
            index[e["entity_id"]] = e
        log("tool", name=name, args=args, n=len(ents), mode=self.qloo.mode)
        return {"source": self.qloo.mode, "entities": ents}


def _inject_fault(plan):
    plan = json.loads(json.dumps(plan)) if isinstance(plan, dict) else {}
    stops = plan.get("stops") if isinstance(plan.get("stops"), list) else []
    if len(stops) > 1 and isinstance(stops[1], dict):
        stops[1].pop("evidence", None)
    if len(stops) > 2:
        stops[2] = {"time": stops[2].get("time", "11:30 PM") if isinstance(stops[2], dict) else "11:30 PM",
                    "label": "Nightcap", "name": "Rooftop Bar X", "why": "trendy spot",
                    "evidence": {"entity_id": "made-up-0000", "score": 0.95}}
    return plan
