import json

import pytest

from tasteproof.agent import Agent, MAX_REPAIRS
from tasteproof.qloo import QlooClient, QlooError, extract_entities
from tasteproof.schema import drop_invalid, validate_plan


@pytest.fixture
def qloo():
    return QlooClient(mode="mock")


def _index(q):
    idx = {}
    for d in (q.insights({"filter.type": "urn:entity:place"}), q.insights({"filter.type": "urn:entity:artist"})):
        for e in extract_entities(d):
            idx[e["entity_id"]] = e
    return idx


def _good_plan(idx):
    places = [e for e in idx.values() if e["type"] == "urn:entity:place"][:3]
    arts = [e for e in idx.values() if e["type"] == "urn:entity:artist"][:2]
    return {"schema_version": "itinerary.v1", "title": "t",
            "stops": [{"time": "7:00 PM", "label": "Dinner", "name": p["name"], "why": "fits",
                       "evidence": {"entity_id": p["entity_id"], "score": p["score"]}} for p in places],
            "playlist": [{"name": a["name"], "evidence": {"entity_id": a["entity_id"], "score": a["score"]}} for a in arts]}


def test_mock_mode_default_without_key(monkeypatch):
    monkeypatch.delenv("QLOO_API_KEY", raising=False)
    assert QlooClient().mode == "mock"


def test_live_mode_requires_key(monkeypatch):
    monkeypatch.delenv("QLOO_API_KEY", raising=False)
    with pytest.raises(QlooError):
        QlooClient(mode="live")


def test_insights_requires_filter_type(qloo):
    with pytest.raises(QlooError):
        qloo.insights({})


def test_mock_responses_labeled_and_logged(qloo):
    d = qloo.search("Phoebe Bridgers")
    assert d["source"] == "mock"
    assert d["results"][0]["name"] == "Phoebe Bridgers"
    qloo.insights({"filter.type": "urn:entity:place", "filter.location.query": "Austin", "take": 3})
    qloo.tags({"filter.query": "folk"})
    assert [r["path"] for r in qloo.request_log] == ["/search", "/v2/insights", "/v2/tags"]
    assert qloo.request_log[1]["result_count"] == 3
    assert all(r["mode"] == "mock" for r in qloo.request_log)


def test_live_request_shape(monkeypatch):
    seen = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps({"success": True, "results": {"entities": []}}).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["key"] = req.full_url, req.get_header("X-api-key")
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    QlooClient(mode="live", api_key="k123").insights({"filter.type": "urn:entity:place"})
    assert seen["url"].startswith("https://hackathon.api.qloo.com/v2/insights?filter.type=urn%3Aentity%3Aplace")
    assert seen["key"] == "k123"


def test_valid_plan_passes(qloo):
    idx = _index(qloo)
    assert validate_plan(_good_plan(idx), idx)["ok"]


def test_missing_citation_and_unknown_entity_rejected(qloo):
    idx = _index(qloo)
    plan = _good_plan(idx)
    del plan["stops"][1]["evidence"]
    plan["stops"][2]["evidence"]["entity_id"] = "fake-id"
    res = validate_plan(plan, idx)
    assert not res["ok"] and res["bad_stops"] == [1, 2]
    assert any("stops[1].evidence" in e for e in res["errors"])
    assert any("not found in this run" in e for e in res["errors"])


def test_score_and_name_must_match(qloo):
    idx = _index(qloo)
    plan = _good_plan(idx)
    plan["stops"][0]["evidence"]["score"] = 0.99
    plan["playlist"][0]["name"] = "Someone Else"
    res = validate_plan(plan, idx)
    assert res["bad_stops"] == [0] and res["bad_playlist"] == [0]


def test_entity_from_other_run_rejected(qloo):
    plan = _good_plan(_index(qloo))
    assert not validate_plan(plan, {})["ok"]  # empty index = nothing cited this run


def test_drop_invalid(qloo):
    idx = _index(qloo)
    plan = _good_plan(idx)
    plan["stops"][0]["evidence"]["entity_id"] = "nope"
    it, dropped = drop_invalid(plan, idx)
    assert dropped == ["stops[0]"] and len(it.stops) == 2


class FakeLLM:
    """Scripted LLM: searches, gets insights, then submits plans from `plans`."""
    model = "fake"

    def __init__(self, plans):
        self.plans, self.i = plans, 0

    def chat(self, messages, tools):
        self.i += 1
        if self.i == 1:
            return _call("qloo_search", {"query": "Phoebe Bridgers"})
        if self.i == 2:
            return _call("qloo_insights", {"filter_type": "urn:entity:place", "location": "Austin"})
        if self.i == 3:
            return _call("qloo_insights", {"filter_type": "urn:entity:artist"})
        p = self.plans[min(self.i - 4, len(self.plans) - 1)]
        return _call("validate_plan", {"plan": p(messages)})


def _call(name, args):
    return {"content": "", "tool_calls": [{"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_agent_repair_then_pass(qloo):
    idx = _index(QlooClient(mode="mock"))
    good = _good_plan(idx)
    bad = json.loads(json.dumps(good)); bad["stops"][1].pop("evidence")
    r = Agent(FakeLLM([lambda m: bad, lambda m: good]), qloo).run("q")
    assert [a["ok"] for a in r["attempts"]] == [False, True]
    assert r["repairs"] == 1 and r["dropped"] == [] and len(r["itinerary"]["stops"]) == 3


def test_agent_drops_after_max_repairs(qloo):
    idx = _index(QlooClient(mode="mock"))
    bad = _good_plan(idx); bad["stops"][0]["evidence"]["entity_id"] = "hallucinated"
    llm = FakeLLM([lambda m: bad])
    r = Agent(llm, qloo).run("q")
    assert len(r["attempts"]) == 1 + MAX_REPAIRS
    assert r["dropped"] == ["stops[0]"] and len(r["itinerary"]["stops"]) == 2
    assert any(t["kind"] == "final_check" for t in r["trace"])


def test_agent_fault_injection_repairs(qloo):
    idx = _index(QlooClient(mode="mock"))
    good = _good_plan(idx)
    r = Agent(FakeLLM([lambda m: good]), qloo, inject_fault=True).run("q")
    assert r["attempts"][0]["ok"] is False and r["attempts"][1]["ok"] is True
    assert any(t["kind"] == "fault_injected" for t in r["trace"])


def test_rate_limiter_per_ip_and_daily(monkeypatch):
    from tasteproof.ratelimit import RateLimiter
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("SUPABASE_RL_URL", raising=False)
    t = [1_000_000.0]
    rl = RateLimiter(per_ip=2, daily=3, clock=lambda: t[0])
    assert rl.check("a") == (True, None)
    assert rl.check("a") == (True, None)
    assert rl.check("a") == (False, "ip")
    assert rl.check("b") == (True, None)
    assert rl.check("c") == (False, "daily")
    t[0] += 86400  # next day, hour window passed
    assert rl.check("a") == (True, None)
