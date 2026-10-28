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


# Synthetic entities in the live response shapes (confirmed 2026-10-27). Values are made up;
# real Qloo responses are not stored in this public repo.
LIVE_SHAPE_INSIGHTS = {"success": True, "results": {"entities": [{
    "name": "Example Cafe", "entity_id": "00000000-0000-0000-0000-000000000001", "type": "urn:entity",
    "subtype": "urn:entity:place", "popularity": 0.5, "disambiguation": "1 Main St",
    "properties": {"address": "1 Main St, Austin, TX"},
    "query": {"affinity": 0.81, "distance": 0, "measurements": {"audience_growth": 0}},
    "tags": [{"id": "urn:tag:accessibility:place:wheelchair_accessible_entrance", "name": "Wheelchair accessible entrance", "type": "urn:tag:accessibility:place"},
             {"id": "urn:tag:service_type:qloo:takeout", "name": "Takeout", "type": "urn:tag:service_type:qloo"},
             {"id": "urn:tag:ambience:qloo:cozy", "name": "Cozy", "type": "urn:tag:ambience:qloo"},
             {"id": "urn:tag:genre:place:cafe", "name": "Cafe", "type": "urn:tag:genre:place"}]},
    {"name": "No Signal Bar", "entity_id": "00000000-0000-0000-0000-000000000002", "type": "urn:entity",
     "subtype": "urn:entity:place", "popularity": 0.66, "query": {"distance": 0}, "tags": []}]}}
LIVE_SHAPE_SEARCH = {"results": [{"name": "Example Artist", "entity_id": "00000000-0000-0000-0000-000000000003",
    "types": ["urn:entity:artist"], "popularity": 0.9,
    "tags": [{"name": "Indie", "tag_id": "urn:tag:genre:indie", "type": "urn:tag:genre", "value": "urn:tag:genre:indie"}]}]}


def test_extract_live_insights_shape():
    a, b = extract_entities(LIVE_SHAPE_INSIGHTS)
    assert a["type"] == "urn:entity:place" and a["score"] == 0.81
    assert a["tags"][:2] == ["Cafe", "Cozy"] and "Takeout" not in a["tags"][:2]
    assert a["address"] == "1 Main St, Austin, TX"
    assert b["score"] == 0.66 and b["tags"] == []  # no affinity without a signal -> popularity


def test_extract_live_search_shape():
    (e,) = extract_entities(LIVE_SHAPE_SEARCH)
    assert e["type"] == "urn:entity:artist" and e["score"] == 0.9 and e["tags"] == ["Indie"]


# ---- requested category + insights fallback --------------------------------
from tasteproof.intent import detect_categories, resolve_category_tags  # noqa: E402


class StubQloo(QlooClient):
    """Mock client whose insights return nothing when given `empty_when` params."""

    def __init__(self, empty_when=lambda p: False, tags=None):
        super().__init__(mode="mock")
        self.empty_when, self.stub_tags, self.calls = empty_when, tags, []

    def insights(self, params):
        self.calls.append(dict(params))
        if self.empty_when(params):
            return {"success": True, "results": {"entities": []}}
        return super().insights(params)

    def tags(self, params):
        if self.stub_tags is not None:
            q = params.get("filter.query", "")
            return {"success": True, "results": {"tags": self.stub_tags.get(q, [])}}
        return super().tags(params)


COFFEE_TAG = {"id": "urn:tag:category:place:coffee_shop", "name": "Coffee shop", "type": "urn:tag:category:place"}


def test_detect_categories():
    assert detect_categories("Date night in Chicago, we love coffee shops") == ["coffee shop"]
    assert detect_categories("a cozy cafe then cocktails") == ["coffee shop", "cocktail bar"]
    assert detect_categories("date night for Radiohead fans") == []


def test_resolve_category_prefers_exact_then_falls_through_terms():
    q = StubQloo(tags={"coffee shop": [], "cafe": [{"id": "urn:tag:genre:x", "name": "Cafe racer"}, COFFEE_TAG | {"name": "Cafe"}]})
    assert resolve_category_tags(q, "coffee shop") == [{"id": COFFEE_TAG["id"], "name": "Cafe"}]
    assert resolve_category_tags(StubQloo(tags={}), "coffee shop") == []


def test_requested_category_filters_place_insights():
    q = StubQloo(tags={"coffee shop": [COFFEE_TAG]})
    out = Agent(FakeLLM([lambda m: _good_plan(_index(QlooClient(mode="mock")))]), q).run("date night in Austin, coffee shops please")
    place_calls = [c for c in q.calls if c["filter.type"] == "urn:entity:place"]
    assert place_calls[0]["filter.tags"] == COFFEE_TAG["id"]
    assert any(t["kind"] == "category_filter" for t in out["trace"])
    assert any("category" in s for s in out["steps"])


def test_no_category_no_tag_filter():
    q = StubQloo()
    Agent(FakeLLM([lambda m: _good_plan(_index(QlooClient(mode="mock")))]), q).run("date night in Austin")
    assert all("filter.tags" not in c for c in q.calls)


def test_insights_fallback_drops_signal_when_empty():
    q = StubQloo(empty_when=lambda p: "signal.interests.entities" in p)
    llm = FakeLLM([lambda m: _good_plan(_index(QlooClient(mode="mock")))])
    a = Agent(llm, q)
    steps, trace = [], []
    res = a._tool("qloo_insights", {"filter_type": "urn:entity:place", "location": "Austin", "entity_ids": ["X"]},
                  {}, [], {}, steps, lambda k, **kw: trace.append({"kind": k, **kw}))
    assert res["entities"] and "fallback" in res["note"]
    assert "signal.interests.entities" not in q.calls[-1]
    assert any(t["kind"] == "insights_fallback" for t in trace)


def test_insights_fallback_exhausted_returns_note():
    q = StubQloo(empty_when=lambda p: True)
    a = Agent(FakeLLM([]), q)
    res = a._tool("qloo_insights", {"filter_type": "urn:entity:place", "location": "Austin", "entity_ids": ["X"], "tag_ids": ["t"]},
                  {}, [], {}, [], lambda k, **kw: None)
    assert res["entities"] == [] and "no results" in res["note"]
    assert len(q.calls) == 4  # original + 3 relaxations
