# TasteProof

**A planning agent where every recommendation has to show its Qloo evidence.** Anything the LLM can't back with a real Qloo entity ID and affinity score from the current run is rejected, repaired, or dropped before you see it.

Built for the Qloo Agentic Hackathon. **Live demo:** https://tasteproof-qloo.vercel.app

> **Mock vs live:** until a Qloo hackathon API key is configured (`QLOO_API_KEY`), the app runs in **mock mode**: Qloo responses come from labeled fixtures in `fixtures/` (all names start with "Mock …"), and the UI shows a **MOCK DATA** badge and banner. Response field names were confirmed against the live hackathon API on 2026-10-27 (see the docstring in `tasteproof/qloo.py`). Setting `QLOO_API_KEY` switches every call to the live API with no code change; the production deployment runs in live mode.

## How it works
1. You type e.g. *"Plan a Friday night in Austin for fans of Phoebe Bridgers."*
2. A Grok model (xAI API, tool calling) drives a loop over four tools:
   - `qloo_search` → `GET /search` (seed entity IDs)
   - `qloo_insights` → `GET /v2/insights` with `filter.type`, `signal.interests.entities`, `filter.tags`, `filter.location.query`
   - `qloo_tags` → `GET /v2/tags`
   - `validate_plan` → schema + citation check
3. Every entity returned by Qloo in this run goes into an **evidence index**.
4. The plan must match the `itinerary.v1` Pydantic schema, and each stop / playlist item must cite an `entity_id` from the evidence index, with a matching `score` (±0.005) and name.
5. On failure the agent gets the errors (as JSON paths) plus the list of allowed entities and tries again: **at most 2 repairs**. Anything still failing is **dropped**.
6. The UI shows the plan cards with their citations, plus a **Validation trace** tab: each attempt, its errors, the final check, tool-call counts, the Qloo request log and the full trace.

A "stress test" checkbox deliberately corrupts the first draft (removes a citation, inserts an uncited "Rooftop Bar X") so the repair loop can be seen; the trace marks it as injected.

## Architecture
```
index.html            single-page UI (plan cards + validation trace)
api/plan.py           Vercel Python function: GET = status, POST {query, inject_fault} = run agent
tasteproof/qloo.py    Qloo adapter (mode=mock|live), logs every request
tasteproof/agent.py   Grok tool-calling loop, repair loop, trace
tasteproof/schema.py  itinerary.v1 (Pydantic) + citation check + drop_invalid
tasteproof/llm.py     xAI chat-completions client (stdlib only)
fixtures/             mock Qloo responses (labeled MOCK)
tests/                pytest suite (offline, scripted LLM)
cli.py                run the agent from a terminal
```
Live mode: base `https://hackathon.api.qloo.com`, header `X-Api-Key: $QLOO_API_KEY`, GET with query-string params.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
export XAI_API_KEY=...            # required
export QLOO_API_KEY=...           # optional; absent = mock mode
export XAI_MODEL=grok-4.20-0309-non-reasoning   # optional override
pytest -q
python cli.py "Plan a Friday night in Austin for fans of Phoebe Bridgers" [--fault]
```
Deploy: `vercel deploy --prod`, with `XAI_API_KEY` (and later `QLOO_API_KEY`) set as Vercel environment variables.

## Known gaps
- Qloo response field names (`entity_id`, `query.affinity`) and the place location parameter are taken from the docs and not yet verified against the live hackathon API.

## License
MIT — see [LICENSE](LICENSE).

## Abuse limits

`/api/plan` is rate-limited: 10 req/hour per IP (`PLAN_PER_IP_HOUR`) and 200 req/day global (`PLAN_DAILY_CAP`). Counters are in-memory per serverless instance plus a global counter in a free Supabase table (`tasteproof_hit` RPC, enabled by `SUPABASE_RL_URL`/`SUPABASE_RL_KEY`); alternatively set `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` to enforce them globally via Upstash. Over-limit requests get HTTP 429 and a "demo limit reached" message.
