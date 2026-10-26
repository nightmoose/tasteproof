"""itinerary.v1 schema and the citation check."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

SCORE_TOLERANCE = 0.005


class Evidence(BaseModel):
    entity_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class Stop(BaseModel):
    time: str = Field(min_length=1)
    label: str = Field(min_length=1)
    name: str = Field(min_length=1)
    why: str = Field(min_length=1)
    evidence: Evidence


class PlaylistItem(BaseModel):
    name: str = Field(min_length=1)
    evidence: Evidence


class Itinerary(BaseModel):
    schema_version: Literal["itinerary.v1"] = "itinerary.v1"
    title: str = ""
    stops: list[Stop] = Field(default_factory=list)
    playlist: list[PlaylistItem] = Field(default_factory=list)


def _fmt_err(prefix: str, e: ValidationError) -> list[str]:
    return [f"{prefix}.{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]


def _check_item(prefix: str, raw, model, index: dict) -> list[str]:
    try:
        item = model.model_validate(raw)
    except ValidationError as e:
        return _fmt_err(prefix, e)
    errs = []
    src = index.get(item.evidence.entity_id)
    if src is None:
        errs.append(f'{prefix}.evidence.entity_id "{item.evidence.entity_id}": not found in this run\'s Qloo results')
        return errs
    if src.get("score") is None or abs(src["score"] - item.evidence.score) > SCORE_TOLERANCE:
        errs.append(f"{prefix}.evidence.score {item.evidence.score}: does not match Qloo score {src.get('score')}")
    if item.name.strip().lower() != str(src.get("name", "")).strip().lower():
        errs.append(f'{prefix}.name "{item.name}": does not match cited entity "{src.get("name")}"')
    return errs


def validate_plan(raw: dict, index: dict) -> dict:
    """Validate a raw plan against itinerary.v1 and this run's Qloo evidence index.

    Returns {"ok", "errors", "bad_stops", "bad_playlist"}; errors use JSON paths.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {"ok": False, "errors": ["plan: must be a JSON object"], "bad_stops": [], "bad_playlist": []}
    if raw.get("schema_version", "itinerary.v1") != "itinerary.v1":
        errors.append("schema_version: must be 'itinerary.v1'")
    stops = raw.get("stops")
    if not isinstance(stops, list) or not stops:
        errors.append("stops: required, must be a non-empty list")
        stops = []
    playlist = raw.get("playlist") or []
    if not isinstance(playlist, list):
        errors.append("playlist: must be a list")
        playlist = []
    bad_stops, bad_pl = [], []
    for i, s in enumerate(stops):
        e = _check_item(f"stops[{i}]", s, Stop, index)
        if e:
            bad_stops.append(i)
            errors += e
    for i, p in enumerate(playlist):
        e = _check_item(f"playlist[{i}]", p, PlaylistItem, index)
        if e:
            bad_pl.append(i)
            errors += e
    return {"ok": not errors, "errors": errors, "bad_stops": bad_stops, "bad_playlist": bad_pl}


def drop_invalid(raw: dict, index: dict) -> tuple[Itinerary, list[str]]:
    """Final fallback: keep only items that pass, drop the rest."""
    raw = raw if isinstance(raw, dict) else {}
    res = validate_plan(raw, index)
    stops = [s for i, s in enumerate(raw.get("stops") or []) if i not in res["bad_stops"]] if isinstance(raw.get("stops"), list) else []
    pl = [p for i, p in enumerate(raw.get("playlist") or []) if i not in res["bad_playlist"]] if isinstance(raw.get("playlist"), list) else []
    dropped = [f"stops[{i}]" for i in res["bad_stops"]] + [f"playlist[{i}]" for i in res["bad_playlist"]]
    it = Itinerary(title=str(raw.get("title", "")), stops=[Stop.model_validate(s) for s in stops],
                   playlist=[PlaylistItem.model_validate(p) for p in pl])
    return it, dropped
