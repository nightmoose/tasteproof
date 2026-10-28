"""Detect a requested place category in the user's query (e.g. "coffee shops")
and resolve it to Qloo tag IDs so place insights honor it."""
from __future__ import annotations

import re

# canonical category -> (query patterns, Qloo tag search terms in preference order)
CATEGORIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "coffee shop": ((r"coffee ?shops?", r"coffee ?houses?", r"cafes?", r"caf\u00e9s?", r"coffee", r"espresso"), ("coffee shop", "cafe", "coffee")),
    "cocktail bar": ((r"cocktail ?bars?", r"cocktails?", r"speakeas(?:y|ies)"), ("cocktail bar", "bar")),
    "wine bar": ((r"wine ?bars?",), ("wine bar",)),
    "brewery": ((r"brewer(?:y|ies)", r"craft beer"), ("brewery",)),
    "bookstore": ((r"book ?stores?", r"book ?shops?"), ("book store", "bookstore")),
    "record store": ((r"record ?stores?", r"vinyl"), ("record store",)),
    "museum": ((r"museums?", r"galler(?:y|ies)"), ("museum", "art museum")),
    "bakery": ((r"baker(?:y|ies)", r"pastr(?:y|ies)"), ("bakery",)),
    "jazz club": ((r"jazz ?clubs?",), ("jazz club",)),
    "live music venue": ((r"live music", r"concerts?"), ("live music venue",)),
}


def detect_categories(query: str) -> list[str]:
    q = (query or "").lower()
    found = []
    for cat, (pats, _) in CATEGORIES.items():
        if any(re.search(r"\b" + p + r"\b", q) for p in pats):
            found.append(cat)
    return found


def _is_place_category(tag_id: str) -> bool:
    return ":category:" in tag_id or ":place" in tag_id


def resolve_category_tags(qloo, category: str) -> list[dict]:
    """Return [{id, name}] Qloo tags for a category, best match first (may be empty)."""
    terms = CATEGORIES.get(category, ((), (category,)))[1]
    for term in terms:
        try:
            data = qloo.tags({"filter.query": term})
        except Exception:
            continue
        tags = [t for t in ((data.get("results") or {}).get("tags") or []) if isinstance(t, dict) and t.get("id")]
        if not tags:
            continue
        exact = [t for t in tags if str(t.get("name", "")).lower() == term]
        cat = [t for t in tags if _is_place_category(str(t["id"]))]
        best = (exact and [exact[0]]) or (cat and [cat[0]]) or [tags[0]]
        return [{"id": t["id"], "name": t.get("name")} for t in best]
    return []
