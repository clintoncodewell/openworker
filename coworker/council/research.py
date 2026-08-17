"""The council's web research: turn one long question into queries a search engine can
actually answer, run them, and hand back deduplicated results.

The council used to send its entire brief — often 1,400 characters — as the search string.
Every engine treats that as a bag of words and matches on the commonest ones. Measured
2026-08-17 against DuckDuckGo, same engine, same minute:

    the full brief   → a Yellow Pages listing for financial planners in Darwin, a YouTube
                       video, a LinkedIn post
    "advisor AI adoption wealth management 2026"
                     → MSCI, Family Wealth Report, fintech.global

Three of five panel members independently reported the first set as noise, so the run cost
what a researched council costs and produced an unresearched one. The fix is not a better
engine. It is asking a shorter question.

A model plans the queries because the useful ones are not in the text: a question about
where an industry lands by January needs "advisor AI adoption 2026", which appears nowhere
in the brief. When that call fails the heuristic below still beats sending the whole thing.
"""

from __future__ import annotations

import re
from typing import Any, Optional

MAX_QUERIES = 3
RESULTS_PER_QUERY = 5
MAX_RESULTS = 12
# Long enough for a real search phrase, short enough that no engine falls back to bag-of-words.
MAX_QUERY_CHARS = 120

QUERY_PLANNER = """\
You write web search queries. Below is a question a panel of analysts must research.

Write up to {n} search queries that would find EVIDENCE bearing on it — reports, data,
filings, industry coverage. Not the question rephrased.

Rules:
- Each query under 12 words. Keyword phrases, the way a person types into a search box.
- No punctuation, no quotes, no boolean operators, no site: filters.
- Cover different angles rather than three wordings of one angle.
- Include a year only when recency is the point.

Output the queries, one per line, and nothing else."""

# Words that carry no retrieval signal. Only used by the fallback, where a model was
# unavailable and something has to be cut.
_STOP = frozenset(
    """a an and are as at be been but by can could do does for from had has have how i if in
    into is it its may might must of on or should so than that the their them then there these
    they this those to was were what when where which who whose why will with would you your
    about above after again all also any because before being below between both did doing
    down during each few further here him his more most other over own same some such only
    very just now""".split()
)


def _clean(line: str) -> str:
    """One planner line reduced to a bare query, or "" if there is nothing usable left."""
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", (line or "").strip())
    text = text.strip().strip('"').strip("'")
    # A planner that ignores the format and writes prose gives itself away with punctuation.
    text = re.sub(r"[\"'`]|\bsite:\S+|\bOR\b|\bAND\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-")
    if not text or len(text) > MAX_QUERY_CHARS or len(text.split()) > 14:
        return ""
    return text


def fallback_query(question: str) -> str:
    """A usable query with no model call: the question's first sentence, stripped of the
    words that carry no retrieval signal, capped at twelve terms."""
    first = re.split(r"(?<=[.?!])\s", (question or "").strip(), maxsplit=1)[0]
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+-]*", first)
    kept = [w for w in words if w.lower() not in _STOP]
    return " ".join((kept or words)[:12])


def plan_queries(
    question: str, *, provider: Any = None, model: str = "", n: int = MAX_QUERIES
) -> list[str]:
    """Search queries for `question`, best effort. Never raises."""
    planned: list[str] = []
    if provider and model:
        try:
            turn = provider.complete(
                model=model,
                messages=[
                    {"role": "system", "content": QUERY_PLANNER.replace("{n}", str(n))},
                    {"role": "user", "content": question},
                ],
            )
            for line in (turn.text or "").splitlines():
                query = _clean(line)
                if query and query.lower() not in {q.lower() for q in planned}:
                    planned.append(query)
        except Exception:
            planned = []  # the fallback below is the whole point of not raising here
    if not planned:
        planned = [fallback_query(question)]
    return [q for q in planned if q][:n]


def search(
    question: str,
    *,
    secrets: Any = None,
    provider: Any = None,
    model: str = "",
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """Plan queries, run them, return deduplicated results. Never raises — the council runs
    fine without research, and a search outage must not cost the panel."""
    from ..web.tool import resolve_provider

    queries = plan_queries(question, provider=provider, model=model)
    try:
        engine = resolve_provider(secrets)
    except Exception as exc:
        return {"ok": False, "queries": queries, "error": f"{exc.__class__.__name__}: {exc}"}

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for query in queries:
        try:
            hits = engine.search(query, max_results=RESULTS_PER_QUERY)
        except Exception as exc:
            errors.append(f"{query}: {exc.__class__.__name__}")
            continue
        for hit in hits:
            row = hit.to_dict()
            url = (row.get("url") or "").strip()
            # Dedupe on URL: overlapping queries are the point, duplicate rows are not.
            if not url or url in seen:
                continue
            seen.add(url)
            row["query"] = query  # which angle found it — shown in the sources panel
            results.append(row)
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    out: dict[str, Any] = {
        "ok": bool(results),
        "provider": getattr(engine, "name", "?"),
        "queries": queries,
        "results": results,
    }
    if errors:
        out["errors"] = errors
    if not results and not errors:
        out["error"] = "the search returned nothing"
    return out
