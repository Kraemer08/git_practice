"""
Analytics for the beverage dashboard.

Turns the stored line items + outlet-day summaries into the payload the
dashboard renders: KPIs, category / beverage mix, day-over-day trends, top
items, and an opportunity matrix. Also builds a Claude-written "where's the
opportunity" briefing for the beverage director.

These reports carry revenue but no cost, so every "opportunity" here is framed
on revenue, mix share, price point and attachment-per-cover — not margin.
Adding item cost later would unlock true profit-based menu engineering.
"""

import calendar
import json
import os
import statistics
from collections import defaultdict

import anthropic

import database as db

MODEL = os.environ.get("BEV_DASHBOARD_MODEL", "claude-sonnet-5")

# Beverage sub-types, grouped.
ALCOHOL_TYPES = ["cocktail", "spirit", "wine_btg", "wine_bottle", "beer"]
NA_TYPES = ["na_beverage", "coffee_tea"]
BEVERAGE_TYPES = ALCOHOL_TYPES + NA_TYPES

# Friendly labels for the UI.
TYPE_LABELS = {
    "cocktail": "Cocktails",
    "spirit": "Spirits",
    "wine_btg": "Wine (by glass)",
    "wine_bottle": "Wine (bottle)",
    "beer": "Beer",
    "na_beverage": "N/A Beverages",
    "coffee_tea": "Coffee & Tea",
    "food": "Food",
    "other": "Other",
}


def _client() -> anthropic.Anthropic:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    from pathlib import Path
    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        return anthropic.Anthropic(auth_token=Path(token_file).read_text().strip())
    raise RuntimeError("No Anthropic credentials found.")


# ── Main dashboard payload ───────────────────────────────────────────────────

def dashboard(outlet: str, start: str | None = None, end: str | None = None) -> dict:
    items = db.get_items(outlet=outlet, start=start, end=end)
    days = db.get_outlet_days(outlet=outlet, start=start, end=end)

    total_covers = sum(d["net_covers"] or 0 for d in days)
    total_checks = sum(d["net_checks"] or 0 for d in days)

    # Reporting periods present in the data. A period may be one day or a whole
    # month; total_days sums their lengths so the timeframe is honest whether
    # the user uploaded 30 daily reports or one monthly report.
    periods = _periods_from_items(items)
    total_days = sum(p["num_days"] for p in periods)
    span_start = periods[0]["start"] if periods else None
    span_end = max((p["end"] for p in periods), default=None)

    total_net = sum(i["net_revenue"] for i in items)
    alcohol_net = sum(i["net_revenue"] for i in items if i["bev_type"] in ALCOHOL_TYPES)
    na_net = sum(i["net_revenue"] for i in items if i["bev_type"] in NA_TYPES)
    beverage_net = alcohol_net + na_net

    kpis = {
        "total_revenue": round(total_net, 2),
        "beverage_revenue": round(beverage_net, 2),
        "alcohol_revenue": round(alcohol_net, 2),
        "na_revenue": round(na_net, 2),
        "beverage_pct": _pct(beverage_net, total_net),
        "alcohol_pct": _pct(alcohol_net, total_net),
        "covers": total_covers,
        "checks": total_checks,
        "avg_check": round(_safe_div(sum(d["gross_revenue"] or 0 for d in days), total_checks), 2),
        "bev_per_cover": round(_safe_div(beverage_net, total_covers), 2),
        "alcohol_per_cover": round(_safe_div(alcohol_net, total_covers), 2),
        "has_covers": total_covers > 0,
        "days": total_days,
        "num_periods": len(periods),
    }

    return {
        "meta": {
            "outlet": outlet,
            "start": span_start,
            "end": span_end,
            "num_days": total_days,
            "num_periods": len(periods),
            "period_label": _period_label(span_start, span_end, total_days),
            "has_data": bool(items),
            "has_covers": total_covers > 0,
        },
        "kpis": kpis,
        "category_mix": _category_mix(items, total_net),
        "beverage_mix": _beverage_mix(items, beverage_net),
        "trend": _trend(items, days),
        "top_items": _top_items(items, limit=12),
        "opportunity": _opportunity_matrix(items),
        "movers": _movers(items, periods),
        "data_quality": _data_quality(items, days, periods),
    }


def _periods_from_items(items: list[dict]) -> list[dict]:
    """Distinct reporting periods in the data, sorted by start date."""
    seen: dict[tuple, int] = {}
    for i in items:
        start = i.get("business_date")
        if not start:
            continue
        key = (start, i.get("period_end_date") or start)
        seen[key] = int(i.get("num_days") or 1)
    return [{"start": k[0], "end": k[1], "num_days": v}
            for k, v in sorted(seen.items())]


def _period_label(start: str | None, end: str | None, num_days: int) -> str:
    """Human label for a span: a date, a month name, or a start→end range."""
    if not start:
        return "—"
    if num_days <= 1 or start == end or not end:
        return start
    sy, sm, sd = (int(x) for x in start.split("-"))
    ey, em, ed = (int(x) for x in end.split("-"))
    if sd == 1 and sy == ey and sm == em and ed == calendar.monthrange(sy, sm)[1]:
        return f"{calendar.month_abbr[sm]} {sy}"   # e.g. "Jun 2026"
    return f"{start} → {end}"


def _category_mix(items: list[dict], total_net: float) -> list[dict]:
    """Top-level split by POS revenue category (Food / Wine / Liquor / Beer / Misc)."""
    agg = defaultdict(lambda: {"net_revenue": 0.0, "units": 0.0})
    for i in items:
        cat = i["revenue_category"] or "Other"
        agg[cat]["net_revenue"] += i["net_revenue"]
        agg[cat]["units"] += i["items_sold"]
    out = [
        {
            "category": cat,
            "net_revenue": round(v["net_revenue"], 2),
            "units": int(v["units"]),
            "pct": _pct(v["net_revenue"], total_net),
        }
        for cat, v in agg.items()
    ]
    return sorted(out, key=lambda x: -x["net_revenue"])


def _beverage_mix(items: list[dict], beverage_net: float) -> list[dict]:
    """Split beverage revenue by sub-type (the beverage program breakdown)."""
    agg = defaultdict(lambda: {"net_revenue": 0.0, "units": 0.0, "n": 0})
    for i in items:
        if i["bev_type"] not in BEVERAGE_TYPES:
            continue
        t = i["bev_type"]
        agg[t]["net_revenue"] += i["net_revenue"]
        agg[t]["units"] += i["items_sold"]
        agg[t]["n"] += 1
    out = []
    for t in BEVERAGE_TYPES:
        if t not in agg:
            continue
        v = agg[t]
        out.append({
            "bev_type": t,
            "label": TYPE_LABELS[t],
            "group": "Alcohol" if t in ALCOHOL_TYPES else "Non-alcoholic",
            "net_revenue": round(v["net_revenue"], 2),
            "units": int(v["units"]),
            "distinct_items": v["n"],
            "avg_price": round(_safe_div(v["net_revenue"], v["units"]), 2),
            "pct": _pct(v["net_revenue"], beverage_net),
        })
    return sorted(out, key=lambda x: -x["net_revenue"])


def _trend(items: list[dict], days: list[dict]) -> list[dict]:
    """One point per reporting period (a day or a month) for the line chart."""
    covers_by_start = {d["business_date"]: (d["net_covers"] or 0) for d in days}
    by_period = defaultdict(lambda: {"total": 0.0, "alcohol": 0.0, "na": 0.0,
                                     "num_days": 1})
    for i in items:
        start = i.get("business_date")
        if not start:
            continue
        key = (start, i.get("period_end_date") or start)
        p = by_period[key]
        p["total"] += i["net_revenue"]
        p["num_days"] = int(i.get("num_days") or 1)
        if i["bev_type"] in ALCOHOL_TYPES:
            p["alcohol"] += i["net_revenue"]
        elif i["bev_type"] in NA_TYPES:
            p["na"] += i["net_revenue"]
    out = []
    for (start, end) in sorted(by_period):
        p = by_period[(start, end)]
        bev = p["alcohol"] + p["na"]
        covers = covers_by_start.get(start, 0)
        out.append({
            "date": start,
            "period_end": end,
            "label": _period_label(start, end, p["num_days"]),
            "num_days": p["num_days"],
            "total_revenue": round(p["total"], 2),
            "beverage_revenue": round(bev, 2),
            "alcohol_revenue": round(p["alcohol"], 2),
            "covers": covers,
            "bev_per_cover": round(_safe_div(bev, covers), 2),
        })
    return out


def _top_items(items: list[dict], limit: int = 12) -> list[dict]:
    """Top beverage items by net revenue over the range (aggregated across days)."""
    agg = defaultdict(lambda: {"net_revenue": 0.0, "units": 0.0, "bev_type": None,
                               "category": None})
    for i in items:
        if i["bev_type"] not in BEVERAGE_TYPES:
            continue
        key = (i["name"] or "?").strip()
        agg[key]["net_revenue"] += i["net_revenue"]
        agg[key]["units"] += i["items_sold"]
        agg[key]["bev_type"] = i["bev_type"]
        agg[key]["category"] = i["revenue_category"]
    rows = [
        {
            "name": name,
            "bev_type": v["bev_type"],
            "label": TYPE_LABELS.get(v["bev_type"], v["bev_type"]),
            "units": int(v["units"]),
            "net_revenue": round(v["net_revenue"], 2),
            "avg_price": round(_safe_div(v["net_revenue"], v["units"]), 2),
        }
        for name, v in agg.items()
    ]
    return sorted(rows, key=lambda x: -x["net_revenue"])[:limit]


def _opportunity_matrix(items: list[dict]) -> dict:
    """
    Volume x Price map for beverage items (no cost data, so this is a
    revenue/price lens, not a true profit menu-engineering quadrant).

      high volume + high price  -> Stars           (protect & feature)
      high volume + low price   -> Volume Drivers   (workhorses; test price)
      low volume  + high price  -> Premium/Occasional (feature or reposition)
      low volume  + low price   -> Sleepers         (re-merchandise or cut)
    """
    # Aggregate beverage items across the range.
    agg = defaultdict(lambda: {"net_revenue": 0.0, "units": 0.0, "bev_type": None})
    for i in items:
        if i["bev_type"] not in BEVERAGE_TYPES:
            continue
        if i["net_revenue"] <= 0 or i["items_sold"] <= 0:
            continue
        key = (i["name"] or "?").strip()
        agg[key]["net_revenue"] += i["net_revenue"]
        agg[key]["units"] += i["items_sold"]
        agg[key]["bev_type"] = i["bev_type"]

    rows = []
    for name, v in agg.items():
        rows.append({
            "name": name,
            "bev_type": v["bev_type"],
            "label": TYPE_LABELS.get(v["bev_type"], v["bev_type"]),
            "units": int(v["units"]),
            "net_revenue": round(v["net_revenue"], 2),
            "avg_price": round(_safe_div(v["net_revenue"], v["units"]), 2),
        })

    if len(rows) < 4:
        return {"enough_data": False, "items": rows, "thresholds": None,
                "quadrants": {}}

    units_median = statistics.median([r["units"] for r in rows])
    price_median = statistics.median([r["avg_price"] for r in rows])

    quadrants = {"star": [], "volume": [], "premium": [], "sleeper": []}
    for r in rows:
        hi_vol = r["units"] >= units_median
        hi_price = r["avg_price"] >= price_median
        r["quadrant"] = (
            "star" if hi_vol and hi_price else
            "volume" if hi_vol and not hi_price else
            "premium" if hi_price and not hi_vol else
            "sleeper"
        )
        quadrants[r["quadrant"]].append(r)

    for q in quadrants.values():
        q.sort(key=lambda x: -x["net_revenue"])

    return {
        "enough_data": True,
        "thresholds": {"units_median": units_median, "price_median": round(price_median, 2)},
        "items": rows,
        "quadrants": quadrants,
        "labels": {
            "star": "Stars — high demand, high ticket. Protect & feature.",
            "volume": "Volume Drivers — high demand, lower ticket. Workhorses; test a small price move.",
            "premium": "Premium / Occasional — high ticket, low pull. Feature, reposition, or reprice.",
            "sleeper": "Sleepers — low pull, low ticket. Re-merchandise or cut to free up the list.",
        },
    }


def _movers(items: list[dict], periods: list[dict]) -> dict:
    """
    Compare the two most recent reporting periods (days or months) to surface
    beverage items gaining or losing momentum. Needs 2+ periods of data.
    """
    if len(periods) < 2:
        return {"enough_data": False, "gainers": [], "decliners": [], "note":
                "Upload a second period (another day or month) to see momentum."}

    prev_p, cur_p = periods[-2], periods[-1]
    prev_key = (prev_p["start"], prev_p["end"])
    cur_key = (cur_p["start"], cur_p["end"])

    def bucket(key: tuple) -> dict:
        agg = defaultdict(lambda: {"net_revenue": 0.0, "bev_type": None})
        for i in items:
            if i["bev_type"] not in BEVERAGE_TYPES:
                continue
            ikey = (i.get("business_date"), i.get("period_end_date") or i.get("business_date"))
            if ikey != key:
                continue
            name = (i["name"] or "?").strip()
            agg[name]["net_revenue"] += i["net_revenue"]
            agg[name]["bev_type"] = i["bev_type"]
        return agg

    prev, cur = bucket(prev_key), bucket(cur_key)
    names = set(prev) | set(cur)
    deltas = []
    for name in names:
        p = prev.get(name, {}).get("net_revenue", 0.0)
        c = cur.get(name, {}).get("net_revenue", 0.0)
        bt = (cur.get(name) or prev.get(name)).get("bev_type")
        deltas.append({
            "name": name, "bev_type": bt, "label": TYPE_LABELS.get(bt, bt),
            "prev": round(p, 2), "current": round(c, 2), "delta": round(c - p, 2),
        })
    gainers = sorted([d for d in deltas if d["delta"] > 0], key=lambda x: -x["delta"])[:8]
    decliners = sorted([d for d in deltas if d["delta"] < 0], key=lambda x: x["delta"])[:8]
    return {
        "enough_data": True,
        "prev_period": _period_label(prev_p["start"], prev_p["end"], prev_p["num_days"]),
        "current_period": _period_label(cur_p["start"], cur_p["end"], cur_p["num_days"]),
        "gainers": gainers,
        "decliners": decliners,
    }


def _data_quality(items: list[dict], days: list[dict], periods: list[dict]) -> dict:
    starts_with_covers = {d["business_date"] for d in days if (d["net_covers"] or 0) > 0}
    missing = [p for p in periods if p["start"] not in starts_with_covers]

    # Flag overlapping periods (e.g. a monthly report AND a daily report inside
    # it), which would double-count revenue.
    overlaps = []
    for a, b in zip(periods, periods[1:]):
        if a["end"] and b["start"] and b["start"] <= a["end"]:
            overlaps.append([_period_label(a["start"], a["end"], a["num_days"]),
                             _period_label(b["start"], b["end"], b["num_days"])])

    return {
        "num_periods": len(periods),
        "days_missing_covers": [_period_label(p["start"], p["end"], p["num_days"]) for p in missing],
        "overlapping_periods": overlaps,
        "has_cost_data": False,
    }


# ── AI opportunity briefing ──────────────────────────────────────────────────

_INSIGHT_TOOL = {
    "name": "beverage_briefing",
    "description": "Return a beverage-program opportunity briefing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {"type": "string", "description": "One-sentence read on the beverage program for this period."},
            "opportunities": {
                "type": "array",
                "description": "3-5 specific, actionable opportunities to grow or improve the beverage program.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "detail": {"type": "string", "description": "1-2 sentences with the specific action and the reasoning, citing the numbers."},
                        "evidence": {"type": "string", "description": "The key metric/number backing this up."},
                        "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["title", "detail", "impact"],
                },
            },
            "watch": {
                "type": "array",
                "description": "0-3 risks or items to keep an eye on.",
                "items": {"type": "string"},
            },
        },
        "required": ["headline", "opportunities"],
    },
}


def ai_insights(outlet: str, start: str | None = None, end: str | None = None) -> dict:
    """Ask Claude to read the aggregated numbers and brief the beverage director."""
    data = dashboard(outlet, start, end)
    if not data["meta"]["has_data"]:
        return {"headline": "No data yet for this outlet and range.",
                "opportunities": [], "watch": []}

    payload = {
        "outlet": outlet,
        "date_range": [data["meta"]["start"], data["meta"]["end"]],
        "num_days": data["meta"]["num_days"],
        "kpis": data["kpis"],
        "category_mix": data["category_mix"],
        "beverage_mix": data["beverage_mix"],
        "top_items": data["top_items"],
        "opportunity_quadrants": {
            k: [f"{r['name']} ({r['label']}, {r['units']}u, ${r['net_revenue']:.0f}, ${r['avg_price']:.0f}/ea)"
                for r in v]
            for k, v in (data["opportunity"].get("quadrants") or {}).items()
        },
        "movers": data["movers"],
        "trend": data["trend"],
        "notes": "Revenue-only data; no item cost/margin available.",
    }

    system = (
        "You are the beverage director's analyst for a luxury resort restaurant "
        "group. You are briefing the director on where the biggest opportunities "
        "are to grow and sharpen the beverage program at one outlet, using its "
        "POS product-mix data. Be specific and quantitative — cite the actual "
        "items, categories and dollar figures. Think like an operator: beverage "
        "attachment per cover, wine-by-the-glass vs bottle movement, cocktail "
        "menu performance, premium pours, and list items that aren't earning "
        "their place. The data is revenue-only (no cost), so frame margin "
        "opportunities as hypotheses to confirm with cost data rather than "
        "certainties. Return the briefing via the beverage_briefing tool."
    )
    user = (
        "Here is the aggregated beverage data for the period. Brief me on the "
        "top opportunities to direct the beverage program.\n\n"
        + json.dumps(payload, indent=2)
    )

    resp = _client().messages.create(
        model=MODEL, max_tokens=2000, system=system,
        tools=[_INSIGHT_TOOL],
        tool_choice={"type": "tool", "name": "beverage_briefing"},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    return {"headline": "Could not generate insights.", "opportunities": [], "watch": []}


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0
