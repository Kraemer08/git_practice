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
    dates = sorted({i["business_date"] for i in items if i["business_date"]})

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
        "days": len(dates),
    }

    return {
        "meta": {
            "outlet": outlet,
            "start": start or (dates[0] if dates else None),
            "end": end or (dates[-1] if dates else None),
            "dates": dates,
            "num_days": len(dates),
            "has_data": bool(items),
            "has_covers": total_covers > 0,
        },
        "kpis": kpis,
        "category_mix": _category_mix(items, total_net),
        "beverage_mix": _beverage_mix(items, beverage_net),
        "trend": _trend(items, days),
        "top_items": _top_items(items, limit=12),
        "opportunity": _opportunity_matrix(items),
        "movers": _movers(items, dates),
        "data_quality": _data_quality(items, days),
    }


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
    """Per-business-day series for line/area charts."""
    covers_by_date = {d["business_date"]: (d["net_covers"] or 0) for d in days}
    by_date = defaultdict(lambda: {"total": 0.0, "alcohol": 0.0, "na": 0.0})
    for i in items:
        d = i["business_date"]
        if not d:
            continue
        by_date[d]["total"] += i["net_revenue"]
        if i["bev_type"] in ALCOHOL_TYPES:
            by_date[d]["alcohol"] += i["net_revenue"]
        elif i["bev_type"] in NA_TYPES:
            by_date[d]["na"] += i["net_revenue"]
    out = []
    for d in sorted(by_date):
        bev = by_date[d]["alcohol"] + by_date[d]["na"]
        covers = covers_by_date.get(d, 0)
        out.append({
            "date": d,
            "total_revenue": round(by_date[d]["total"], 2),
            "beverage_revenue": round(bev, 2),
            "alcohol_revenue": round(by_date[d]["alcohol"], 2),
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


def _movers(items: list[dict], dates: list[str]) -> dict:
    """
    Compare the two most recent business days (or halves of a longer range) to
    surface beverage items gaining or losing momentum. Needs 2+ days of data.
    """
    if len(dates) < 2:
        return {"enough_data": False, "gainers": [], "decliners": [], "note":
                "Upload at least two days of reports to see momentum."}

    mid = len(dates) // 2
    if len(dates) == 2:
        prev_dates, cur_dates = {dates[0]}, {dates[1]}
    else:
        prev_dates, cur_dates = set(dates[:mid]), set(dates[mid:])

    def bucket(which: set[str]) -> dict:
        agg = defaultdict(lambda: {"net_revenue": 0.0, "bev_type": None})
        for i in items:
            if i["bev_type"] not in BEVERAGE_TYPES or i["business_date"] not in which:
                continue
            key = (i["name"] or "?").strip()
            agg[key]["net_revenue"] += i["net_revenue"]
            agg[key]["bev_type"] = i["bev_type"]
        return agg

    prev, cur = bucket(prev_dates), bucket(cur_dates)
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
        "prev_period": sorted(prev_dates),
        "current_period": sorted(cur_dates),
        "gainers": gainers,
        "decliners": decliners,
    }


def _data_quality(items: list[dict], days: list[dict]) -> dict:
    dates_with_items = {i["business_date"] for i in items if i["business_date"]}
    dates_with_covers = {d["business_date"] for d in days if (d["net_covers"] or 0) > 0}
    return {
        "item_days": len(dates_with_items),
        "cover_days": len(dates_with_covers),
        "days_missing_covers": sorted(dates_with_items - dates_with_covers),
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
