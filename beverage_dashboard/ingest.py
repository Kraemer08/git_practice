"""
Ingestion for POS sales reports.

Handles two InfoGenesis exports:
  * "Product Mix-Revenue"          -> item-level sales lines  (report_type 'pmix')
  * "Sales Summary with Revenue"   -> covers / checks / totals (report_type 'sales_summary')

The pipeline is: pull the text out of the PDF (pypdf), detect which report it
is, then hand the text to Claude with a forced tool call so it returns clean,
structured JSON. For Product-Mix reports Claude also *classifies* every line
into a beverage sub-type (cocktail, spirit, wine by the glass, wine bottle,
beer, N/A beverage, coffee/tea, food, other) — the nuance that raw category
codes can't give us (e.g. a "Liquor" line is a signature cocktail vs. a spirit
pour; coffee and soda live under "Food").

CSV / TXT files are supported too: their contents are treated as report text.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import anthropic

import database as db

MODEL = os.environ.get("BEV_DASHBOARD_MODEL", "claude-sonnet-5")

# Extract items in small chunks so no rows get dropped or summarised. A long
# report (the Food section alone can be 100+ lines) is unreliable to transcribe
# in one pass, so we keep each request to a handful of rows.
_CHUNK_CHARS = 6000


def _make_client() -> anthropic.Anthropic:
    """
    Build an Anthropic client.
      * Prefers ANTHROPIC_API_KEY (standard local setup).
      * Falls back to the Claude Code session token when running on the web.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()

    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        token = Path(token_file).read_text().strip()
        return anthropic.Anthropic(auth_token=token)

    raise RuntimeError(
        "No Anthropic credentials found. Set ANTHROPIC_API_KEY before starting "
        "the app."
    )


client = _make_client()

# ── Shared classification guidance ───────────────────────────────────────────

_BEV_TYPES_STR = ", ".join(db.BEV_TYPES)

_CLASSIFY_RULES = """Classify every line item into exactly one `bev_type`:

- cocktail   : mixed / signature / craft cocktails. In this data these are
               usually named creations under the Liquor category
               (e.g. "ORLA- Zeus", "ORLA- Bloody Mary", "Cash & Prizes").
- spirit     : a single spirit sold as a pour — brand names under Liquor
               (e.g. "Grey Goose", "Tito's", "Hendrick's Gin",
               "Don Julio 1942", "Maker's 46 Bourbon", "Suntory Toki").
- wine_btg   : wine by the glass. Names starting with "5oz-", "9oz-", "GL-"
               or otherwise a glass pour under the Wine category.
- wine_bottle: a specific wine sold by the bottle (Wine category, not a pour).
- beer       : beer & hard seltzer (Beer category).
- na_beverage: non-alcoholic drinks — soda, juice, water, lemonade, Shirley
               Temple, milk, mocktails ("ORLA- Mocktail"), N/A beer. Many of
               these appear under the Food category.
- coffee_tea : coffee, espresso, americano, cappuccino, latte, hot/iced tea.
- food       : any food dish.
- other      : non-revenue markers and modifiers — course headers
               ("***Course 1***", "***SIDES***"), "AVELAR", "Hot Share Plates",
               supplements, and lines that are organisational rather than a
               sellable product.

Use the item name AND its revenue category together. When a beverage looks
like it could be either a branded spirit or a house cocktail, prefer `cocktail`
only if the name is a creative/drink name rather than a spirit brand."""

# ── Tool schemas (forced structured output) ──────────────────────────────────

_PMIX_TOOL = {
    "name": "record_product_mix",
    "description": "Record every line item from a Product Mix-Revenue report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "outlet": {
                "type": "string",
                "description": (
                    "The restaurant/outlet, inferred from the profit center "
                    "names (e.g. 'Orla' and 'Orla Bar' -> outlet 'Orla'). "
                    "Prefer one of: " + ", ".join(db.KNOWN_OUTLETS)
                ),
            },
            "store": {"type": "string"},
            "period_start": {"type": "string", "description": "ISO 8601 datetime"},
            "period_end": {"type": "string", "description": "ISO 8601 datetime"},
            "business_date": {
                "type": "string",
                "description": "The business day the period belongs to, YYYY-MM-DD (use the period start date).",
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "profit_center": {"type": "string"},
                        "pos_item_id": {"type": "string"},
                        "name": {"type": "string"},
                        "revenue_category": {
                            "type": "string",
                            "description": "Food, Beer, Wine, Liquor, or Miscellaneous",
                        },
                        "bev_type": {"type": "string", "enum": db.BEV_TYPES},
                        "items_sold": {"type": "number"},
                        "gross_revenue": {"type": "number"},
                        "discounts": {"type": "number"},
                        "net_revenue": {"type": "number"},
                        "avg_price": {"type": "number", "description": "Avg net revenue per item"},
                    },
                    "required": ["name", "revenue_category", "bev_type",
                                 "items_sold", "net_revenue"],
                },
            },
        },
        "required": ["outlet", "items"],
    },
}

_SUMMARY_TOOL = {
    "name": "record_sales_summary",
    "description": "Record the totals from a Sales Summary with Revenue Detail report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "outlet": {"type": "string", "description": "Prefer one of: " + ", ".join(db.KNOWN_OUTLETS)},
            "store": {"type": "string"},
            "period_start": {"type": "string"},
            "period_end": {"type": "string"},
            "business_date": {"type": "string", "description": "YYYY-MM-DD"},
            "net_covers": {"type": "number", "description": "Total net covers (guests) across all profit centers"},
            "net_checks": {"type": "number"},
            "avg_check": {"type": "number", "description": "Blended average check across the outlet"},
            "gross_revenue": {"type": "number"},
            "net_revenue": {"type": "number"},
            "discounts": {"type": "number"},
            "category_revenue": {
                "type": "array",
                "description": "Net revenue by revenue category for the outlet total.",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "gross_revenue": {"type": "number"},
                        "net_revenue": {"type": "number"},
                    },
                    "required": ["category", "net_revenue"],
                },
            },
        },
        "required": ["outlet"],
    },
}


# ── Public entry point ───────────────────────────────────────────────────────

def ingest_file(file_path: str | Path, outlet_hint: str | None = None) -> dict:
    """
    Parse a report file and persist it. Returns a summary dict describing what
    was ingested. `outlet_hint` overrides the outlet if the caller knows it.
    """
    file_path = Path(file_path)
    text = _extract_text(file_path)
    if not text.strip():
        raise ValueError("No readable text found in the file.")

    report_type = _detect_report_type(text)

    if report_type == "sales_summary":
        return _ingest_summary(file_path.name, text, outlet_hint)
    return _ingest_pmix(file_path.name, text, outlet_hint)


# ── Product-Mix ingestion ────────────────────────────────────────────────────

def _ingest_pmix(filename: str, text: str, outlet_hint: str | None) -> dict:
    chunks = _chunk_text(text)

    # Each chunk is an independent Claude call, so run them concurrently — a
    # 35-page monthly report is ~17 chunks and would take many minutes serially.
    def extract(i: int) -> tuple[int, dict]:
        return i, _call_tool(
            system=_pmix_system(),
            user=_pmix_user(chunks[i], i, len(chunks)),
            tool=_PMIX_TOOL,
        )

    results: list[dict] = [{} for _ in chunks]
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as pool:
        for i, result in pool.map(extract, range(len(chunks))):
            results[i] = result

    header = results[0] if results else {}
    all_items: list[dict] = []
    for result in results:
        all_items.extend(result.get("items", []))

    outlet = outlet_hint or header.get("outlet") or "Orla"
    start_date, end_date, num_days = _period_span(
        header.get("period_start"), header.get("period_end"),
        header.get("business_date"))
    gross = sum(db._num(it.get("gross_revenue")) for it in all_items)
    net = sum(db._num(it.get("net_revenue")) for it in all_items)
    disc = sum(db._num(it.get("discounts")) for it in all_items)

    report_id = db.insert_report(
        filename=filename, report_type="pmix", outlet=outlet,
        store=header.get("store"), period_start=header.get("period_start"),
        period_end=header.get("period_end"), business_date=start_date,
        period_end_date=end_date, num_days=num_days,
        gross_revenue=gross, net_revenue=net, discounts=disc,
        item_count=len(all_items),
        raw={"header": {k: v for k, v in header.items() if k != "items"}},
    )
    inserted = db.insert_sales_items(report_id, outlet, start_date, all_items,
                                     period_end_date=end_date, num_days=num_days)

    return {
        "report_type": "pmix",
        "outlet": outlet,
        "business_date": start_date,
        "period_end_date": end_date,
        "num_days": num_days,
        "items_ingested": inserted,
        "net_revenue": round(net, 2),
        "report_id": report_id,
    }


# ── Sales-Summary ingestion ──────────────────────────────────────────────────

def _ingest_summary(filename: str, text: str, outlet_hint: str | None) -> dict:
    result = _call_tool(
        system=_summary_system(),
        user=f"Extract the summary totals from this Sales Summary report.\n\n---\n{text}",
        tool=_SUMMARY_TOOL,
    )
    outlet = outlet_hint or result.get("outlet") or "Orla"
    start_date, end_date, num_days = _period_span(
        result.get("period_start"), result.get("period_end"),
        result.get("business_date"))

    # Compute the blended average check deterministically (gross / checks)
    # rather than trusting the model to blend per-profit-center figures.
    net_checks = int(db._num(result.get("net_checks")))
    gross_rev = db._num(result.get("gross_revenue"))
    avg_check = round(gross_rev / net_checks, 2) if net_checks else db._num(result.get("avg_check"))
    result["avg_check"] = avg_check

    report_id = db.insert_report(
        filename=filename, report_type="sales_summary", outlet=outlet,
        store=result.get("store"), period_start=result.get("period_start"),
        period_end=result.get("period_end"), business_date=start_date,
        period_end_date=end_date, num_days=num_days,
        gross_revenue=db._num(result.get("gross_revenue")),
        net_revenue=db._num(result.get("net_revenue")),
        discounts=db._num(result.get("discounts")),
        item_count=0, raw=result,
    )

    if start_date:
        db.upsert_outlet_day(
            outlet=outlet, business_date=start_date,
            period_end_date=end_date, num_days=num_days,
            net_covers=int(db._num(result.get("net_covers"))) or None,
            net_checks=net_checks or None,
            avg_check=avg_check or None,
            gross_revenue=db._num(result.get("gross_revenue")),
            net_revenue=db._num(result.get("net_revenue")),
            source_report_id=report_id,
        )

    return {
        "report_type": "sales_summary",
        "outlet": outlet,
        "business_date": start_date,
        "period_end_date": end_date,
        "num_days": num_days,
        "net_covers": int(db._num(result.get("net_covers"))),
        "avg_check": avg_check,
        "report_id": report_id,
    }


# ── Claude call ──────────────────────────────────────────────────────────────

def _call_tool(*, system: str, user: str, tool: dict) -> dict:
    """Make one forced-tool-use call and return the tool input dict."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool["name"]:
            return block.input
    raise ValueError(
        f"Claude did not return the expected '{tool['name']}' tool call "
        f"(stop_reason={resp.stop_reason})."
    )


def _pmix_system() -> str:
    return (
        "You transcribe POS Product Mix-Revenue report tables into structured "
        "data. This is an exhaustive data-entry task: record EVERY product row "
        "in the text — every food dish, every drink, every side — with no "
        "exceptions, no summarising, and no merging. It is critical that the "
        "count and dollar totals of your output match the report exactly, so "
        "never skip rows even when a section (e.g. Food) is long and repetitive.\n\n"
        "Each product row has: item ID, name, items consumed, items sold, "
        "items-sold %, gross revenue, discounts, net revenue, net-revenue %, and "
        "avg net revenue. Strip $ and commas from numbers. Track the current "
        "'Profit Center:' and 'Revenue Category:' headers and stamp each item "
        "with them. Skip ONLY the roll-up rows labelled 'Revenue Category "
        "Total', 'Profit Center Total' and 'Grand Total'.\n\n"
        "After transcribing a row, tag it with a bev_type.\n\n" + _CLASSIFY_RULES
    )


def _pmix_user(chunk: str, idx: int, total: int) -> str:
    part = f" (part {idx + 1} of {total})" if total > 1 else ""
    return (
        f"Transcribe EVERY product row in this Product Mix-Revenue report{part} "
        "into the record_product_mix tool. Capture profit_center, pos_item_id, "
        "name, revenue_category, items_sold, gross_revenue, discounts, "
        "net_revenue and avg_price for each row, and assign a bev_type. Include "
        "rows with $0.00 revenue. Do not skip any food items. Ignore only the "
        "'Revenue Category Total', 'Profit Center Total' and 'Grand Total' "
        "summary lines.\n\n---\n" + chunk
    )


def _summary_system() -> str:
    return (
        "You are a hospitality data analyst. Extract the outlet-level totals "
        "from an InfoGenesis 'Sales Summary with Revenue Detail' report: covers, "
        "checks, average check, revenue by category, and overall gross/net "
        "revenue and discounts. Sum across all profit centers for the outlet "
        "totals. Strip currency symbols and commas. Use the record_sales_summary tool."
    )


# ── Text helpers ─────────────────────────────────────────────────────────────

def _extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # CSV / TXT / anything else readable as text
    return file_path.read_text(errors="replace")


def _detect_report_type(text: str) -> str:
    low = text.lower()
    if "sales summary" in low or ("gross receipts" in low and "tenders" in low):
        return "sales_summary"
    if "product mix" in low or "items sold" in low:
        return "pmix"
    # Default: treat unknown tabular files as product mix (item-level).
    return "pmix"


def _chunk_text(text: str) -> list[str]:
    """Split by page-break newlines into <= _CHUNK_CHARS chunks."""
    if len(text) <= _CHUNK_CHARS:
        return [text]
    lines = text.split("\n")
    chunks, cur, size = [], [], 0
    for line in lines:
        if size + len(line) > _CHUNK_CHARS and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _norm_date(val: str | None) -> str | None:
    """Normalise a date string to YYYY-MM-DD."""
    if not val:
        return None
    val = val.strip()
    # Already ISO-ish
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", val)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # US M/D/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", val)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def _to_date(val: str | None) -> date | None:
    s = _norm_date(val)
    if not s:
        return None
    y, m, d = (int(x) for x in s.split("-"))
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _period_span(period_start: str | None, period_end: str | None,
                 business_date: str | None) -> tuple[str | None, str | None, int]:
    """
    Work out the reporting period from the report's "Business Period Starting …
    and Ending …" line.

    InfoGenesis periods run 3:00 AM → next-day 2:59 AM, so a report ending
    "7/1 2:59 AM" actually covers business days through 6/30. Returns
    (first_business_day, last_business_day, num_days). A single-day report spans
    one day; a full month spans ~30.
    """
    start = _to_date(period_start) or _to_date(business_date)
    end = _to_date(period_end)
    if start and end and end > start:
        num_days = (end - start).days           # 7/1 - 6/1 = 30
        last_day = start + timedelta(days=num_days - 1)
        return start.isoformat(), last_day.isoformat(), num_days
    if start:
        return start.isoformat(), start.isoformat(), 1
    return None, None, 1
