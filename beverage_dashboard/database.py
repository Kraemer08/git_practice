"""
SQLite data layer for the Beverage Sales Dashboard.

Stores POS reports (InfoGenesis Product-Mix and Sales-Summary exports),
the item-level sales lines extracted from Product-Mix reports, and a
per-outlet / per-day summary (covers, checks, avg check) used to compute
attachment metrics like beverage revenue per cover.

Data volumes are small (a few hundred line items per outlet per day), so
aggregation for the dashboard is done in Python (see analytics.py); this
module is responsible for storage and simple filtered reads.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "beverage_dashboard.db"

# Outlets the beverage program spans. Orla has data today; the others are
# seeded so the dashboard can show them as targets before data is uploaded.
KNOWN_OUTLETS = ["Orla", "Azure", "IRD", "Banquets"]

# Beverage sub-types the ingestion classifier assigns to every line item.
BEV_TYPES = [
    "cocktail",       # signature / craft cocktails
    "spirit",         # single-spirit pours (vodka, gin, tequila, whiskey…)
    "wine_btg",       # wine by the glass (5oz / 9oz / GL pours)
    "wine_bottle",    # wine by the bottle
    "beer",           # beer & hard seltzer
    "na_beverage",    # soda, juice, water, mocktails
    "coffee_tea",     # coffee, espresso, tea
    "food",           # food items
    "other",          # modifiers, course markers, misc
]

# Which sub-types roll up into which high-level class.
_ALCOHOL_TYPES = {"cocktail", "spirit", "wine_btg", "wine_bottle", "beer"}
_NA_BEV_TYPES = {"na_beverage", "coffee_tea"}


def bev_class_for(bev_type: str) -> str:
    """Roll a beverage sub-type up to a high-level class."""
    if bev_type in _ALCOHOL_TYPES:
        return "alcohol"
    if bev_type in _NA_BEV_TYPES:
        return "na_beverage"
    if bev_type == "food":
        return "food"
    return "other"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filename       TEXT    NOT NULL,
            report_type    TEXT    NOT NULL,   -- 'pmix' | 'sales_summary'
            outlet         TEXT    NOT NULL,
            store          TEXT,
            period_start   TEXT,
            period_end     TEXT,
            business_date  TEXT,               -- YYYY-MM-DD the period belongs to
            gross_revenue  REAL DEFAULT 0,
            net_revenue    REAL DEFAULT 0,
            discounts      REAL DEFAULT 0,
            item_count     INTEGER DEFAULT 0,
            uploaded_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            raw_json       TEXT
        );

        CREATE TABLE IF NOT EXISTS sales_items (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id        INTEGER REFERENCES reports(id) ON DELETE CASCADE,
            outlet           TEXT,
            business_date    TEXT,
            profit_center    TEXT,
            pos_item_id      TEXT,
            name             TEXT,
            revenue_category TEXT,             -- Food / Beer / Wine / Liquor / Misc
            bev_class        TEXT,             -- alcohol / na_beverage / food / other
            bev_type         TEXT,             -- see BEV_TYPES
            items_sold       REAL DEFAULT 0,
            gross_revenue    REAL DEFAULT 0,
            discounts        REAL DEFAULT 0,
            net_revenue      REAL DEFAULT 0,
            avg_price        REAL DEFAULT 0,
            raw_json         TEXT
        );

        -- One row per outlet / business day, sourced from Sales-Summary
        -- reports. Holds the denominators (covers, checks) that item-level
        -- PMIX data can't provide on its own.
        CREATE TABLE IF NOT EXISTS outlet_days (
            outlet            TEXT,
            business_date     TEXT,
            net_covers        INTEGER,
            net_checks        INTEGER,
            avg_check         REAL,
            gross_revenue     REAL,
            net_revenue       REAL,
            source_report_id  INTEGER,
            PRIMARY KEY (outlet, business_date)
        );

        CREATE INDEX IF NOT EXISTS idx_items_outlet_date
            ON sales_items(outlet, business_date);
        CREATE INDEX IF NOT EXISTS idx_items_bev_type
            ON sales_items(bev_type);
        """
    )
    conn.commit()
    conn.close()


# ── Reports ──────────────────────────────────────────────────────────────────

def insert_report(
    *, filename: str, report_type: str, outlet: str, store: str | None,
    period_start: str | None, period_end: str | None, business_date: str | None,
    gross_revenue: float, net_revenue: float, discounts: float,
    item_count: int, raw: dict,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO reports
           (filename, report_type, outlet, store, period_start, period_end,
            business_date, gross_revenue, net_revenue, discounts, item_count, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (filename, report_type, outlet, store, period_start, period_end,
         business_date, gross_revenue, net_revenue, discounts, item_count,
         json.dumps(raw)),
    )
    report_id = cur.lastrowid
    conn.commit()
    conn.close()
    return report_id


def list_reports(outlet: str | None = None) -> list[dict]:
    conn = get_conn()
    if outlet:
        rows = conn.execute(
            "SELECT * FROM reports WHERE outlet=? ORDER BY business_date DESC, uploaded_at DESC",
            (outlet,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY business_date DESC, uploaded_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_report(report_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM sales_items WHERE report_id=?", (report_id,))
    conn.execute("DELETE FROM outlet_days WHERE source_report_id=?", (report_id,))
    conn.execute("DELETE FROM reports WHERE id=?", (report_id,))
    conn.commit()
    conn.close()


def report_exists(outlet: str, business_date: str, report_type: str) -> dict | None:
    """Return an existing report for this outlet/date/type, if any (dedupe helper)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM reports WHERE outlet=? AND business_date=? AND report_type=? LIMIT 1",
        (outlet, business_date, report_type),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Sales items ──────────────────────────────────────────────────────────────

def insert_sales_items(report_id: int, outlet: str, business_date: str | None,
                       items: list[dict]) -> int:
    conn = get_conn()
    inserted = 0
    for it in items:
        bev_type = it.get("bev_type") or "other"
        if bev_type not in BEV_TYPES:
            bev_type = "other"
        try:
            conn.execute(
                """INSERT INTO sales_items
                   (report_id, outlet, business_date, profit_center, pos_item_id,
                    name, revenue_category, bev_class, bev_type, items_sold,
                    gross_revenue, discounts, net_revenue, avg_price, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report_id, outlet, business_date,
                    it.get("profit_center"), str(it.get("pos_item_id") or ""),
                    it.get("name"), it.get("revenue_category"),
                    bev_class_for(bev_type), bev_type,
                    _num(it.get("items_sold")), _num(it.get("gross_revenue")),
                    _num(it.get("discounts")), _num(it.get("net_revenue")),
                    _num(it.get("avg_price")), json.dumps(it),
                ),
            )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return inserted


def get_items(outlet: str | None = None, start: str | None = None,
              end: str | None = None) -> list[dict]:
    """Filtered item rows for analytics. Dates are inclusive YYYY-MM-DD strings."""
    clauses, params = [], []
    if outlet:
        clauses.append("outlet = ?")
        params.append(outlet)
    if start:
        clauses.append("business_date >= ?")
        params.append(start)
    if end:
        clauses.append("business_date <= ?")
        params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = get_conn()
    rows = conn.execute(f"SELECT * FROM sales_items {where}", params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Outlet-days (covers / checks) ────────────────────────────────────────────

def upsert_outlet_day(*, outlet: str, business_date: str, net_covers: int | None,
                      net_checks: int | None, avg_check: float | None,
                      gross_revenue: float | None, net_revenue: float | None,
                      source_report_id: int) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO outlet_days
           (outlet, business_date, net_covers, net_checks, avg_check,
            gross_revenue, net_revenue, source_report_id)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(outlet, business_date) DO UPDATE SET
             net_covers=excluded.net_covers,
             net_checks=excluded.net_checks,
             avg_check=excluded.avg_check,
             gross_revenue=excluded.gross_revenue,
             net_revenue=excluded.net_revenue,
             source_report_id=excluded.source_report_id""",
        (outlet, business_date, net_covers, net_checks, avg_check,
         gross_revenue, net_revenue, source_report_id),
    )
    conn.commit()
    conn.close()


def get_outlet_days(outlet: str | None = None, start: str | None = None,
                    end: str | None = None) -> list[dict]:
    clauses, params = [], []
    if outlet:
        clauses.append("outlet = ?")
        params.append(outlet)
    if start:
        clauses.append("business_date >= ?")
        params.append(start)
    if end:
        clauses.append("business_date <= ?")
        params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = get_conn()
    rows = conn.execute(
        f"SELECT * FROM outlet_days {where} ORDER BY business_date", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Dashboard reference data ─────────────────────────────────────────────────

def outlet_catalog() -> list[dict]:
    """Every known outlet plus whether it has data and its date coverage."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT outlet,
                  COUNT(*)               AS report_count,
                  MIN(business_date)     AS first_date,
                  MAX(business_date)     AS last_date
           FROM reports GROUP BY outlet"""
    ).fetchall()
    conn.close()
    by_outlet = {r["outlet"]: dict(r) for r in rows}

    catalog = []
    for name in KNOWN_OUTLETS:
        info = by_outlet.get(name, {})
        catalog.append({
            "outlet": name,
            "has_data": bool(info.get("report_count")),
            "report_count": info.get("report_count", 0),
            "first_date": info.get("first_date"),
            "last_date": info.get("last_date"),
        })
    # Include any outlet found in data that isn't in the known list.
    for name, info in by_outlet.items():
        if name not in KNOWN_OUTLETS:
            catalog.append({
                "outlet": name,
                "has_data": True,
                "report_count": info.get("report_count", 0),
                "first_date": info.get("first_date"),
                "last_date": info.get("last_date"),
            })
    return catalog


def date_bounds(outlet: str | None = None) -> dict:
    conn = get_conn()
    q = "SELECT MIN(business_date) AS min_d, MAX(business_date) AS max_d FROM reports"
    params: list = []
    if outlet:
        q += " WHERE outlet=?"
        params.append(outlet)
    row = conn.execute(q, params).fetchone()
    conn.close()
    return {"min_date": row["min_d"], "max_date": row["max_d"]}


def _num(val: Any) -> float:
    try:
        if val is None:
            return 0.0
        return float(str(val).replace(",", "").replace("$", "").replace("%", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


# Initialise on import.
init_db()
