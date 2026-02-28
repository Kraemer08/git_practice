"""
SQLite database layer for the Wine Advisor application.
Stores wine data extracted from catalogues, uploaded documents, and restaurant concepts.
"""

import sqlite3
import json
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "wine_advisor.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            file_id     TEXT    NOT NULL,
            supplier    TEXT,
            doc_type    TEXT    NOT NULL DEFAULT 'supplier',
            contact_name  TEXT,
            contact_email TEXT,
            wine_count  INTEGER DEFAULT 0,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS wines (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id     INTEGER REFERENCES documents(id),
            name            TEXT,
            producer        TEXT,
            appellation     TEXT,
            region          TEXT,
            country         TEXT,
            vintage         TEXT,
            grape_varieties TEXT,
            style           TEXT,
            price           REAL,
            currency        TEXT DEFAULT 'USD',
            unit            TEXT DEFAULT 'bottle',
            description     TEXT,
            importer        TEXT,
            alcohol         TEXT,
            score           TEXT,
            raw_json        TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS wines_fts USING fts5(
            name, producer, appellation, region, country,
            vintage, grape_varieties, style, description, importer,
            content=wines, content_rowid=id
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT UNIQUE NOT NULL,
            cuisine_type    TEXT,
            price_tier      TEXT,
            guest_profile   TEXT,
            wine_style_notes TEXT,
            additional_notes TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_profile (
            id              INTEGER PRIMARY KEY DEFAULT 1,
            name            TEXT,
            title           TEXT,
            company         TEXT,
            location        TEXT,
            email           TEXT,
            phone           TEXT,
            writing_samples TEXT,
            style_summary   TEXT,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Triggers to keep FTS index in sync
    c.executescript("""
        CREATE TRIGGER IF NOT EXISTS wines_ai AFTER INSERT ON wines BEGIN
            INSERT INTO wines_fts(rowid, name, producer, appellation, region,
                country, vintage, grape_varieties, style, description, importer)
            VALUES (new.id, new.name, new.producer, new.appellation, new.region,
                new.country, new.vintage, new.grape_varieties, new.style,
                new.description, new.importer);
        END;

        CREATE TRIGGER IF NOT EXISTS wines_ad AFTER DELETE ON wines BEGIN
            INSERT INTO wines_fts(wines_fts, rowid, name, producer, appellation,
                region, country, vintage, grape_varieties, style, description, importer)
            VALUES ('delete', old.id, old.name, old.producer, old.appellation,
                old.region, old.country, old.vintage, old.grape_varieties, old.style,
                old.description, old.importer);
        END;
    """)

    # Migrate existing databases that predate optional columns
    for migration in [
        "ALTER TABLE documents ADD COLUMN doc_type TEXT NOT NULL DEFAULT 'supplier'",
        "ALTER TABLE documents ADD COLUMN contact_name TEXT",
        "ALTER TABLE documents ADD COLUMN contact_email TEXT",
        "ALTER TABLE documents ADD COLUMN relationship_notes TEXT",
    ]:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()


# ── Documents ──────────────────────────────────────────────────────────────────

def insert_document(filename: str, file_id: str, supplier: str,
                    doc_type: str = "supplier") -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents (filename, file_id, supplier, doc_type) VALUES (?,?,?,?)",
        (filename, file_id, supplier, doc_type),
    )
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def update_document_wine_count(doc_id: int, count: int) -> None:
    conn = get_conn()
    conn.execute("UPDATE documents SET wine_count=? WHERE id=?", (count, doc_id))
    conn.commit()
    conn.close()


def list_documents() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_document_supplier(doc_id: int, supplier: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE documents SET supplier=? WHERE id=?", (supplier, doc_id))
    conn.commit()
    conn.close()


def update_document_contacts(doc_id: int, contact_name: str, contact_email: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE documents SET contact_name=?, contact_email=? WHERE id=?",
        (contact_name or None, contact_email or None, doc_id),
    )
    conn.commit()
    conn.close()


def delete_document(doc_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM wines WHERE document_id=?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()


# ── Wines ───────────────────────────────────────────────────────────────────────

def insert_wines(doc_id: int, wines: list[dict]) -> int:
    conn = get_conn()
    c = conn.cursor()
    inserted = 0
    for w in wines:
        try:
            grape_varieties = w.get("grape_varieties")
            if isinstance(grape_varieties, list):
                grape_varieties = ", ".join(grape_varieties)

            c.execute(
                """INSERT INTO wines
                   (document_id, name, producer, appellation, region, country,
                    vintage, grape_varieties, style, price, currency, unit,
                    description, importer, alcohol, score, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id,
                    w.get("name"),
                    w.get("producer"),
                    w.get("appellation"),
                    w.get("region"),
                    w.get("country"),
                    str(w.get("vintage", "")) if w.get("vintage") else None,
                    grape_varieties,
                    w.get("style"),
                    _to_float(w.get("price")),
                    w.get("currency", "USD"),
                    w.get("unit", "bottle"),
                    w.get("description"),
                    w.get("importer"),
                    w.get("alcohol"),
                    w.get("score"),
                    json.dumps(w),
                ),
            )
            inserted += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return inserted


def _to_float(val: Any) -> float | None:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def search_wines(query: str = "", filters: dict | None = None, limit: int = 50) -> list[dict]:
    filters = filters or {}
    conn = get_conn()

    # Build SQL-level filter clauses
    filter_clauses: list[str] = []
    filter_params: list = []
    if filters.get("style"):
        filter_clauses.append("LOWER(w.style) = LOWER(?)")
        filter_params.append(filters["style"])
    if filters.get("country"):
        filter_clauses.append("LOWER(w.country) = LOWER(?)")
        filter_params.append(filters["country"])
    if filters.get("max_price") is not None:
        filter_clauses.append("w.price IS NOT NULL AND w.price <= ?")
        filter_params.append(float(filters["max_price"]))
    if filters.get("min_price") is not None:
        filter_clauses.append("w.price IS NOT NULL AND w.price >= ?")
        filter_params.append(float(filters["min_price"]))
    if filters.get("doc_type"):
        filter_clauses.append("d.doc_type = ?")
        filter_params.append(filters["doc_type"])
    if filters.get("doc_id") is not None:
        filter_clauses.append("w.document_id = ?")
        filter_params.append(int(filters["doc_id"]))

    if query:
        extra = (" AND " + " AND ".join(filter_clauses)) if filter_clauses else ""
        rows = conn.execute(
            f"""SELECT w.* FROM wines w
               JOIN wines_fts f ON w.id = f.rowid
               JOIN documents d ON w.document_id = d.id
               WHERE wines_fts MATCH ?{extra}
               ORDER BY rank LIMIT ?""",
            (query, *filter_params, limit),
        ).fetchall()
    else:
        clauses_str = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""
        rows = conn.execute(
            f"""SELECT w.* FROM wines w
               JOIN documents d ON w.document_id = d.id
               {clauses_str}
               ORDER BY w.producer, w.name LIMIT ?""",
            (*filter_params, limit),
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def get_document_stats(doc_id: int) -> dict:
    """Return countries and styles present in a specific document."""
    conn = get_conn()
    stats = {
        "countries": [r[0] for r in conn.execute(
            "SELECT DISTINCT country FROM wines WHERE document_id=? AND country IS NOT NULL ORDER BY country",
            (doc_id,),
        ).fetchall()],
        "styles": [r[0] for r in conn.execute(
            "SELECT DISTINCT style FROM wines WHERE document_id=? AND style IS NOT NULL ORDER BY style",
            (doc_id,),
        ).fetchall()],
        "total": conn.execute(
            "SELECT COUNT(*) FROM wines WHERE document_id=?", (doc_id,)
        ).fetchone()[0],
    }
    conn.close()
    return stats


def get_wine(wine_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM wines WHERE id=?", (wine_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def count_wines() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM wines").fetchone()[0]
    conn.close()
    return n


def get_wine_stats() -> dict:
    conn = get_conn()
    stats = {
        "total": conn.execute("SELECT COUNT(*) FROM wines").fetchone()[0],
        "countries": [r[0] for r in conn.execute(
            "SELECT DISTINCT country FROM wines WHERE country IS NOT NULL ORDER BY country"
        ).fetchall()],
        "styles": [r[0] for r in conn.execute(
            "SELECT DISTINCT style FROM wines WHERE style IS NOT NULL ORDER BY style"
        ).fetchall()],
        "suppliers": [r[0] for r in conn.execute(
            "SELECT DISTINCT importer FROM wines WHERE importer IS NOT NULL ORDER BY importer"
        ).fetchall()],
        "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
    }
    conn.close()
    return stats


# ── Concepts ────────────────────────────────────────────────────────────────────

def upsert_concept(name: str, cuisine_type: str, price_tier: str,
                   guest_profile: str, wine_style_notes: str,
                   additional_notes: str) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO concepts (name, cuisine_type, price_tier, guest_profile,
               wine_style_notes, additional_notes)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(name) DO UPDATE SET
               cuisine_type=excluded.cuisine_type,
               price_tier=excluded.price_tier,
               guest_profile=excluded.guest_profile,
               wine_style_notes=excluded.wine_style_notes,
               additional_notes=excluded.additional_notes""",
        (name, cuisine_type, price_tier, guest_profile, wine_style_notes, additional_notes),
    )
    concept_id = c.lastrowid or conn.execute(
        "SELECT id FROM concepts WHERE name=?", (name,)
    ).fetchone()[0]
    conn.commit()
    conn.close()
    return concept_id


def list_concepts() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM concepts ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_concept(name: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM concepts WHERE name=?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── User Profile ─────────────────────────────────────────────────────────────

def get_user_profile() -> dict:
    conn = get_conn()
    row = conn.execute("SELECT * FROM user_profile WHERE id=1").fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "id": 1, "name": "", "title": "", "company": "", "location": "",
        "email": "", "phone": "", "writing_samples": "", "style_summary": "",
    }


def save_user_profile(data: dict) -> None:
    """Upsert the single user profile row (id=1). Only updates provided fields."""
    allowed = {"name", "title", "company", "location", "email", "phone",
               "writing_samples", "style_summary"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return
    conn = get_conn()
    existing = conn.execute("SELECT id FROM user_profile WHERE id=1").fetchone()
    if existing:
        set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=CURRENT_TIMESTAMP"
        vals = list(updates.values()) + [1]
        conn.execute(f"UPDATE user_profile SET {set_clause} WHERE id=?", vals)
    else:
        all_fields = ["name", "title", "company", "location", "email",
                      "phone", "writing_samples", "style_summary"]
        vals = tuple(updates.get(f, "") for f in all_fields)
        conn.execute(
            """INSERT INTO user_profile (id, name, title, company, location, email,
               phone, writing_samples, style_summary) VALUES (1,?,?,?,?,?,?,?,?)""",
            vals,
        )
    conn.commit()
    conn.close()


def update_document_relationship_notes(doc_id: int, notes: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE documents SET relationship_notes=? WHERE id=?",
        (notes or None, doc_id),
    )
    conn.commit()
    conn.close()


# Initialise on import
init_db()
