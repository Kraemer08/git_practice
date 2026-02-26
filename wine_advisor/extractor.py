"""
Document processor: sends wine catalogues / price books to Claude and
extracts structured wine data.

PDFs are sent as inline base64 documents; plain-text and CSV are
embedded directly in the prompt.
"""

import base64
import io
import json
import os
import re
from pathlib import Path

import anthropic
from pypdf import PdfReader, PdfWriter

from database import delete_document, insert_document, insert_wines, update_document_wine_count


def _make_client() -> anthropic.Anthropic:
    """
    Build an Anthropic client.
    - Prefers ANTHROPIC_API_KEY (standard setup, e.g. running on your Mac).
    - Falls back to the Claude Code session token when running inside the
      Claude Code on-web environment.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()

    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        token = Path(token_file).read_text().strip()
        return anthropic.Anthropic(auth_token=token)

    raise RuntimeError(
        "No Anthropic credentials found. Set the ANTHROPIC_API_KEY environment "
        "variable before starting the app."
    )


client = _make_client()

EXTRACTION_SYSTEM = """You are a specialist wine data extractor working for a luxury resort buyer.
Your job is to parse wine trade documents—price books, catalogues, and distributor lists—and
return every wine product as a clean JSON array.

For each wine include these fields (use null when information is absent):
  name           – wine label/product name
  producer       – winery or producer name
  appellation    – specific AOC / DOC / AVA / GI
  region         – broader region (e.g. "Burgundy", "Napa Valley")
  country        – country of origin
  vintage        – 4-digit year string, or "NV"
  grape_varieties – array of grape names (["Chardonnay"] or ["Cabernet Sauvignon","Merlot"])
  style          – one of: red, white, rosé, sparkling, fortified, dessert, orange
  price          – numeric price (strip symbols / commas)
  currency       – ISO code, default "USD"
  unit           – "bottle", "case/12", "case/6", "magnum", etc.
  description    – tasting notes or marketing copy (may be null)
  importer       – importer / distributor name if visible
  alcohol        – e.g. "13.5%"
  score          – critic score if listed, e.g. "95 WS"

Rules:
- Return ONLY a valid JSON array—no markdown fences, no commentary.
- If you find no wines at all, return [].
- Each wine is a separate object; do not combine products."""

USER_PROMPT = (
    "Extract every wine product from this document and return a JSON array "
    "following the schema in the system prompt. Document supplier: {supplier}."
)


def upload_and_extract(file_path: str | Path, supplier: str) -> tuple[int, str]:
    """
    Send a file to Claude, extract wine data, and persist to the database.

    PDFs are base64-encoded and sent inline (no Files API required).
    Returns (number_of_wines_inserted, filename).
    """
    file_path = Path(file_path)
    filename = file_path.name
    mime = _guess_mime(filename)

    # 1 ── Create a document record
    doc_id = insert_document(filename, filename, supplier)

    try:
        # 2 ── Build the content block
        content_block: list = [
            {"type": "text", "text": USER_PROMPT.format(supplier=supplier)},
        ]

        if mime == "application/pdf":
            wines = _extract_pdf_wines(file_path, supplier)
        else:
            # Plain text / CSV: embed inline
            text_content = file_path.read_text(errors="replace")
            content_block[0]["text"] += f"\n\n---\n{text_content}"

            # 3 ── Ask Claude to extract wine data (streaming for large files)
            wines = _call_claude(content_block)

        # 4 ── (wines already parsed)

        # 5 ── Persist wines
        inserted = insert_wines(doc_id, wines)
        update_document_wine_count(doc_id, inserted)
    except Exception:
        # Roll back the document record so no orphaned 0-wine entries remain
        delete_document(doc_id)
        raise

    return inserted, filename


def _pdf_chunk_b64(file_path: Path, start_page: int, end_page: int) -> str:
    """Return a base64-encoded PDF containing pages [start_page, end_page)."""
    reader = PdfReader(str(file_path))
    writer = PdfWriter()
    for i in range(start_page, min(end_page, len(reader.pages))):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_claude(content_block: list) -> list[dict]:
    """Send one content block to Claude and return parsed wines."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": content_block}],
    )
    text_block = next(
        (b for b in response.content if b.type == "text"),
        None,
    )
    if text_block is None:
        raise ValueError(
            f"Claude returned no text block. Stop reason: {response.stop_reason}. "
            f"Block types present: {[b.type for b in response.content]}"
        )
    return _parse_wine_json(text_block.text)


# Pages per chunk — keeps each request well under the 200k-token limit.
# A 200-page catalogue at ~1000 tokens/page fits comfortably in 50-page chunks.
_CHUNK_PAGES = 50


def _extract_pdf_wines(file_path: Path, supplier: str) -> list[dict]:
    """
    Split a PDF into page chunks, extract wines from each, and merge results.
    Falls back to a single request for small PDFs.
    """
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    all_wines: list[dict] = []

    for start in range(0, total_pages, _CHUNK_PAGES):
        end = min(start + _CHUNK_PAGES, total_pages)
        chunk_b64 = _pdf_chunk_b64(file_path, start, end)
        content_block = [
            {
                "type": "text",
                "text": (
                    USER_PROMPT.format(supplier=supplier)
                    + f" (pages {start + 1}–{end} of {total_pages})"
                ),
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": chunk_b64,
                },
            },
        ]
        all_wines.extend(_call_claude(content_block))

    return all_wines


def delete_file_from_api(file_id: str) -> None:
    try:
        client.beta.files.delete(file_id)
    except Exception:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".pdf":  "application/pdf",
        ".csv":  "text/plain",
        ".txt":  "text/plain",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls":  "application/vnd.ms-excel",
    }.get(ext, "application/octet-stream")


def _parse_wine_json(raw: str) -> list[dict]:
    """Extract a JSON array from Claude's text response."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []
