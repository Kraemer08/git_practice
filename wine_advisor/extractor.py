"""
Document processor: uploads wine catalogues / price books to the Files API,
then asks Claude to extract structured wine data.

Supports PDF files (passed directly) and plain-text / CSV content.
"""

import json
import re
from pathlib import Path

import anthropic

from database import insert_document, insert_wines, update_document_wine_count

client = anthropic.Anthropic()

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
    Upload a file to the Files API, extract wine data with Claude, and
    persist to the database.

    Returns (number_of_wines_inserted, file_id).
    """
    file_path = Path(file_path)
    filename = file_path.name
    mime = _guess_mime(filename)

    # 1 ── Upload to Files API
    with file_path.open("rb") as fh:
        uploaded = client.beta.files.upload(
            file=(filename, fh, mime),
        )
    file_id = uploaded.id

    # 2 ── Create a document record
    doc_id = insert_document(filename, file_id, supplier)

    # 3 ── Ask Claude to extract wine data (streaming for large files)
    content_block: list = [
        {"type": "text", "text": USER_PROMPT.format(supplier=supplier)},
    ]

    if mime in ("application/pdf",):
        content_block.append({
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        })
    else:
        # Plain text / CSV: read and embed inline
        text_content = file_path.read_text(errors="replace")
        content_block[0]["text"] += f"\n\n---\n{text_content}"

    with client.beta.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": content_block}],
        betas=["files-api-2025-04-14"],
    ) as stream:
        full_text = stream.get_final_message().content[-1].text

    # 4 ── Parse JSON from response
    wines = _parse_wine_json(full_text)

    # 5 ── Persist wines
    inserted = insert_wines(doc_id, wines)
    update_document_wine_count(doc_id, inserted)

    return inserted, file_id


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
