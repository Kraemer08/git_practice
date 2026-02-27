"""
Document processor: sends wine catalogues / price books to Claude and
extracts structured wine data.

Extraction uses Claude's tool_use feature (record_wines tool) for robust,
schema-validated output — eliminating fragile free-text JSON parsing.
PDFs are chunked into _CHUNK_PAGES-page sections to stay well within token limits.
"""

import base64
import io
import os
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
Your job is to parse wine trade documents—price books, catalogues, wine lists, and distributor
lists—and record every wine product using the record_wines tool.

Rules:
- Call record_wines exactly once with ALL wines found in the document section.
- Include every wine you can identify; do not skip any.
- Use null for any field not present in the source document.
- Do not combine separate products into one entry."""

USER_PROMPT = (
    "Extract every wine product from this document section and call the record_wines tool "
    "with all wines found. Document supplier/source: {supplier}."
)

# Using tool_use instead of free-text JSON output means:
#   - The SDK deserialises each tool call automatically; no parsing needed.
#   - Schema is enforced by the API; fields have the correct types.
#   - Far more reliable than extracting a JSON array from prose.
EXTRACTION_TOOLS = [
    {
        "name": "record_wines",
        "description": (
            "Record all wines found in the current section of the document. "
            "Use null for any field not present in the source material."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wines": {
                    "type": "array",
                    "description": "Every wine product found in this document section.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":            {"type": ["string", "null"], "description": "Wine label/product name"},
                            "producer":        {"type": ["string", "null"], "description": "Winery or producer name"},
                            "appellation":     {"type": ["string", "null"], "description": "AOC / DOC / AVA / GI"},
                            "region":          {"type": ["string", "null"], "description": "Broader region, e.g. Burgundy"},
                            "country":         {"type": ["string", "null"], "description": "Country of origin"},
                            "vintage":         {"type": ["string", "null"], "description": "4-digit year or NV"},
                            "grape_varieties": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "List of grape variety names",
                            },
                            "style": {
                                "type": ["string", "null"],
                                "description": "One of: red, white, rosé, sparkling, fortified, dessert, orange",
                            },
                            "price":       {"type": ["number", "null"], "description": "Numeric price (strip currency symbols)"},
                            "currency":    {"type": ["string", "null"], "description": "ISO currency code, default USD"},
                            "unit":        {"type": ["string", "null"], "description": "bottle / case/12 / case/6 / magnum etc."},
                            "description": {"type": ["string", "null"], "description": "Tasting notes or marketing copy"},
                            "importer":    {"type": ["string", "null"], "description": "Importer or distributor name"},
                            "alcohol":     {"type": ["string", "null"], "description": "ABV e.g. 13.5%"},
                            "score":       {"type": ["string", "null"], "description": "Critic score if listed, e.g. 95 WS"},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["wines"],
        },
    }
]


def upload_and_extract(file_path: str | Path, supplier: str,
                       doc_type: str = "supplier") -> tuple[int, str]:
    """
    Send a file to Claude, extract wine data, and persist to the database.

    PDFs are base64-encoded and sent inline (no Files API required).
    Returns (number_of_wines_inserted, filename).
    """
    file_path = Path(file_path)
    filename = file_path.name
    mime = _guess_mime(filename)

    doc_id = insert_document(filename, filename, supplier, doc_type)

    try:
        if mime == "application/pdf":
            wines = _extract_pdf_wines(file_path, supplier)
        else:
            text_content = file_path.read_text(errors="replace")
            content_block = [{
                "type": "text",
                "text": USER_PROMPT.format(supplier=supplier) + f"\n\n---\n{text_content}",
            }]
            wines = _call_claude(content_block)

        inserted = insert_wines(doc_id, wines)
        update_document_wine_count(doc_id, inserted)
    except Exception:
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
    """
    Send one content block to Claude and return extracted wines via tool_use.
    """
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=64000,
        system=EXTRACTION_SYSTEM,
        tools=EXTRACTION_TOOLS,
        tool_choice={"type": "any"},        # Claude MUST call at least one tool
        messages=[{"role": "user", "content": content_block}],
    ) as stream:
        final = stream.get_final_message()

    print(f"[extractor] stop_reason={final.stop_reason}  "
          f"output_tokens={final.usage.output_tokens}")

    wines: list[dict] = []
    for block in final.content:
        if block.type == "tool_use" and block.name == "record_wines":
            batch = block.input.get("wines") or []
            print(f"[extractor] tool_use block yielded {len(batch)} wines")
            wines.extend(batch)

    if not wines and final.stop_reason not in ("tool_use", "end_turn"):
        print(f"[extractor] WARNING: no wines returned. "
              f"stop_reason={final.stop_reason}  "
              f"content_types={[b.type for b in final.content]}")

    return wines


# PDF chunk size.  5 pages per API call keeps output tokens well within the
# 64k budget even for very dense index/listing pages (~50 wines/page →
# ~25k tokens/chunk worst case).  An 81-page wine list = ~17 API calls.
_CHUNK_PAGES = 5


def _extract_pdf_wines(file_path: Path, supplier: str) -> list[dict]:
    """Split a PDF into page chunks, extract wines from each, and merge."""
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    all_wines: list[dict] = []

    for start in range(0, total_pages, _CHUNK_PAGES):
        end = min(start + _CHUNK_PAGES, total_pages)
        print(f"[extractor] processing pages {start + 1}–{end} of {total_pages}")
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
        chunk_wines = _call_claude(content_block)
        print(f"[extractor] pages {start + 1}–{end}: found {len(chunk_wines)} wines")
        all_wines.extend(chunk_wines)

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
