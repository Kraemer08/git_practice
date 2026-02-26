"""
Flask web application for the Wine Advisor.

Routes:
  GET  /                    – main UI
  GET  /api/stats           – database stats
  GET  /api/documents       – list uploaded documents
  POST /api/upload          – upload a catalogue
  DELETE /api/documents/<id> – delete a document + its wines
  GET  /api/wines           – search/browse wines
  GET  /api/wines/<id>      – get one wine
  GET  /api/concepts        – list concepts
  POST /api/concepts        – create/update a concept
  POST /api/chat            – streaming chat endpoint (SSE)
"""

import json
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import database as db
from extractor import upload_and_extract

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".pdf", ".csv", ".txt"}

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ── Helpers ────────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def _err(msg: str, status: int = 400) -> Response:
    return jsonify({"error": msg}), status


# ── Static / UI ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


# ── Database stats ─────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_wine_stats())


# ── Documents ──────────────────────────────────────────────────────────────────

@app.route("/api/documents")
def api_list_documents():
    return jsonify(db.list_documents())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return _err("No file part in request.")
    file = request.files["file"]
    if not file.filename:
        return _err("No file selected.")
    if not _allowed(file.filename):
        return _err(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")

    supplier = request.form.get("supplier", "Unknown Supplier").strip() or "Unknown Supplier"

    # Save locally first
    safe_name = secure_filename(file.filename)
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    file.save(tmp_path)

    try:
        count, file_id = upload_and_extract(tmp_path, supplier)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        tmp_path.unlink(missing_ok=True)
        return _err(f"Extraction failed ({type(exc).__name__}): {exc}", 500)
    finally:
        tmp_path.unlink(missing_ok=True)

    return jsonify({
        "message": f"Extracted {count} wines from '{safe_name}'.",
        "wine_count": count,
        "file_id": file_id,
    })


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def api_delete_document(doc_id):
    docs = db.list_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        return _err("Document not found.", 404)

    db.delete_document(doc_id)
    return jsonify({"message": "Document and its wines deleted."})


# ── Wines ───────────────────────────────────────────────────────────────────────

@app.route("/api/wines")
def api_search_wines():
    query = request.args.get("q", "")
    filters = {}
    for key in ("style", "country"):
        if v := request.args.get(key):
            filters[key] = v
    for key in ("min_price", "max_price"):
        if v := request.args.get(key):
            try:
                filters[key] = float(v)
            except ValueError:
                pass
    limit = min(int(request.args.get("limit", 50)), 200)
    wines = db.search_wines(query=query, filters=filters, limit=limit)
    return jsonify(wines)


@app.route("/api/wines/<int:wine_id>")
def api_get_wine(wine_id):
    wine = db.get_wine(wine_id)
    if not wine:
        return _err("Wine not found.", 404)
    return jsonify(wine)


# ── Concepts ────────────────────────────────────────────────────────────────────

@app.route("/api/concepts")
def api_list_concepts():
    return jsonify(db.list_concepts())


@app.route("/api/concepts", methods=["POST"])
def api_save_concept():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return _err("'name' is required.")
    db.upsert_concept(
        name=name,
        cuisine_type=data.get("cuisine_type", ""),
        price_tier=data.get("price_tier", ""),
        guest_profile=data.get("guest_profile", ""),
        wine_style_notes=data.get("wine_style_notes", ""),
        additional_notes=data.get("additional_notes", ""),
    )
    return jsonify({"message": f"Concept '{name}' saved."})


# ── Chat (SSE streaming) ────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not messages:
        return _err("'messages' array is required.")

    # Validate last message is from user
    if not messages or messages[-1].get("role") != "user":
        return _err("Last message must have role 'user'.")

    def generate():
        from advisor import chat_stream
        try:
            for chunk in chat_stream(messages):
                # SSE format
                payload = json.dumps({"text": chunk})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
