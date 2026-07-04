"""
Flask app for the Beverage Sales Dashboard.

Routes
  GET  /                          – dashboard UI
  GET  /api/outlets               – outlet catalog (which have data)
  GET  /api/reports               – uploaded reports (optional ?outlet=)
  POST /api/upload                – upload a PMIX or Sales-Summary report
  DELETE /api/reports/<id>        – delete a report + its rows
  GET  /api/dashboard             – dashboard payload (?outlet=&start=&end=)
  GET  /api/insights              – AI opportunity briefing (?outlet=&start=&end=)
"""

import os
import traceback
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import analytics
import database as db
import ingest

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".pdf", ".csv", ".txt"}

app = Flask(__name__, template_folder="templates")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


def _err(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/api/outlets")
def api_outlets():
    return jsonify({
        "outlets": db.outlet_catalog(),
        "known": db.KNOWN_OUTLETS,
    })


@app.route("/api/reports")
def api_reports():
    outlet = request.args.get("outlet") or None
    return jsonify(db.list_reports(outlet))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return _err("No file part in request.")
    file = request.files["file"]
    if not file.filename:
        return _err("No file selected.")
    if Path(file.filename).suffix.lower() not in ALLOWED_EXT:
        return _err(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}")

    outlet_hint = (request.form.get("outlet") or "").strip() or None
    replace = request.form.get("replace", "true").lower() != "false"

    safe_name = secure_filename(file.filename)
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    file.save(tmp_path)

    try:
        # Peek to detect type/outlet/date so we can de-dupe before inserting.
        text = ingest._extract_text(tmp_path)
        report_type = ingest._detect_report_type(text)

        result = ingest.ingest_file(tmp_path, outlet_hint=outlet_hint)

        # Replace any prior report for the same outlet/date/type so re-uploading
        # a corrected export doesn't double-count.
        if replace:
            _dedupe(result, keep_id=result.get("report_id"))

        result["filename"] = safe_name
        result["message"] = _upload_message(result)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Ingestion failed ({type(exc).__name__}): {exc}", 500)
    finally:
        tmp_path.unlink(missing_ok=True)


def _dedupe(result: dict, keep_id: int | None) -> None:
    outlet = result.get("outlet")
    date = result.get("business_date")
    rtype = result.get("report_type")
    if not (outlet and date and rtype):
        return
    for rep in db.list_reports(outlet):
        if (rep["business_date"] == date and rep["report_type"] == rtype
                and rep["id"] != keep_id):
            db.delete_report(rep["id"])


def _upload_message(result: dict) -> str:
    if result["report_type"] == "pmix":
        return (f"Imported {result['items_ingested']} line items for {result['outlet']} "
                f"on {result['business_date']} (${result['net_revenue']:,.0f} net).")
    return (f"Imported daily summary for {result['outlet']} on {result['business_date']} "
            f"— {result['net_covers']} covers, ${result['avg_check']:,.2f} avg check.")


@app.route("/api/reports/<int:report_id>", methods=["DELETE"])
def api_delete_report(report_id):
    db.delete_report(report_id)
    return jsonify({"message": "Report deleted."})


@app.route("/api/dashboard")
def api_dashboard():
    outlet = request.args.get("outlet")
    if not outlet:
        return _err("'outlet' is required.")
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    return jsonify(analytics.dashboard(outlet, start, end))


@app.route("/api/insights")
def api_insights():
    outlet = request.args.get("outlet")
    if not outlet:
        return _err("'outlet' is required.")
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    try:
        return jsonify(analytics.ai_insights(outlet, start, end))
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Insight generation failed ({type(exc).__name__}): {exc}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
