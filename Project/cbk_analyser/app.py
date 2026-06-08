"""
app.py — Flask backend for CBK Report Analyser
"""

import io
import uuid
from flask import Flask, request, jsonify, render_template
from analyser import analyse_pdf, semantic_search, extract_text_from_pdf

app = Flask(__name__)
app.secret_key = "cbk-nlp-2024-secret"

# All uploaded reports: report_id -> { text, sent_cache }
_store = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    try:
        pdf_bytes = pdf_file.read()

        # Run full analysis (includes vectorisation)
        results = analyse_pdf(io.BytesIO(pdf_bytes))

        # Pull out the sentence cache and store it server-side
        sent_cache = results.pop("_sent_cache", None)

        report_id = uuid.uuid4().hex[:10]
        _store[report_id] = {
            "text":       extract_text_from_pdf(io.BytesIO(pdf_bytes))[0],
            "sent_cache": sent_cache,
        }

        results["filename"]  = pdf_file.filename
        results["report_id"] = report_id

        return jsonify({"success": True, "data": results})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/search", methods=["POST"])
def search():
    body      = request.get_json(silent=True) or {}
    query     = body.get("query", "").strip()
    report_id = body.get("report_id", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    entry = _store.get(report_id)
    if not entry:
        # Fall back to most recently stored report
        if not _store:
            return jsonify({"error": "No report loaded. Please upload a PDF first."}), 400
        entry = list(_store.values())[-1]

    try:
        results = semantic_search(
            query, entry["text"], top_n=3, cached=entry["sent_cache"]
        )
        return jsonify({"success": True, "results": results, "query": query})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
