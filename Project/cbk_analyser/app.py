"""
app.py — Flask backend for CBK Mwananchi Economic Assistant
"""

import io
import uuid
from flask import Flask, request, jsonify, render_template
from analyser import (
    analyse_pdf, semantic_search, extract_text_from_pdf,
    get_predictive_suggestions, calculate_health_score,
    calculate_cost_of_living_impact, generate_chat_response,
    detect_language,
)

app = Flask(__name__)
app.secret_key = "cbk-mwananchi-2025-secret"

# report_id → { text, sent_cache, key_figures }
_store = {}


@app.route("/")
def index():
    return render_template("index.html")


# ── Upload & analyse ──────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    try:
        pdf_bytes = pdf_file.read()
        results   = analyse_pdf(io.BytesIO(pdf_bytes))

        sent_cache  = results.pop("_sent_cache", None)
        key_figures = results.get("key_figures", {})

        report_id = uuid.uuid4().hex[:10]
        _store[report_id] = {
            "text":        extract_text_from_pdf(io.BytesIO(pdf_bytes))[0],
            "sent_cache":  sent_cache,
            "key_figures": key_figures,
        }

        results["filename"]  = pdf_file.filename
        results["report_id"] = report_id

        # Pre-compute health score and attach
        results["health"] = calculate_health_score(key_figures)

        return jsonify({"success": True, "data": results})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Semantic search ───────────────────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search():
    body      = request.get_json(silent=True) or {}
    query     = body.get("query", "").strip()
    report_id = body.get("report_id", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    entry = _store.get(report_id) or (list(_store.values())[-1] if _store else None)
    if not entry:
        return jsonify({"error": "No report loaded. Please upload a PDF first."}), 400

    try:
        results = semantic_search(query, entry["text"], top_n=3, cached=entry["sent_cache"])
        return jsonify({"success": True, "results": results, "query": query})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── AI Chatbot ────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    body      = request.get_json(silent=True) or {}
    message   = body.get("message", "").strip()
    report_id = body.get("report_id", "")
    lang      = body.get("lang", "")

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    if lang not in ("en", "sw"):
        lang = detect_language(message)

    entry = _store.get(report_id) or (list(_store.values())[-1] if _store else None)

    if not entry:
        no_report = (
            "Habari! Tafadhali pakia ripoti ya CBK kwanza ili niweze kujibu maswali yako."
            if lang == "sw" else
            "Hello! Please upload a CBK report first so I can answer your questions about Kenya's economy."
        )
        return jsonify({"answer": no_report, "type": "no_report", "sources": []})

    try:
        results  = semantic_search(message, entry["text"], top_n=3, cached=entry["sent_cache"])
        response = generate_chat_response(message, results, entry["key_figures"], lang)
        return jsonify({
            "success": True,
            "answer":  response["answer"],
            "type":    response["type"],
            "sources": results,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Predictive text ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    body    = request.get_json(silent=True) or {}
    partial = body.get("text", "").strip()
    return jsonify({"suggestions": get_predictive_suggestions(partial)})


# ── Economic health score ─────────────────────────────────────────────────────
@app.route("/health-score", methods=["GET"])
def health_score():
    report_id = request.args.get("report_id", "")
    entry = _store.get(report_id) or (list(_store.values())[-1] if _store else None)
    if not entry:
        return jsonify({"error": "No report loaded"}), 400
    return jsonify({"success": True, **calculate_health_score(entry["key_figures"])})


# ── Cost-of-living impact ─────────────────────────────────────────────────────
@app.route("/cost-impact", methods=["POST"])
def cost_impact():
    body      = request.get_json(silent=True) or {}
    report_id = body.get("report_id", "")

    try:
        salary = float(body.get("salary", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid salary value"}), 400

    if salary <= 0:
        return jsonify({"error": "Please enter a positive salary"}), 400

    entry = _store.get(report_id) or (list(_store.values())[-1] if _store else None)
    if not entry:
        return jsonify({"error": "No report loaded"}), 400

    result = calculate_cost_of_living_impact(salary, entry["key_figures"])
    if result is None:
        return jsonify({"error": "Inflation data not found in this report."}), 400

    return jsonify({"success": True, "result": result})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
