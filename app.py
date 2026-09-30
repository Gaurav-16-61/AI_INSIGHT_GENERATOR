"""Flask backend: accepts ANY file type."""
import math
import uuid
from collections import OrderedDict

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

import analyzers as A

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
RESULTS = OrderedDict()  # id -> result (last 20 kept)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File too large (limit is 25 MB)."), 413


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file uploaded."), 400
    try:
        res = A.analyze_file(f.read(), f.filename)
    except Exception as e:
        return jsonify(error=f"Could not process file: {e}"), 400
    res["filename"] = f.filename
    rid = uuid.uuid4().hex[:12]
    RESULTS[rid] = res
    while len(RESULTS) > 20:
        RESULTS.popitem(last=False)
    return jsonify(clean({"id": rid, "filename": f.filename, "kind_label": res["kind_label"],
                          "kpis": [{"label": k, "value": v} for k, v in res["kpis"]],
                          "insights": res["insights"], "charts": res["charts"],
                          "columns": res["columns"], "preview_html": res["preview_html"]}))


@app.post("/api/ai/<rid>")
def ai(rid):
    r = RESULTS.get(rid)
    if not r:
        return jsonify(error="Session expired. Re-upload the file."), 404
    text = A.ai_summary(r)
    if text is None:
        return jsonify(error="Set ANTHROPIC_API_KEY on the server to enable AI summaries."), 501
    r["narrative"] = text
    return jsonify(text=text)


@app.get("/api/report/<rid>")
def report(rid):
    r = RESULTS.get(rid)
    if not r:
        return "Session expired. Re-upload the file.", 404
    return Response(A.build_report(r, r.get("narrative")), mimetype="text/html",
                    headers={"Content-Disposition": "attachment; filename=insight_report.html"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
