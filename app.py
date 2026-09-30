"""Flask backend for the AI Data Science Insight Generator."""
import math
import uuid
from collections import OrderedDict

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

import insight_engine as ie

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit

RESULTS = OrderedDict()  # in-memory store: id -> pipeline result (last 20 kept)


def clean(o):
    """Make numpy/pandas values JSON-safe (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file uploaded."), 400
    try:
        res = ie.run_pipeline(f, f.filename, use_ai=False)
    except Exception as e:
        return jsonify(error=f"Could not process file: {e}"), 400
    rid = uuid.uuid4().hex[:12]
    RESULTS[rid] = {**res, "filename": f.filename}
    while len(RESULTS) > 20:
        RESULTS.popitem(last=False)
    p = res["profile"]
    return jsonify(clean({
        "id": rid, "filename": f.filename,
        "kpis": {"rows": p["rows"], "columns": p["columns"], "duplicates": p["duplicates"],
                 "insights": len(res["insights"])},
        "columns": p["columns_info"], "insights": res["insights"],
        "charts": res["charts"],
        "preview": res["df"].head(10).astype(str).to_dict(orient="records"),
    }))


@app.post("/api/ai/<rid>")
def ai(rid):
    r = RESULTS.get(rid)
    if not r:
        return jsonify(error="Session expired. Re-upload the file."), 404
    text = ie.ai_narrative(r["profile"], r["insights"], r["filename"])
    if text is None:
        return jsonify(error="Set ANTHROPIC_API_KEY on the server to enable AI summaries."), 501
    r["narrative"] = text
    return jsonify(text=text)


@app.get("/api/report/<rid>")
def report(rid):
    r = RESULTS.get(rid)
    if not r:
        return "Session expired. Re-upload the file.", 404
    html = ie.build_html_report(r["df"], r["profile"], r["insights"], r["charts"],
                                r.get("narrative"), r["filename"])
    return Response(html, mimetype="text/html",
                    headers={"Content-Disposition": "attachment; filename=insight_report.html"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
