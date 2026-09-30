"""Core engine: load -> profile -> insights -> charts -> (optional AI narrative) -> HTML report."""
import base64
import html
import io
import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- 1. Loading ----------
def load_file(file, name: str) -> pd.DataFrame:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        df = pd.read_csv(file)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(file)
    elif ext == "json":
        df = pd.read_json(file)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return coerce_dtypes(df)


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn object columns that look like dates into datetime."""
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[c], errors="coerce")
        if df[c].notna().sum() and parsed.notna().sum() / df[c].notna().sum() > 0.9:
            df[c] = parsed
    return df


# ---------- 2. Profiling ----------
def profile(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"name": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 2), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            info.update(mean=round(s.mean(), 3), std=round(s.std(), 3), min=s.min(),
                        median=s.median(), max=s.max(), skew=round(s.skew(), 3))
        cols.append(info)
    return {
        "rows": len(df), "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "numeric": df.select_dtypes("number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
        "datetime": df.select_dtypes("datetime").columns.tolist(),
        "columns_info": cols,
    }


# ---------- 3. Rule-based insights ----------
def generate_insights(df: pd.DataFrame, p: dict) -> list[dict]:
    out = []
    add = lambda sev, cat, txt: out.append({"severity": sev, "category": cat, "text": txt})
    n = len(df)

    if p["duplicates"]:
        add("warning", "Data quality", f"{p['duplicates']} duplicate rows ({p['duplicates']/n:.1%}) found.")
    for c in p["columns_info"]:
        if c["missing_pct"] > 30:
            add("critical", "Data quality", f"'{c['name']}' is {c['missing_pct']}% missing - consider dropping or imputing.")
        elif c["missing_pct"] > 5:
            add("warning", "Data quality", f"'{c['name']}' has {c['missing_pct']}% missing values.")
        if c["unique"] == 1:
            add("info", "Data quality", f"'{c['name']}' is constant and carries no information.")
        elif c["unique"] == n and c["name"] in p["categorical"]:
            add("info", "Structure", f"'{c['name']}' is unique per row - likely an identifier.")

    for c in p["numeric"]:
        s = df[c].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        if iqr > 0:
            frac = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
            if frac > 0.05:
                add("warning", "Outliers", f"'{c}' has {frac:.1%} outliers (IQR rule).")
        sk = s.skew()
        if abs(sk) > 1:
            add("info", "Distribution", f"'{c}' is {'right' if sk > 0 else 'left'}-skewed (skew={sk:.2f}); a log transform may help.")

    if len(p["numeric"]) >= 2:
        corr = df[p["numeric"]].corr()
        seen = set()
        for a in corr.columns:
            for b in corr.columns:
                if a != b and (b, a) not in seen and abs(corr.loc[a, b]) >= 0.7:
                    seen.add((a, b))
                    r = corr.loc[a, b]
                    add("info", "Correlation", f"Strong {'positive' if r > 0 else 'negative'} correlation between '{a}' and '{b}' (r={r:.2f}).")

    for c in p["categorical"]:
        vc = df[c].value_counts(normalize=True)
        if len(vc) > 1 and vc.iloc[0] > 0.8:
            add("warning", "Imbalance", f"'{c}' is dominated by '{vc.index[0]}' ({vc.iloc[0]:.0%}).")

    for d in p["datetime"]:
        for c in p["numeric"][:3]:
            sub = df[[d, c]].dropna()
            if len(sub) > 10:
                x = sub[d].map(pd.Timestamp.toordinal).astype(float)
                slope = np.polyfit(x, sub[c], 1)[0]
                add("info", "Trend", f"'{c}' shows an overall {'upward' if slope > 0 else 'downward'} trend over '{d}'.")

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda i: order[i["severity"]])


# ---------- 4. Charts (base64 PNGs) ----------
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_charts(df: pd.DataFrame, p: dict) -> dict:
    charts = {}
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values(ascending=False).head(15)
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 3))
        miss.plot.barh(ax=ax, color="#d9534f"); ax.set_xlabel("% missing"); ax.set_title("Missing values")
        charts["Missing values"] = _fig_to_b64(fig)
    if len(p["numeric"]) >= 2:
        corr = df[p["numeric"][:15]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
        fig.colorbar(im); ax.set_title("Correlation heatmap")
        charts["Correlation heatmap"] = _fig_to_b64(fig)
    for c in p["numeric"][:6]:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df[c].dropna(), bins=30, color="#4a7bd0"); ax.set_title(f"Distribution: {c}")
        charts[f"Distribution: {c}"] = _fig_to_b64(fig)
    for c in p["categorical"][:4]:
        vc = df[c].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        vc.iloc[::-1].plot.barh(ax=ax, color="#5cb85c"); ax.set_title(f"Top values: {c}")
        charts[f"Top values: {c}"] = _fig_to_b64(fig)
    return charts


# ---------- 5. Optional AI narrative ----------
def ai_narrative(p: dict, insights: list[dict], filename: str) -> str | None:
    """Uses Claude if ANTHROPIC_API_KEY is set; otherwise returns None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        summary = {k: p[k] for k in ("rows", "columns", "duplicates", "numeric", "categorical", "datetime")}
        summary["columns_info"] = p["columns_info"][:40]
        bullets = "\n".join(f"- [{i['severity']}] {i['text']}" for i in insights[:40])
        prompt = (f"You are a senior data scientist. Dataset: {filename}.\n"
                  f"Profile: {summary}\n\nAutomated findings:\n{bullets}\n\n"
                  "Write: 1) an executive summary (4-5 sentences), 2) key risks, "
                  "3) recommended cleaning/feature-engineering steps, 4) suggested ML/analysis "
                  "directions. Be specific to the data and use only the facts given.")
        client = anthropic.Anthropic()
        msg = client.messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                                     max_tokens=1200, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:  # never break the report because of the LLM
        return f"(AI narrative unavailable: {e})"


# ---------- 6. HTML report ----------
def build_html_report(df, p, insights, charts, narrative, filename) -> str:
    e = html.escape
    badge = {"critical": "#d9534f", "warning": "#f0ad4e", "info": "#5bc0de"}
    ins = "".join(f"<li><span class='b' style='background:{badge[i['severity']]}'>{i['severity']}</span>"
                  f"<b>{e(i['category'])}:</b> {e(i['text'])}</li>" for i in insights) or "<li>No issues found.</li>"
    rows = "".join(
        f"<tr><td>{e(str(c['name']))}</td><td>{c['dtype']}</td><td>{c['missing_pct']}%</td><td>{c['unique']}</td>"
        f"<td>{c.get('mean', '')}</td><td>{c.get('min', '')}</td><td>{c.get('max', '')}</td></tr>"
        for c in p["columns_info"])
    imgs = "".join(f"<div class='c'><img src='data:image/png;base64,{b}' alt='{e(t)}'></div>" for t, b in charts.items())
    narr = f"<h2>AI Summary</h2><div class='n'>{e(narrative)}</div>" if narrative else ""
    preview = df.head(10).to_html(index=False, border=0)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Insight Report - {e(filename)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{border-bottom:3px solid #4a7bd0}} table{{border-collapse:collapse;width:100%;font-size:.85rem;display:block;overflow-x:auto}}
td,th{{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}} .k{{display:flex;gap:1rem;flex-wrap:wrap}}
.k div{{background:#f3f6fb;padding:1rem;border-radius:8px;flex:1;min-width:120px;text-align:center}} .k b{{font-size:1.5rem;display:block}}
.b{{color:#fff;padding:2px 8px;border-radius:10px;font-size:.7rem;margin-right:8px;text-transform:uppercase}}
li{{margin:.5rem 0}} .g{{display:flex;flex-wrap:wrap;gap:1rem}} .c img{{max-width:100%;width:470px}} .n{{white-space:pre-wrap;background:#f9f9f9;padding:1rem;border-left:4px solid #4a7bd0}}
</style></head><body>
<h1>AI Data Science Insight Report</h1><p>File: <b>{e(filename)}</b> &middot; Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="k"><div><b>{p['rows']:,}</b>Rows</div><div><b>{p['columns']}</b>Columns</div>
<div><b>{p['duplicates']}</b>Duplicates</div><div><b>{len(insights)}</b>Insights</div></div>
{narr}<h2>Key Insights</h2><ul>{ins}</ul>
<h2>Column Summary</h2><table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Mean</th><th>Min</th><th>Max</th></tr>{rows}</table>
<h2>Visualizations</h2><div class="g">{imgs}</div>
<h2>Data Preview</h2>{preview}</body></html>"""


def run_pipeline(file, name: str, use_ai: bool = True) -> dict:
    df = load_file(file, name)
    p = profile(df)
    insights = generate_insights(df, p)
    charts = make_charts(df, p)
    narrative = ai_narrative(p, insights, name) if use_ai else None
    report = build_html_report(df, p, insights, charts, narrative, name)
    return {"df": df, "profile": p, "insights": insights, "charts": charts,
            "narrative": narrative, "report_html": report}
