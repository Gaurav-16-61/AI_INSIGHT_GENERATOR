# AI Data Science Insight Generator (Web)
Flask backend + HTML/JS frontend. Upload CSV/Excel/JSON -> insights, charts, downloadable report.

## Run locally
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python app.py                                        # open http://localhost:5000

## Optional AI summary
    export ANTHROPIC_API_KEY=your_key                    # Windows: set ANTHROPIC_API_KEY=your_key

## Structure
    app.py             Flask routes (/api/analyze, /api/ai/<id>, /api/report/<id>)
    insight_engine.py  pandas analysis, charts, report builder
    static/index.html  frontend

## Deploy (free options)
Render / Railway / PythonAnywhere: add `gunicorn` to requirements.txt and use start command
    gunicorn app:app
Set ANTHROPIC_API_KEY as an environment variable on the host (never commit it).
