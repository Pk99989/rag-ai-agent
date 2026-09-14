pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY (free key: https://console.groq.com)

python ingest.py        # builds the vector index from docs_*.md
pytest -v                # unit tests — no API key needed
python eval_run.py       # RBAC regression (no key needed) + generation smoke test (needs key)

streamlit run app.py     # opens the chatbot at http://localhost:8501
```