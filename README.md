# Semantic Document Search

A small REST API that stores short text documents and searches them by **semantic
similarity** (meaning, not keywords). Built with FastAPI, SQLAlchemy 2.0, Pydantic v2,
SQLite, and `sentence-transformers` (`all-MiniLM-L6-v2`).

## Steps to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

> First start downloads the embedding model (~90 MB) once, then caches it.

Open http://127.0.0.1:8000/docs for the interactive API.

### Seed the sample documents

```bash
for d in \
 '{"title":"RCS Thruster Firing Procedure","content":"This procedure describes the cold gas thruster ignition sequence for the reaction control system. It covers pre-ignition checks, valve actuation order, and post-firing telemetry verification."}' \
 '{"title":"Solar Panel Deployment Sequence","content":"Detailed steps for deploying the photovoltaic arrays after orbital insertion. Includes hinge release commands, deployment angle monitoring, and power generation confirmation checks."}' \
 '{"title":"Thermal Control System Overview","content":"Description of the passive and active thermal regulation mechanisms onboard Nyx. Covers heat pipe routing, multi-layer insulation zones, and heater activation thresholds."}' \
 '{"title":"Communication Link Budget","content":"Analysis of the S-band uplink and downlink margins. Includes antenna gain figures, free-space path loss calculations, and minimum required Eb/N0 for command reception."}' \
 '{"title":"Propellant Tank Pressurisation Procedure","content":"Step-by-step guide for pressurising the hydrazine tank prior to orbital manoeuvre. Covers regulator settings, pressure transducer readings, and abort criteria."}' ; do
  curl -s -X POST http://127.0.0.1:8000/documents -H "Content-Type: application/json" -d "$d" > /dev/null
done
```

### Try a semantic search

```bash
curl "http://127.0.0.1:8000/documents/search?q=thruster%20firing%20sequence"
# top hit is the RCS Thruster doc even though "propulsion" queries also match it
curl "http://127.0.0.1:8000/documents/search?q=propulsion&filter_title=thruster"
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/documents` | Store a document; returns it with `id` and `created_at` (201). 409 if an identical `title`+`content` already exists. |
| `GET` | `/documents/search?q=&top_k=5&filter_title=` | Rank stored docs by cosine similarity (200). |
| `DELETE` | `/documents/{id}` | Delete by id (204), or 404 if missing. |

## Tests

```bash
pytest
```

Tests use the **real** model so semantic ranking is actually verified — each
query's *top* hit must be the correct doc — plus `top_k`, the `filter_title`
filter, duplicate (409), delete/404, and input validation.

## Design notes

- **Layout:** `models` (ORM) · `schemas` (Pydantic) · `routers` (HTTP) · `embeddings`
  (model + ranking) · `database` (engine/session). One concern per file.
- **Embeddings** are computed at ingestion and stored in the same row as JSON via a
  small SQLAlchemy `TypeDecorator` — no vector DB needed.
- Vectors are L2-normalised at encode time, so cosine similarity is a single NumPy
  matrix-vector dot product (fast, no divide-by-zero).
- The model is loaded once via FastAPI's `lifespan` (singleton); sync endpoints run in
  FastAPI's thread pool and inference is read-only, so sharing the model is safe.

## Assumptions

- The embedding is computed over `title + content` (titles carry strong keywords).
- `top_k` defaults to 5, bounded to `[1, 100]`.
- `q`/`title`/`content`/`filter_title` are whitespace-stripped; empty or
  whitespace-only → 422.
- Reposting an identical `title`+`content` returns 409 (no duplicate rows).
- `created_at` is timezone-aware UTC.
