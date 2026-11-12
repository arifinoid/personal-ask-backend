# personal-ask-backend

Single FastAPI service powering the right-side ask panel on `arifinoid.web.id`.

## Endpoints

- `GET /health` — liveness + corpus doc count + retrieval-only flag
- `POST /v1/chat` — `{ "question": "..." }` → `{ answer, sources, retrieval_only }`

See `docs/API.md` for the full reference.

## Retrieval

Pure-Python TF-IDF (`retrieval.py`) over `corpus.json`. No numpy, no
embedding model, no vector DB — deliberately small.

Regenerate `corpus.json` from the Next.js pages in `personal-web`:

```bash
# run inside personal-web
bun run build:ask-index
# then copy public/ask-index.json → personal-ask-backend/corpus.json
```

Restart the service after regenerating.

## Optional LLM generation

Set these env vars (copy `.env.example` to `.env`):

- `LLM_BASE_URL` — OpenAI-compatible chat completions base URL
- `LLM_API_KEY` — bearer token
- `LLM_MODEL` — model name (default `gpt-3.5-turbo`)

When unset, the service returns retrieved sources only (`retrieval_only: true`).
No key is ever exposed to the browser.

## Run locally

```bash
cp .env.example .env
# edit .env as needed

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

set -a; . .env; set +a
uvicorn main:app --host 127.0.0.1 --port 8080
```

Then:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Where does Rohmad work?"}'
```

## Deploy

See `docs/DEPLOY.md` for systemd + nginx on the VPS.
