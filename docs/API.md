# API reference — personal-ask-backend

Single service powering the ask panel on `arifinoid.web.id`. Retrieval is
pure-Python TF-IDF over a static corpus; LLM generation is optional.

## Endpoints

### `GET /health`
Liveness check. Returns corpus doc count and whether generation is enabled.

```
curl https://arifinoid.web.id/health
```

Response:
```json
{
  "status": "ok",
  "corpus_docs": 5,
  "retrieval_only": true
}
```

### `POST /v1/chat`
Ask a question about the portfolio site.

```bash
curl -X POST https://arifinoid.web.id/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Where does Rohmad work now?"}'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    { "title": "Experience", "url": "/explore/", "snippet": "..." }
  ],
  "retrieval_only": true
}
```

- `answer` — generated answer when `LLM_BASE_URL`/`LLM_API_KEY` are set; otherwise a retrieval-only listing of top sources.
- `sources` — top-3 TF-IDF matches from the corpus.
- `retrieval_only` — `true` when no LLM is configured.

## Corpus

Regenerate `corpus.json` from the Next.js pages in `personal-web`:

```
cd personal-web
bun run build:ask-index
# copies public/ask-index.json → personal-ask-backend/corpus.json
```

Restart the service after regenerating.

## Deploy

See `README.md` for the systemd + nginx setup on the VPS.
