# Deploy — personal-ask-backend

Runs on the same VPS that serves `arifinoid.web.id`, behind nginx.

## RAM warning

This VPS has limited memory (~1.89 GiB, no GPU). **Do not run Ollama locally** — even a 1B parameter model will OOM or thrash swap.

Run in **retrieval-only mode** (default): leave `LLM_BASE_URL` unset. The service returns top-3 sources with titles, URLs, and snippets — zero model overhead, instant responses, ~200 MiB RAM.

If you want generated answers, point `LLM_BASE_URL` at an external inference API (e.g. OpenRouter) instead of running a model on this VPS.

## systemd unit

`/etc/systemd/system/ask-api.service`:
```ini
[Unit]
Description=ask.arifinoid.web.id FastAPI service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/arifinoid/personal-ask-backend
EnvironmentFile=/var/www/arifinoid/personal-ask-backend/.env
ExecStart=/usr/bin/uvicorn main:app --host 127.0.0.1 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

Commands:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ask-api
sudo systemctl status ask-api
```

## .env (VPS)

```ini
# CORS
ALLOWED_ORIGINS=https://arifinoid.web.id,http://localhost:3000

# LLM generation — LEAVE EMPTY for retrieval-only mode (recommended on this VPS)
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=llama3.2:1b

# Bind address for uvicorn
HOST=127.0.0.1
PORT=8080

# Directory containing extra documents (txt / md / pdf) for RAG
DOCS_DIR=/var/www/arifinoid/ask-docs
```

## Documents

Drop your resume and project docs (`.txt`, `.md`, `.markdown`, `.pdf`) into `$DOCS_DIR`:

```bash
sudo mkdir -p /var/www/arifinoid/ask-docs
sudo chown <your-user>:<your-user> /var/www/arifinoid/ask-docs
# copy your resume.pdf, notes.md, etc. here
```

Restart after adding/removing documents:
```bash
sudo systemctl restart ask-api
```

## nginx

Inside the `arifinoid.web.id` server block:
```nginx
location /health {
    proxy_pass http://127.0.0.1:8080;
}
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

Then `sudo nginx -t && sudo systemctl reload nginx`.

## External LLM (optional)

If you want generated answers without running a local model, set in `.env`:

```ini
LLM_BASE_URL=https://api.openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=meta-llama/llama-3.2-1b-instruct
```

The backend posts to any OpenAI-compatible `/v1/chat/completions` endpoint.

## Verify

```bash
curl http://localhost:8080/health
# {"status":"ok","corpus_docs":6,"retrieval_only":true,"llm_configured":false}

curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"what is rohmads experience?"}'
```

CORS is locked to `ALLOWED_ORIGINS` from `.env`.
