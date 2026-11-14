# Deploy — personal-ask-backend

Runs as the `ask-api` systemd unit on the VPS that serves `arifinoid.web.id`,
behind nginx under `/health` and `/v1/`.

The unit, the service user, the nginx routes and the env file are declared in
the NixOS repo:

```
nix/modules/nixos/vps-ask-api.nix
```

Nothing about the runtime is hand-edited on the VPS anymore.

## Layout on the VPS

| Path | What |
| --- | --- |
| `/srv/ask-api/current` | symlink to the active release |
| `/srv/ask-api/releases/<sha>` | one directory per deployed commit |
| `/etc/ask-api.env` | env file, rendered from sops-nix at activation |
| `/var/lib/ask-api/docs` | extra RAG documents (txt / md / pdf) |
| `/var/lib/ask-api` | `HOME` for the `ask-api` user |

Python and every dependency (`fastapi`, `uvicorn`, `pydantic`, `httpx`,
`python-dotenv`, `pypdf`) come from the Nix closure built by the module. There
is no venv and no `pip install` step on the host.

## Deploy flow

`.github/workflows/deploy.yml` runs on every push to `main`:

1. lint + `py_compile`
2. `rsync` the commit to `/srv/ask-api/releases/<sha>/`
3. atomically flip `/srv/ask-api/current`
4. `systemctl restart ask-api`
5. `curl` `http://127.0.0.1:8080/health` on the VPS

If step 3–5 fails, the previous release is restored and the job exits red.
Only the three newest releases are kept.

## RAM warning

This VPS has ~1.9 GiB of RAM and no GPU. **Do not run Ollama locally** — even
a 1B parameter model will OOM or thrash swap. The default is retrieval-only
mode (~200 MiB), which returns the top-3 sources instead of a generated
answer.

For generated answers, point `LLM_BASE_URL` at an external OpenAI-compatible
API (for example OpenRouter) instead of running a model on the VPS.

## Changing settings

Everything in `/etc/ask-api.env` is declared in `vps-ask-api.nix`:

```ini
ALLOWED_ORIGINS=https://arifinoid.web.id,https://www.arifinoid.web.id
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=llama3.2:1b
DOCS_DIR=/var/lib/ask-api/docs
```

Edit the `sops.templates."ask-api-env"` block, then rebuild the VPS. A real
`LLM_API_KEY` should be moved to a sops secret and referenced with
`${config.sops.placeholder.<name>}` instead of being committed to the NixOS
repo.

## Documents

Drop `.txt`, `.md`, `.markdown` or `.pdf` files into `/var/lib/ask-api/docs`
(owned by `ask-api:ask-api`), then restart the service so the corpus is
reloaded:

```bash
sudo install -m 0640 -o ask-api -g ask-api resume.pdf /var/lib/ask-api/docs/
sudo systemctl restart ask-api
```

## Verify

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok","corpus_docs":5,"retrieval_only":true,"llm_configured":false}

curl -X POST http://127.0.0.1:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"what is rohmads experience?"}'
```

From outside the VPS use `https://arifinoid.web.id/health` and
`https://arifinoid.web.id/v1/chat`.

CORS is locked to `ALLOWED_ORIGINS` from the env file.

## Rollback

Point the symlink at an older release and restart:

```bash
ls -1dt /srv/ask-api/releases/*/
sudo ln -sfn /srv/ask-api/releases/<sha> /srv/ask-api/current.new
sudo mv -Tf /srv/ask-api/current.new /srv/ask-api/current
sudo systemctl restart ask-api
```
