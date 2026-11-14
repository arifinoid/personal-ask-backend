# Documents — extra knowledge for the ask panel

The service answers from two places:

| source | what it is | where it lives |
| --- | --- | --- |
| `corpus.json` | the site pages (home, about, experience, projects, blog) | repo root, generated from `personal-web` |
| `DOCS_DIR` | your own documents (resume, project write-ups, notes) | `/var/lib/ask-api/docs` on the VPS |

Both are indexed together at startup, so a question like "what did Rohmad build at
Alto Network?" can be answered from the resume even when the site page is thin.

## What happens to an uploaded file

1. At startup the service reads every file in `DOCS_DIR`.
   Supported: `.md`, `.markdown`, `.txt`, `.pdf` (text-based PDFs, extracted with
   `pypdf`). Anything else, and empty files, are skipped.
2. The text is split into **chunks of up to 700 characters on line boundaries**
   (`chunk_text` in `main.py`). Each chunk is its own retrieval unit, which is why
   a long resume does not drown out short site pages.
3. The title comes from the filename (`resume.pdf` → `Resume`,
   `project-onefish.md` → `Project Onefish`) and every chunk is published at
   `/docs/<filename>`.
4. The in-memory TF-IDF index is rebuilt. **The service only re-reads documents on
   restart.**

## Adding a document

From Windows PowerShell (quote the path, it has spaces):

```powershell
scp "C:\Users\Arif Kiki\Downloads\Rohmad Arifin resume.pdf" `
    root@43.173.26.172:/var/lib/ask-api/docs/
```

Then fix ownership and restart — the service runs as `ask-api:ask-api` and nginx
serves the same files through the `ask-api` group:

```bash
ssh root@43.173.26.172 '
  chown ask-api:ask-api /var/lib/ask-api/docs/ &&
  chmod 0750 /var/lib/ask-api/docs &&
  chmod 0640 /var/lib/ask-api/docs/* &&
  systemctl restart ask-api
'
```

From macOS/Linux the same thing is one line:

```bash
scp "Rohmad Arifin resume.pdf" root@43.173.26.172:/var/lib/ask-api/docs/ && \
ssh root@43.173.26.172 'chown ask-api:ask-api /var/lib/ask-api/docs/* && chmod 0640 /var/lib/ask-api/docs/* && systemctl restart ask-api'
```

## Verifying

```bash
curl -s https://arifinoid.web.id/health
# {"status":"ok","corpus_docs":14,...}   <- grows with every chunk you add

curl -s https://arifinoid.web.id/v1/chat \
  -H 'content-type: application/json' \
  -d '{"question":"What is Rohmad'\''s education?"}'
```

`corpus_docs` counts site pages **plus chunks**, so 5 site pages + one 9-chunk
resume reports 14. If the number does not move, the file was unreadable (scanned
PDF), had an unsupported extension, or the service was not restarted.

The PDF itself is served at `https://arifinoid.web.id/docs/resume.pdf`, which is
where the source links in answers point. `autoindex` is off, so a file is only
reachable by its exact name.

## Writing good knowledge documents

- **One topic per file.** `project-payment-switching.md`, `project-onefish-dls.md`,
  `resume.pdf`. A file named after the topic gives the retriever a useful title.
- **Markdown headings help.** Chunks break on line boundaries, so a heading and
  the lines under it usually stay in the same chunk.
- **Text, not scans.** If `pdftotext -layout file.pdf` prints nothing useful, the
  PDF is an image; run `ocrmypdf` first or paste the text into a `.md` file.
- **Long is fine, monolithic is not.** A 5,000-word document is chunked
  automatically, but each chunk competes alone, so keep sections self-contained
  (name the project in the section, not only in the title).
- **No secrets.** Everything in `DOCS_DIR` is world-downloadable at
  `/docs/<name>`. Phone numbers are redacted from answers automatically; email
  addresses are left alone because they are already public on the site.
- **Dates/metrics stay readable.** Redaction only triggers on 9–15 digit runs, so
  "2024 – 2026" and "99.99% SLA" survive untouched.

## How retrieval uses them

- Function words ("what", "is", "his") and single-letter tokens are dropped from
  queries and documents.
- Results are ranked by TF-IDF cosine similarity with a title boost, expanded for
  simple plurals ("projects" also matches "project").
- A page whose **whole title appears in the question** is boosted ("What is his
  experience?" lands on the page titled *Experience*, not on a short note that
  says "developer experience" once).
- Section headings and acronyms (`EDUCATION`, `SKILLS`, `SLA`) are boosted, so a
  resume's education section beats a passing mention in its contact header.
- Hits weaker than 55% of the best hit are discarded, so an unrelated page can no
  longer ride along on a shared word like "engineer".
- At most three sources are cited, **one per document** — several chunks of the
  resume may feed the bullets, but the answer links the resume once.
- Snippets are windowed on word/sentence boundaries around the matched term, then
  cleaned of PDF artifacts (`•Led ...` → `Led ...`).

## Refreshing the site pages

`corpus.json` is generated, not hand-written:

```bash
cd personal-web
bun run build:ask-index        # writes public/ask-index.json + ../personal-ask-backend/corpus.json
cd ../personal-ask-backend
git add corpus.json && git commit -m "chore: refresh ask corpus"
```

Merging to `main` redeploys the service (release symlink swap + restart).

## Troubleshooting

| symptom | cause |
| --- | --- |
| `corpus_docs` unchanged after upload | not restarted, unsupported extension, empty file, or scanned PDF |
| answer cites a document but shows no content | the matched chunk had nothing but boilerplate; add a real summary section |
| `/docs/<file>` returns 403 | file not `0640` or not owned by `ask-api:ask-api` |
| `/docs/<file>` returns 404 | filename mismatch (spaces vs `%20`) or the nginx route is missing |
