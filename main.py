"""ask.arifinoid.web.id — single FastAPI service for the ask panel.

Endpoints:
  GET  /health        -> liveness
  POST /v1/chat       -> { question } -> { answer, sources, retrieval_only }

Retrieval is pure-Python TF-IDF over a static corpus built from corpus.json
and/or documents stored in DOCS_DIR (txt/md/pdf). Optional LLM generation is
gated behind LLM_BASE_URL; when absent the service returns synthesized answers
from retrieved sources (retrieval_only mode).
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import retrieval

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus.json"
DOCS_DIR = Path(os.getenv("DOCS_DIR")) if os.getenv("DOCS_DIR") else None

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "https://arifinoid.web.id,http://localhost:3000"
    ).split(",")
    if o.strip()
]

LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""
LLM_API_KEY = os.getenv("LLM_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")

RETRIEVAL_ONLY_NOTICE = (
    "LLM generation not configured on this service. "
    "Below are the most relevant sections from the site — open a link to read more."
)


def load_corpus(
    corpus_path: Path = CORPUS_PATH, docs_dir: Path | None = DOCS_DIR
) -> list[dict]:
    docs: list[dict] = []

    # 1. Static corpus.json (site pages)
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            docs.extend(data)

    # 2. Documents from docs_dir (txt / md / pdf)
    if docs_dir and docs_dir.is_dir():
        for path in sorted(docs_dir.iterdir()):
            if not path.is_file():
                continue
            content = _read_file(path)
            if not content.strip():
                continue

            # One long file (a resume, a project write-up) is a bad retrieval
            # unit: TF-IDF dilutes it and the best window can sit anywhere in
            # it. Split on line boundaries so each chunk competes on its own.
            chunks = chunk_text(content)
            title = path.stem.replace("_", " ").replace("-", " ").title()
            url = f"/docs/{path.name}"
            for index, chunk in enumerate(chunks, 1):
                doc_id = f"file:{path.stem}" if len(chunks) == 1 else f"file:{path.stem}#{index}"
                docs.append({"id": doc_id, "title": title, "url": url, "content": chunk})

    return docs


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split extracted document text into retrieval-sized chunks.

    Lines are packed until they reach max_chars, which keeps headings next to
    the section they introduce. A single oversized line is split on spaces.
    """
    blocks = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if current:
            chunks.append(" ".join(current))
            current, length = [], 0

    for block in blocks:
        if len(block) > max_chars:
            flush()
            chunks.extend(_split_long_block(block, max_chars))
            continue
        if current and length + len(block) + 1 > max_chars:
            flush()
        current.append(block)
        length += len(block) + 1
    flush()

    return chunks


def _split_long_block(block: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in block.split():
        if current and length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8")
    if ext == ".pdf":
        return _read_pdf(path)
    return ""


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:  # noqa: BLE001 — pypdf not installed
        return ""
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


CORPUS = load_corpus()
RETRIEVER = retrieval.TfidfRetriever(CORPUS) if CORPUS else None


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_only: bool


app = FastAPI(title="ask-arifinoid", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "corpus_docs": len(CORPUS),
        "retrieval_only": not bool(LLM_BASE_URL),
        "llm_configured": bool(LLM_BASE_URL),
    }


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    question = req.question.strip()
    # Retrieve more than we cite: a long document can occupy several chunk
    # slots, and one link per document is all the answer needs.
    hits = _select_hits(RETRIEVER.search(question, top_k=8)) if RETRIEVER else []
    chunks = [_to_source(hit) for hit in hits]
    sources = [_to_source(hit) for hit in _unique_hits(hits)]

    retrieval_only = not bool(LLM_BASE_URL)
    if retrieval_only:
        answer = build_retrieval_only_answer(question, sources, chunks)
    else:
        answer = generate_with_llm(question, sources)

    return ChatResponse(answer=answer, sources=sources, retrieval_only=retrieval_only)


def _select_hits(hits: list[dict], min_ratio: float = 0.55) -> list[dict]:
    """Drop weak matches relative to the best hit.

    Retrieval-only answers have no model to discard irrelevant snippets, so a
    page that merely shares a common word ("engineer") would otherwise end up
    as a bullet next to the real answer.
    """
    if not hits:
        return []
    floor = hits[0]["score"] * min_ratio
    return [hit for hit in hits if hit["score"] >= floor]


def _unique_hits(hits: list[dict], limit: int = 3) -> list[dict]:
    """Keep the best hit per URL so chunked documents don't crowd the answer."""
    picked: list[dict] = []
    seen: set[str] = set()
    for hit in hits:
        url = hit.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        picked.append(hit)
        if len(picked) >= limit:
            break
    return picked


_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact_phone_numbers(text: str) -> str:
    """Hide phone numbers that documents such as a resume carry in headers.

    Candidates need 9-15 digits, which keeps date ranges ("2024 - 2026"), SLA
    figures ("99.99%") and record counts out of scope.
    """

    def replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        return "[phone redacted]" if 9 <= len(digits) <= 15 else match.group()

    return _PHONE_CANDIDATE.sub(replace, text)


def _to_source(hit: dict) -> Source:
    return Source(
        title=hit["title"],
        url=hit["url"],
        snippet=redact_phone_numbers(hit["snippet"]),
    )


def build_retrieval_only_answer(
    question: str, sources: list[Source], chunks: list[Source] | None = None
) -> str:
    q_lower = question.lower()
    greetings = {"hi", "hello", "hey", "hi there", "hello there", "greetings"}
    if q_lower in greetings or q_lower.startswith(("hi ", "hello ", "hey ")):
        return (
            "Hi there! I can answer questions about Rohmad's portfolio site.\n"
            "Try asking about his experience, projects, or blog — for example:\n"
            "• What projects has he built?\n"
            "• Where does he work now?\n"
            "• What's his experience with Kubernetes?\n"
            "• Tell me about the unified login system"
        )
    if not sources:
        return (
            "I couldn't find anything relevant on the site for that question.\n"
            "Try asking about his experience, projects, or blog notes instead:\n"
            "• What projects has Rohmad built?\n"
            "• Where does he work now?\n"
            "• What's his experience with Kubernetes?\n"
            "• Tell me about the unified login system"
        )

    answer = _synthesize_answer(question, chunks if chunks is not None else sources)

    # Add source links at the bottom
    answer += "\n\nSources:\n"
    for i, s in enumerate(sources, 1):
        answer += f"{i}. [{s.title}]({s.url})\n"

    return answer


def _synthesize_answer(question: str, sources: list[Source]) -> str:
    q_lower = question.lower()

    # Detect intent
    is_projects = any(w in q_lower for w in ["project", "build", "make", "create"])
    is_work = any(
        w in q_lower
        for w in [
            "work",
            "job",
            "company",
            "employer",
            "alto",
            "aia",
            "efishery",
            "jubelio",
            "sicepat",
        ]
    )
    is_experience = any(
        w in q_lower
        for w in ["experience", "year", "background", "career", "history"]
    )
    is_skills = any(
        w in q_lower
        for w in [
            "skill",
            "tech",
            "stack",
            "language",
            "framework",
            "tool",
            "kubernetes",
            "python",
            "react",
            "java",
            "go",
            "node",
        ]
    )
    is_contact = any(
        w in q_lower
        for w in [
            "contact",
            "email",
            "reach",
            "linkedin",
            "github",
            "where",
            "location",
            "kediri",
        ]
    )
    is_blog = any(w in q_lower for w in ["blog", "note", "write", "article", "post"])

    # Greeting / intro based on intent
    if is_projects:
        intro = "Rohmad has built several projects across his career:"
    elif is_work:
        intro = "Here's where Rohmad has worked:"
    elif is_experience:
        intro = "Rohmad's background spans several years across multiple companies:"
    elif is_skills:
        intro = "Rohmad's technical skills include:"
    elif is_contact:
        intro = "Here's how to reach Rohmad:"
    elif is_blog:
        intro = "Here's what's on the blog:"
    else:
        intro = "Here's what I found:"

    # Extract key facts from the snippets returned by the retriever
    bullets: list[str] = []
    seen: set[str] = set()
    for s in sources:
        snippet = (s.snippet or "").strip()
        key = snippet[:60]
        if key in seen or len(snippet) < 20:
            continue
        seen.add(key)

        clean = _clean_snippet(snippet)
        if clean:
            bullets.append(f"• {clean}")

    if not bullets:
        return "I found some relevant pages but couldn't extract a clear summary."

    return "\n".join([intro, *bullets])


def _clean_snippet(text: str) -> str:
    """Clean raw page text into a readable sentence."""
    if not text:
        return ""

    # Remove common code artifacts
    text = text.replace("export const metadata", "")
    text = text.replace("export default function", "")
    text = text.replace("return (", "")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("{ }", "")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # PDF extraction glues bullet glyphs to the following word ("•Led ...")
    # and leaves table rules ("|") behind; drop them at the window edges.
    text = re.sub(r"^[•·*|-]+\s*", "", text)
    text = re.sub(r"\s*[•·|]+$", "", text)

    # The retriever prefixes an ellipsis when a snippet starts mid-sentence.
    # Drop that partial sentence so the bullet starts at a sentence boundary.
    if text.startswith("…"):
        text = text[1:].lstrip()
        cut = text.find(". ")
        if cut != -1 and cut + 2 < len(text) // 2:
            text = text[cut + 2:]

    # Drop a trailing ellipsis; the summary adds its own "..." marker.
    text = text.rstrip("… ").strip()

    # Take first ~200 chars as a summary
    if len(text) > 200:
        text = text[:200].rsplit(" ", 1)[0] + "..."

    return text


def generate_with_llm(question: str, sources: list[Source]) -> str:
    import httpx

    context = "\n\n".join(
        f"Source {i+1} [{s.title}]({s.url}): {s.snippet}"
        for i, s in enumerate(sources)
    ) or "No relevant sources found."

    system = (
        "You answer questions about Rohmad Arifin's personal portfolio site. "
        "Use only the provided sources. If the answer isn't in the sources, say so. "
        "Cite sources by title and link. Be concise."
    )
    user = f"Question: {question}\n\nSources:\n{context}"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    try:
        resp = httpx.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 — degrade to retrieval-only on any failure
        return build_retrieval_only_answer(question, sources) + (
            f"\n\n(LLM generation failed: {exc}; showing retrieved sources instead.)"
        )
