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


def load_corpus() -> list[dict]:
    docs: list[dict] = []

    # 1. Static corpus.json (site pages)
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            docs.extend(data)

    # 2. Documents from DOCS_DIR (txt / md / pdf)
    if DOCS_DIR and DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.iterdir()):
            if not path.is_file():
                continue
            content = _read_file(path)
            if not content.strip():
                continue
            docs.append(
                {
                    "id": f"file:{path.stem}",
                    "title": path.stem.replace("_", " ").title(),
                    "url": f"/docs/{path.name}",
                    "content": content,
                }
            )

    return docs


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
    hits = RETRIEVER.search(question, top_k=3) if RETRIEVER else []
    sources = [
        Source(title=h["title"], url=h["url"], snippet=h["snippet"]) for h in hits
    ]

    retrieval_only = not bool(LLM_BASE_URL)
    if retrieval_only:
        answer = build_retrieval_only_answer(question, sources)
    else:
        answer = generate_with_llm(question, sources)

    return ChatResponse(answer=answer, sources=sources, retrieval_only=retrieval_only)


def build_retrieval_only_answer(question: str, sources: list[Source]) -> str:
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

    answer = _synthesize_answer(question, sources)

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

    parts: list[str] = []

    # Greeting / intro based on intent
    if is_projects:
        parts.append("Rohmad has built several projects across his career:")
    elif is_work:
        parts.append("Here's where Rohmad has worked:")
    elif is_experience:
        parts.append("Rohmad's background spans several years across multiple companies:")
    elif is_skills:
        parts.append("Rohmad's technical skills include:")
    elif is_contact:
        parts.append("Here's how to reach Rohmad:")
    elif is_blog:
        parts.append("Here's what's on the blog:")
    else:
        parts.append("Here's what I found:")

    # Extract key facts from source snippets
    seen: set[str] = set()
    for s in sources:
        snippet = (s.content or "").strip()
        key = snippet[:60]
        if key in seen or len(snippet) < 20:
            continue
        seen.add(key)

        clean = _clean_snippet(snippet)
        if clean:
            parts.append(f"• {clean}")

    if not parts:
        parts.append("I found some relevant pages but couldn't extract a clear summary.")

    return "\n".join(parts)


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
