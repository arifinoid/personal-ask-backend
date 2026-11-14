import os

# main.py reads its configuration at import time. Pin retrieval-only mode
# before importing it so the tests never depend on a developer's .env.
os.environ["LLM_BASE_URL"] = ""
os.environ["LLM_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import retrieval  # noqa: E402

TEST_DOCS = [
    {
        "id": "home",
        "title": "Home",
        "url": "/",
        "content": (
            "I'm Rohmad Arifin, a fullstack software engineer. I lead a team of "
            "11 engineers at Alto Network. Contact: rohmadarifin007@gmail.com."
        ),
    },
    {
        "id": "projects",
        "title": "Projects",
        "url": "/projects/",
        "content": (
            "Payment switching internal platform at Alto Network: high-volume "
            "data viewing and download for 27 million records per day. "
            "Unified login system at eFishery: authentication services and a "
            "JavaScript SDK used across multiple platforms."
        ),
    },
    {
        "id": "explore",
        "title": "Experience",
        "url": "/explore/",
        "content": (
            "Toolbox: TypeScript, React, Next.js, Node, Go, Python, Kubernetes. "
            "Fullstack associate manager at Alto Network since 2026."
        ),
    },
]


@pytest.fixture
def retriever() -> retrieval.TfidfRetriever:
    return retrieval.TfidfRetriever(TEST_DOCS)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, retriever: retrieval.TfidfRetriever):
    """TestClient backed by the small deterministic corpus above."""
    monkeypatch.setattr(main, "RETRIEVER", retriever)
    return TestClient(main.app)
