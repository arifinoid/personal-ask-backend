from fastapi.testclient import TestClient

import main


def test_health_reports_status_and_corpus_size(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["corpus_docs"] == len(main.CORPUS)
    assert body["retrieval_only"] is True
    assert body["llm_configured"] is False


def test_chat_answers_a_question_that_retrieves_documents(client: TestClient) -> None:
    """Regression: this path raised AttributeError and returned 500."""
    response = client.post("/v1/chat", json={"question": "What projects has Rohmad built?"})

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_only"] is True
    assert body["sources"], "expected at least one retrieved source"
    assert body["answer"].startswith("Rohmad has built several projects")
    assert "• " in body["answer"]
    assert "Sources:" in body["answer"]


def test_chat_sources_expose_title_url_and_snippet(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "What projects has Rohmad built?"})

    sources = response.json()["sources"]
    assert sources
    for source in sources:
        assert set(source) == {"title", "url", "snippet"}
        assert source["url"].startswith("/")


def test_chat_lists_source_links_at_the_end(client: TestClient) -> None:
    body = client.post("/v1/chat", json={"question": "projects"}).json()

    links = [line for line in body["answer"].splitlines() if line.startswith("1. [")]
    assert links, body["answer"]
    assert body["sources"][0]["title"] in links[0]


def test_chat_greets_without_retrieval(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Hi there!")
    assert body["sources"] == []


def test_chat_reports_no_hits_calmly(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "zzzqqq123"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("I couldn't find anything relevant")
    assert body["sources"] == []


def test_chat_rejects_a_blank_question(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "question is required"


def test_chat_rejects_a_missing_question(client: TestClient) -> None:
    assert client.post("/v1/chat", json={}).status_code == 422


def test_chat_works_without_a_retriever(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(main, "RETRIEVER", None)

    response = client.post("/v1/chat", json={"question": "projects"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


class StubRetriever:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        return self.hits[:top_k]


def hit(title: str, url: str, snippet: str, score: float) -> dict:
    return {"title": title, "url": url, "snippet": snippet, "score": score}


def test_select_hits_drops_matches_far_below_the_best_score() -> None:
    hits = [
        hit("Resume", "/docs/resume.pdf", "EDUCATION University of Brawijaya", 0.9),
        hit("Home", "/", "fullstack software engineer", 0.2),
    ]

    assert main._select_hits(hits) == [hits[0]]


def test_select_hits_keeps_close_matches() -> None:
    hits = [
        hit("Resume", "/docs/resume.pdf", "EDUCATION University of Brawijaya", 0.9),
        hit("About", "/about/", "fullstack software engineer", 0.8),
    ]

    assert main._select_hits(hits) == hits


def test_unique_hits_keeps_one_result_per_url() -> None:
    hits = [
        hit("Resume", "/docs/resume.pdf", "EDUCATION University of Brawijaya", 0.9),
        hit("Resume", "/docs/resume.pdf", "SKILLS TypeScript, React", 0.8),
        hit("Home", "/", "fullstack software engineer", 0.7),
    ]

    assert main._unique_hits(hits) == [hits[0], hits[2]]


def test_unique_hits_respects_the_limit() -> None:
    hits = [hit(str(i), f"/{i}/", "text long enough to be a snippet", 0.5) for i in range(5)]

    assert len(main._unique_hits(hits, limit=2)) == 2


def test_chat_cites_a_chunked_document_once_but_answers_from_several_chunks(
    monkeypatch, client: TestClient
) -> None:
    hits = [
        hit("Resume", "/docs/resume.pdf", "EDUCATION University of Brawijaya, Malang.", 0.9),
        hit("Resume", "/docs/resume.pdf", "SKILLS TypeScript, React, Next.js, Go.", 0.85),
    ]
    monkeypatch.setattr(main, "RETRIEVER", StubRetriever(hits))

    body = client.post("/v1/chat", json={"question": "education and skills"}).json()

    assert [source["url"] for source in body["sources"]] == ["/docs/resume.pdf"]
    assert "University of Brawijaya" in body["answer"]
    assert "TypeScript" in body["answer"]
    assert body["answer"].count("1. [Resume]") == 1


def test_chat_never_returns_500_for_template_questions(client: TestClient) -> None:
    templates = [
        "What projects has Rohmad built?",
        "Where does he work now?",
        "What's his experience with Kubernetes?",
        "Tell me about the unified login system",
    ]

    for question in templates:
        response = client.post("/v1/chat", json={"question": question})
        assert response.status_code == 200, question
        assert response.json()["answer"].strip(), question
