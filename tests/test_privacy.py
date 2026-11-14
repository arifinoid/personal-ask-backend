import main


def test_redacts_an_international_phone_number() -> None:
    assert main.redact_phone_numbers("call +62 813-1841-4457 now") == (
        "call [phone redacted] now"
    )


def test_redacts_a_local_indonesian_number() -> None:
    assert main.redact_phone_numbers("0813 1841 4457") == "[phone redacted]"


def test_redacts_numbers_inside_resume_headers() -> None:
    text = "Kediri, Indonesia rohmadarifin007@gmail.com +6281318414457 Rohmad Arifin"

    assert main.redact_phone_numbers(text) == (
        "Kediri, Indonesia rohmadarifin007@gmail.com [phone redacted] Rohmad Arifin"
    )


def test_keeps_date_ranges_untouched() -> None:
    for text in (
        "September 2011 – April 2016 | Malang, Indonesia",
        "Alto Network 2024 - 2026",
        "March 2026 – Present",
    ):
        assert main.redact_phone_numbers(text) == text


def test_keeps_metrics_untouched() -> None:
    for text in ("99.99% SLA", "27 million records per day", "27M records/day"):
        assert main.redact_phone_numbers(text) == text


def test_keeps_email_addresses_untouched() -> None:
    assert main.redact_phone_numbers("rohmadarifin007@gmail.com") == "rohmadarifin007@gmail.com"


def test_chat_redacts_phone_numbers_from_sources(monkeypatch, client) -> None:
    hits = [
        {
            "title": "Resume",
            "url": "/docs/resume.pdf",
            "snippet": "Rohmad Arifin +6281318414457 fullstack software engineer.",
            "score": 0.9,
        }
    ]
    stub = type("Stub", (), {"search": lambda *_args, **_kwargs: hits})
    monkeypatch.setattr(main, "RETRIEVER", stub())

    body = client.post("/v1/chat", json={"question": "contact"}).json()

    assert "+6281318414457" not in body["answer"]
    assert "+6281318414457" not in body["sources"][0]["snippet"]
    assert "[phone redacted]" in body["answer"]
