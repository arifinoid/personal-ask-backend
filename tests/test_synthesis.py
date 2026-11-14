import pytest

import main


def source(snippet: str, title: str = "Page", url: str = "/page/") -> main.Source:
    return main.Source(title=title, url=url, snippet=snippet)


PROJECT_SNIPPET = "Payment switching platform at Alto Network handling 27 million records per day."
WORK_SNIPPET = "Fullstack associate manager at Alto Network, leading 11 engineers on Scrum."


def test_synthesis_uses_the_snippet_field() -> None:
    """Regression: this read a non-existent Source.content and raised."""
    answer = main._synthesize_answer("projects", [source(PROJECT_SNIPPET)])

    assert PROJECT_SNIPPET in answer


@pytest.mark.parametrize(
    ("question", "intro"),
    [
        ("what projects has he built", "Rohmad has built several projects"),
        ("where does he work", "Here's where Rohmad has worked"),
        ("his experience", "Rohmad's background spans"),
        ("his kubernetes skills", "Rohmad's technical skills include"),
        ("how to contact him", "Here's how to reach Rohmad"),
        ("what is on the blog", "Here's what's on the blog"),
        ("something else entirely", "Here's what I found"),
    ],
)
def test_synthesis_picks_an_intro_for_the_detected_intent(question: str, intro: str) -> None:
    assert main._synthesize_answer(question, [source(PROJECT_SNIPPET)]).startswith(intro)


def test_synthesis_bullets_each_snippet() -> None:
    answer = main._synthesize_answer(
        "where does he work", [source(WORK_SNIPPET), source(PROJECT_SNIPPET)]
    )

    lines = answer.splitlines()
    assert lines[0] == "Here's where Rohmad has worked:"
    assert lines[1] == f"• {WORK_SNIPPET}"
    assert lines[2] == f"• {PROJECT_SNIPPET}"


def test_synthesis_skips_duplicate_snippets() -> None:
    answer = main._synthesize_answer("projects", [source(PROJECT_SNIPPET), source(PROJECT_SNIPPET)])

    assert answer.count(PROJECT_SNIPPET) == 1


def test_synthesis_skips_snippets_too_short_to_be_useful() -> None:
    answer = main._synthesize_answer("projects", [source("too short"), source(PROJECT_SNIPPET)])

    assert "too short" not in answer
    assert PROJECT_SNIPPET in answer


def test_synthesis_falls_back_when_nothing_can_be_extracted() -> None:
    answer = main._synthesize_answer("projects", [source("tiny"), source("")])

    assert answer == "I found some relevant pages but couldn't extract a clear summary."


def test_synthesis_handles_sources_without_usable_text() -> None:
    assert main._synthesize_answer("projects", []) == (
        "I found some relevant pages but couldn't extract a clear summary."
    )
