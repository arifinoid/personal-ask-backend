import re

import retrieval


def make_retriever() -> retrieval.TfidfRetriever:
    docs = [
        {
            "id": "home",
            "title": "Home",
            "url": "/",
            "content": (
                "Rohmad builds developer tools. He also writes notes about "
                "Nix and Neovim, and keeps dotfiles tidy."
            ),
        },
        {
            "id": "projects",
            "title": "Projects",
            "url": "/projects/",
            "content": (
                "Payment platform at Alto Network, processing 27 million "
                "records per day with a 99.99% SLA."
            ),
        },
        {
            "id": "blog",
            "title": "Blog",
            "url": "/blog/",
            "content": "Notes from the buffer about shipping software quietly.",
        },
    ]
    return retrieval.TfidfRetriever(docs)


def assert_window_boundaries(content: str, snippet: str) -> None:
    """Every snippet must start and end on a space boundary of the source."""
    body = snippet.strip("…")
    start = content.index(body)
    end = start + len(body)
    assert start == 0 or content[start - 1] == " "
    assert end == len(content) or content[end] == " "


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert retrieval.tokenize("Kubernetes, React.js & Go!") == [
        "kubernetes",
        "react",
        "js",
        "go",
    ]


def test_tokenize_drops_stopwords_and_possessive_leftovers() -> None:
    assert retrieval.tokenize("What is Rohmad's education?") == ["rohmad", "education"]


def test_expand_terms_adds_singular_form_for_long_plural_terms() -> None:
    assert retrieval.expand_terms(["projects", "go"]) == ["projects", "project", "go"]


def test_stopword_only_query_returns_no_hits() -> None:
    assert make_retriever().search("what is it") == []


def test_empty_retriever_returns_no_hits() -> None:
    assert retrieval.TfidfRetriever([]).search("anything") == []


def test_search_without_terms_returns_no_hits() -> None:
    assert make_retriever().search("!!! ???") == []


def test_search_returns_nothing_for_unrelated_question() -> None:
    assert make_retriever().search("zzzqqq123") == []


def test_search_respects_top_k() -> None:
    retriever = make_retriever()

    assert len(retriever.search("notes nix neovim alto", top_k=2)) == 2


def test_title_terms_outrank_body_only_matches() -> None:
    retriever = retrieval.TfidfRetriever(
        [
            {"id": "a", "title": "Projects", "url": "/projects/", "content": "payment"},
            {
                "id": "b",
                "title": "Blog",
                "url": "/blog/",
                "content": "projects projects projects notes",
            },
        ]
    )

    hits = retriever.search("projects", top_k=2)

    assert [hit["title"] for hit in hits] == ["Projects", "Blog"]


def test_heading_terms_picks_up_caps_and_markdown_headings() -> None:
    assert retrieval.heading_terms("SKILLS TypeScript\n## Education") == {
        "skills",
        "education",
    }


def test_heading_terms_ignores_ordinary_words() -> None:
    assert retrieval.heading_terms("Rohmad builds developer tools in Kediri") == set()


def test_a_section_heading_scores_higher_than_the_same_words_in_body_text() -> None:
    # Chunked resumes put the section heading inline ("EDUCATION University
    # of ..."). The heading marks what the chunk is about, so the same words
    # in a heading must outrank an identical chunk where they are prose.
    retriever = retrieval.TfidfRetriever(
        [
            {
                "id": "heading",
                "title": "Resume",
                "url": "/docs/resume.pdf",
                "content": "EDUCATION University of Brawijaya, Malang",
            },
            {
                "id": "prose",
                "title": "Resume",
                "url": "/docs/resume.pdf",
                "content": "Education University of Brawijaya, Malang",
            },
        ]
    )

    hits = retriever.search("education", top_k=2)

    assert hits[0]["score"] > hits[1]["score"]


def test_query_that_names_a_page_title_wins_over_a_passing_mention() -> None:
    # "What is his experience?" should land on the page titled Experience,
    # not on a short note that happens to say "developer experience" once.
    retriever = retrieval.TfidfRetriever(
        [
            {
                "id": "experience",
                "title": "Experience",
                "url": "/explore/",
                "content": (
                    "A seasoned career across logistics, insurtech, agritech, "
                    "fintech and developer platforms. Fullstack associate "
                    "manager at Alto Network: leading eleven engineers, "
                    "shipping internal dashboards and payment tools. Previously "
                    "senior engineer at AIA Indonesia, platform engineer at "
                    "eFishery, full-stack engineer at Jubelio, and React Native "
                    "developer at SiCepat Express."
                ),
            },
            {
                "id": "blog",
                "title": "Blog",
                "url": "/blog/",
                "content": "Notes on engineering, developer experience, and shipping.",
            },
        ]
    )

    hits = retriever.search("what is his experience", top_k=2)

    assert [hit["title"] for hit in hits] == ["Experience", "Blog"]


def test_title_coverage_does_not_boost_an_unrelated_page() -> None:
    retriever = retrieval.TfidfRetriever(
        [
            {
                "id": "experience",
                "title": "Experience",
                "url": "/explore/",
                "content": "A seasoned career across logistics and fintech platforms.",
            }
        ]
    )

    assert retriever.search("what is his background") == []


def test_plural_query_matches_singular_term_in_corpus() -> None:
    retriever = retrieval.TfidfRetriever(
        [{"id": "n", "title": "Journal", "url": "/journal/", "content": "a single note"}]
    )

    hits = retriever.search("notes")

    assert [hit["title"] for hit in hits] == ["Journal"]


def test_snippet_returns_whole_short_content_without_ellipsis() -> None:
    assert retrieval.TfidfRetriever._snippet("Short and sweet.", "short") == "Short and sweet."


def test_snippet_of_empty_content_is_empty() -> None:
    assert retrieval.TfidfRetriever._snippet("", "anything") == ""


def test_snippet_keeps_the_matched_term() -> None:
    content = (
        "A long introduction paragraph that pushes the interesting part of the "
        "text far enough along to require windowing. Kubernetes runs the platform."
    )

    snippet = retrieval.TfidfRetriever._snippet(content, "kubernetes", max_chars=80)

    assert "kubernetes" in snippet.lower()


def test_snippet_starts_and_ends_on_word_boundaries() -> None:
    content = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
        "xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )

    snippet = retrieval.TfidfRetriever._snippet(content, "omicron", max_chars=60)

    assert "omicron" in snippet
    assert_window_boundaries(content, snippet)


def test_snippet_marks_truncation_on_both_ends() -> None:
    content = ("prefix " * 20) + ("word " * 200)

    snippet = retrieval.TfidfRetriever._snippet(content, "word", max_chars=60)

    assert snippet.startswith("…")
    assert snippet.endswith("…")
    assert_window_boundaries(content, snippet)


def test_snippet_does_not_start_mid_word() -> None:
    content = (
        "Something before the window that is long enough to force a cut. "
        "Extraordinary Kubernetes platform."
    )

    snippet = retrieval.TfidfRetriever._snippet(content, "kubernetes", max_chars=50)

    first_word = snippet.lstrip("…").split(" ")[0]
    assert re.search(rf"(^|\s){re.escape(first_word)}($|\s)", content) is not None
    assert_window_boundaries(content, snippet)


def test_snippet_prefers_a_nearby_sentence_boundary() -> None:
    content = (
        "First sentence here. Second sentence that rambles on for a while "
        "before the match, mentioning Kubernetes and platforms."
    )

    snippet = retrieval.TfidfRetriever._snippet(content, "kubernetes", max_chars=140)

    assert snippet.startswith("Second sentence")
