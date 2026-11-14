import main


def test_clean_snippet_collapses_whitespace() -> None:
    assert main._clean_snippet("alpha\n\n  beta\t gamma") == "alpha beta gamma"


def test_clean_snippet_decodes_html_entities() -> None:
    assert main._clean_snippet("Tom &amp; Jerry &lt;tag&gt;&nbsp;here") == "Tom & Jerry <tag> here"


def test_clean_snippet_removes_code_artifacts() -> None:
    assert main._clean_snippet("export const metadata export default function") == ""


def test_clean_snippet_drops_a_partial_leading_sentence() -> None:
    text = "…fragment of a sentence. A complete sentence starts here and continues."

    assert main._clean_snippet(text) == "A complete sentence starts here and continues."


def test_clean_snippet_keeps_a_leading_partial_when_it_is_most_of_the_text() -> None:
    text = "…a partial sentence that takes up nearly the whole snippet. End."

    assert main._clean_snippet(text).startswith("a partial sentence")


def test_clean_snippet_removes_a_trailing_ellipsis() -> None:
    assert main._clean_snippet("a clean sentence…") == "a clean sentence"


def test_clean_snippet_truncates_long_text_on_a_word_boundary() -> None:
    text = " ".join(["word"] * 80)

    cleaned = main._clean_snippet(text)

    assert cleaned.endswith("...")
    assert len(cleaned) <= 203
    assert cleaned[: -3].split(" ")[-1] == "word"


def test_clean_snippet_leaves_empty_text_empty() -> None:
    assert main._clean_snippet("") == ""
