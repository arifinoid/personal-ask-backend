import json

import main


def write_corpus(tmp_path, docs: list[dict]):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(docs), encoding="utf-8")
    return path


def test_chunk_text_keeps_a_short_document_whole() -> None:
    assert main.chunk_text("Rohmad builds developer tools.") == ["Rohmad builds developer tools."]


def test_chunk_text_splits_on_line_boundaries() -> None:
    text = "\n".join(f"Line {i} of the document with some words." for i in range(40))

    chunks = main.chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "Line 0" in chunks[0]
    assert "Line 39" in chunks[-1]


def test_chunk_text_preserves_every_word() -> None:
    text = "\n".join(f"entry-{i}" for i in range(100))

    chunks = main.chunk_text(text, max_chars=80)

    assert " ".join(chunks).split() == text.split()


def test_chunk_text_ignores_blank_lines_and_split_oversized_lines() -> None:
    long_line = " ".join(["word"] * 120)

    chunks = main.chunk_text(f"\n\n{long_line}\n\n", max_chars=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_chunk_text_of_blank_text_is_empty() -> None:
    assert main.chunk_text("   \n\n \t") == []


def test_load_corpus_combines_site_pages_and_documents(tmp_path) -> None:
    corpus = write_corpus(
        tmp_path,
        [{"id": "home", "title": "Home", "url": "/", "content": "site page"}],
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "project_one.md").write_text("A project write-up.", encoding="utf-8")

    docs = main.load_corpus(corpus_path=corpus, docs_dir=docs_dir)

    assert [doc["id"] for doc in docs] == ["home", "file:project_one"]
    document = docs[1]
    assert document["title"] == "Project One"
    assert document["url"] == "/docs/project_one.md"
    assert document["content"] == "A project write-up."


def test_load_corpus_chunks_a_long_document(tmp_path) -> None:
    corpus = write_corpus(tmp_path, [])
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "resume.md").write_text(
        "\n".join(f"Resume line {i} with enough words to fill a chunk." for i in range(60)),
        encoding="utf-8",
    )

    docs = main.load_corpus(corpus_path=corpus, docs_dir=docs_dir)

    assert len(docs) > 1
    assert [doc["id"] for doc in docs] == [f"file:resume#{i}" for i in range(1, len(docs) + 1)]
    assert {doc["title"] for doc in docs} == {"Resume"}
    assert {doc["url"] for doc in docs} == {"/docs/resume.md"}


def test_load_corpus_skips_empty_and_unsupported_files(tmp_path) -> None:
    corpus = write_corpus(tmp_path, [])
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "empty.md").write_text("   \n\n", encoding="utf-8")
    (docs_dir / "notes.rst").write_text("unsupported extension", encoding="utf-8")

    assert main.load_corpus(corpus_path=corpus, docs_dir=docs_dir) == []


def test_load_corpus_without_docs_dir_reads_only_the_corpus(tmp_path) -> None:
    corpus = write_corpus(tmp_path, [{"id": "home", "title": "Home", "url": "/", "content": "x"}])

    docs = main.load_corpus(corpus_path=corpus, docs_dir=None)

    assert [doc["id"] for doc in docs] == ["home"]


def test_load_corpus_ignores_a_missing_corpus_file(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "note.txt").write_text("Only a document.", encoding="utf-8")

    docs = main.load_corpus(corpus_path=tmp_path / "nope.json", docs_dir=docs_dir)

    assert [doc["title"] for doc in docs] == ["Note"]
