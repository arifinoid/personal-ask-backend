"""Pure-Python TF-IDF retrieval over a static corpus.

No numpy / scikit-learn / sentence-transformers — deliberately boring so the
service stays small and dependency-light. Good enough for a handful of pages.

Title terms are boosted so short queries (e.g. "projects") match the page whose
title contains that word, and a page whose entire title appears in the query
("What is his experience?") is boosted further. Section headings get the same
treatment: a document that says "EDUCATION ..." should beat one that only
mentions "education" in passing. Query terms are expanded with their singular
form (stripping a trailing 's') so "project" matches "projects" etc.
"""

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Function words carry no signal for retrieval and, worse, make every document
# match: "What is Rohmad's education?" used to score pages on "what" and the
# leftover "s" from the possessive. Content terms are what should decide.
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as into than then
    is are was were be been being am do does did have has had
    i me my we our us you your he him his she her it its they them their
    this that these those there here
    what which who whom whose when where why how
    can could should would will shall may might must
    not no yes about tell show give please any some more most
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [term for term in _TOKEN_RE.findall(text.lower()) if _is_content_term(term)]


def _is_content_term(term: str) -> bool:
    # Single characters are possessive leftovers ("rohmad's" -> "s") or noise.
    return len(term) > 1 and term not in _STOPWORDS


def expand_terms(terms: list[str]) -> list[str]:
    """Return terms plus their singular forms for basic plural matching."""
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        if term.endswith("s") and len(term) > 3:
            expanded.append(term[:-1])
    return expanded


# Section headings say what a chunk is about: resumes write "EDUCATION" and
# "SKILLS", markdown writes "## Education". Chunked documents are joined with
# spaces, so the marker survives mid-string; match it there, not per line.
_MARKDOWN_HEADING_RE = re.compile(r"(?:^|\s)#{1,6}\s+([A-Za-z][\w-]*)")
_CAPS_HEADING_RE = re.compile(r"\b[A-Z][A-Z0-9&/.-]{2,}\b")


def heading_terms(content: str) -> set[str]:
    """Terms that appear as a heading or acronym in a document chunk."""
    terms = {match.group(1).lower() for match in _MARKDOWN_HEADING_RE.finditer(content)}
    terms.update(term.lower() for term in _CAPS_HEADING_RE.findall(content))
    return terms


class TfidfRetriever:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        # term -> number of docs containing it
        self.df: dict[str, int] = defaultdict(int)
        # doc index -> {term: normalized tfidf weight}
        self.doc_vectors: list[dict[str, float]] = []
        # doc index -> set of title terms (for boosting at query time)
        self.title_terms: list[set[str]] = []
        # doc index -> set of heading/acronym terms (same purpose)
        self.heading_terms: list[set[str]] = []
        self._build(docs)

    def _build(self, docs: list[dict]) -> None:
        doc_tfs: list[dict[str, int]] = []
        for doc in docs:
            title_terms = set(tokenize(doc.get("title", "")))
            self.title_terms.append(title_terms)
            self.heading_terms.append(heading_terms(str(doc.get("content", ""))))
            text = " ".join(
                str(doc.get(k, "")) for k in ("title", "content", "url")
            )
            tf = defaultdict(int)
            for term in tokenize(text):
                tf[term] += 1
            doc_tfs.append(tf)
            for term in tf:
                self.df[term] += 1

        n = len(docs)
        for tf in doc_tfs:
            vec: dict[str, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                idf = math.log((n + 1) / (self.df[term] + 1)) + 1.0
                weight = count * idf
                vec[term] = weight
                norm_sq += weight * weight
            norm = math.sqrt(norm_sq) or 1.0
            self.doc_vectors.append({t: w / norm for t, w in vec.items()})

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.docs:
            return []
        q_terms = expand_terms(tokenize(query))
        if not q_terms:
            return []
        q_tf = defaultdict(int)
        for term in q_terms:
            q_tf[term] += 1
        n = len(self.docs)
        q_vec: dict[str, float] = {}
        q_norm_sq = 0.0
        for term, count in q_tf.items():
            if term not in self.df:
                continue
            idf = math.log((n + 1) / (self.df[term] + 1)) + 1.0
            weight = count * idf
            q_vec[term] = weight
            q_norm_sq += weight * weight
        q_norm = math.sqrt(q_norm_sq) or 1.0
        q_vec = {t: w / q_norm for t, w in q_vec.items()}

        scores: list[tuple[float, int]] = []
        title_boost = 1.6
        heading_boost = 1.45
        # A page whose whole title appears in the query is the page being asked
        # about ("What is his experience?" -> /explore/, titled "Experience").
        # Without this, a short page that happens to mention the word once
        # outranks the page that is actually about it.
        covered_boost = 2.0
        query_terms = set(q_tf)
        for i, doc_vec in enumerate(self.doc_vectors):
            score = 0.0
            for qt, qw in q_vec.items():
                dw = doc_vec.get(qt, 0.0)
                if dw > 0:
                    score += qw * dw
                    # boost when the query term sits in the doc title
                    if qt in self.title_terms[i]:
                        score += qw * dw * (title_boost - 1.0)
                    # ... or in a section heading ("EDUCATION University of ...")
                    elif qt in self.heading_terms[i]:
                        score += qw * dw * (heading_boost - 1.0)
            if score > 0 and self.title_terms[i] and self.title_terms[i] <= query_terms:
                score *= covered_boost
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)

        results = []
        for score, i in scores[:top_k]:
            doc = self.docs[i]
            snippet = self._snippet(doc.get("content", ""), query)
            results.append(
                {
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "snippet": snippet,
                    "score": round(score, 4),
                }
            )
        return results

    @staticmethod
    def _snippet(content: str, query: str, max_chars: int = 220) -> str:
        terms = set(tokenize(query))
        if not content:
            return ""
        lower = content.lower()
        best_idx = -1
        for term in terms:
            idx = lower.find(term)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
        # Keep some context before the match. Scaling it to the window keeps the
        # matched term inside even when callers ask for a short snippet.
        context = min(60, max_chars // 3)
        start = max(0, best_idx - context) if best_idx != -1 else 0
        # Snap the window to word boundaries so snippets never start or end
        # mid-word ("econd. Software engineer...").
        sentence_start = False
        if start > 0:
            limit = min(start + 40, best_idx) if best_idx != -1 else start + 40
            next_space = content.find(" ", start, limit)
            if next_space != -1:
                start = next_space + 1
            # Prefer a nearby sentence boundary so the snippet reads from the
            # start of a sentence instead of the middle of one.
            lookback = min(100, max_chars // 3)
            boundary = max(
                content.rfind(". ", max(0, start - lookback), start),
                content.rfind(": ", max(0, start - lookback), start),
            )
            if boundary != -1:
                start = boundary + 2
                sentence_start = True
        end = start + max_chars
        if end < len(content):
            last_space = content.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        snippet = content[start:end]
        if start > 0 and not sentence_start:
            snippet = "…" + snippet
        if end < len(content):
            snippet = snippet + "…"
        return snippet.strip()
