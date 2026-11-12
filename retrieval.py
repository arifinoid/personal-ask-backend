"""Pure-Python TF-IDF retrieval over a static corpus.

No numpy / scikit-learn / sentence-transformers — deliberately boring so the
service stays small and dependency-light. Good enough for a handful of pages.
"""

import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfRetriever:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        # term -> set of doc indices containing it
        self.df: dict[str, int] = defaultdict(int)
        # doc index -> {term: tfidf weight}
        self.doc_vectors: list[dict[str, float]] = []
        self._build(docs)

    def _build(self, docs: list[dict]) -> None:
        # raw term frequency per doc
        doc_tfs: list[dict[str, int]] = []
        for doc in docs:
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
        q_tf = defaultdict(int)
        for term in tokenize(query):
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
        for i, doc_vec in enumerate(self.doc_vectors):
            score = sum(q_vec[t] * doc_vec.get(t, 0.0) for t in q_vec)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)

        results = []
        for score, i in scores[:top_k]:
            doc = self.docs[i]
            content = doc.get("content", "")
            snippet = self._snippet(content, query)
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
        # find the first occurrence of any query term
        lower = content.lower()
        best_idx = -1
        for term in terms:
            idx = lower.find(term)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
        start = max(0, best_idx - 60) if best_idx != -1 else 0
        snippet = content[start : start + max_chars]
        if start > 0:
            snippet = "…" + snippet
        if start + max_chars < len(content):
            snippet = snippet + "…"
        return snippet.strip()
