from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Sequence, Tuple

from nltk.corpus import wordnet as wn


def get_tf(term: str, doc_id: int, index_pkg: Dict[str, Any]) -> int:
    """Return f(term, doc_id): the number of times term appears in doc_id."""
    postings = index_pkg.get("inverted", {}).get(term, {}).get("postings", {})
    return int(postings.get(doc_id, {}).get("tf", 0))


def get_df(term: str, index_pkg: Dict[str, Any]) -> int:
    """Return df(term): the number of indexed documents that contain term."""
    term_entry = index_pkg.get("inverted", {}).get(term)
    if term_entry is None:
        return 0
    return int(term_entry.get("df", 0))


def get_doc_length(doc_id: int, index_pkg: Dict[str, Any]) -> int:
    """Return |D|, the token length of doc_id."""
    return int(index_pkg["__META__"]["doc_lengths"][doc_id])


def get_collection_stats(index_pkg: Dict[str, Any]) -> Tuple[int, float]:
    """Return (N, avgdl) from index metadata."""
    meta = index_pkg["__META__"]
    return int(meta["N"]), float(meta["avgdl"])


def bm25_idf(term: str, index_pkg: Dict[str, Any]) -> float:
    """Compute BM25 IDF for term using the Week 5 slide formula."""
    N, _ = get_collection_stats(index_pkg)
    df = get_df(term, index_pkg)
    if df <= 0:
        return 0.0
    return float(np.log(1.0 + (N - df + 0.5) / (df + 0.5)))


def bm25_term_score(
    term: str,
    doc_id: int,
    index_pkg: Dict[str, Any],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute one BM25 term contribution for one document."""
    tf = get_tf(term, doc_id, index_pkg)
    if tf == 0:
        # A query term absent from this document contributes zero.
        return 0.0

    _, avgdl = get_collection_stats(index_pkg)
    doc_len = get_doc_length(doc_id, index_pkg)
    idf = bm25_idf(term, index_pkg)
    # This combines term-frequency saturation with document-length normalisation.
    denominator = tf + k1 * (1.0 - b + b * doc_len / avgdl)
    return idf * (tf * (k1 + 1.0)) / denominator


def bm25_score(
    query_terms: Sequence[str],
    doc_id: int,
    index_pkg: Dict[str, Any],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute BM25(D, Q) by summing term contributions over query terms."""
    # The document score is the sum of its contribution from each query term.
    return sum(bm25_term_score(term, doc_id, index_pkg, k1=k1, b=b) for term in query_terms)


def wordnet_candidates(term: str, max_synsets: int = 1, max_candidates: int = 6) -> List[str]:
    """Return candidate expansion terms from NLTK WordNet."""
    candidates: List[str] = []

    for synset in wn.synsets(term, pos=wn.NOUN)[:max_synsets]:
        for lemma in synset.lemmas():
            word = lemma.name().replace("_", " ").lower()
            if word != term.lower() and word not in candidates:
                candidates.append(word)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def expand_query_wordnet(
    query_terms: Sequence[str],
    max_terms_per_word: int = 2,
) -> List[str]:
    """Optional extension: return a small WordNet-assisted expansion."""
    # Preserve original terms and cap additions to reduce uncontrolled query drift.
    expanded: List[str] = list(query_terms)

    for term in query_terms:
        candidates = wordnet_candidates(term)
        for candidate in candidates[:max_terms_per_word]:
            if candidate not in expanded:
                expanded.append(candidate)

    return expanded
