"""Week 3 Workshop: Text representations."""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_vocab",
    "bow_vectorize",
    "compute_idf",
    "tfidf_vectorize",
    "mean_pool_glove",
    "max_pool_glove",
    "cosine_similarity",
]


def build_vocab(tokenised_docs: list[list[str]]) -> dict[str, int]:
    """Build a deterministic vocabulary dictionary from tokenised documents.

    The dictionary maps token -> column index.
    Terms are sorted so the same corpus always gives the same mapping.
    """
    terms = sorted({token for doc in tokenised_docs for token in doc})
    return {term: idx for idx, term in enumerate(terms)}


def bow_vectorize(tokenised_docs: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """Create a raw-count Bag-of-Words matrix.

    Output shape: (number_of_documents, number_of_vocab_terms).
    """
    X = np.zeros((len(tokenised_docs), len(vocab)), dtype=np.float32)

    for row, tokens in enumerate(tokenised_docs):
        for token in tokens:
            if token in vocab:
                X[row, vocab[token]] += 1.0

    return X


def compute_idf(tokenised_docs: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    """Compute IDF values aligned with the vocabulary dictionary.

    Formula: idf(t) = log(N / df(t)).
    Output shape: (len(vocab),).
    """
    n_docs = len(tokenised_docs)
    df = np.zeros(len(vocab), dtype=np.float32)

    for tokens in tokenised_docs:
        for token in set(tokens):
            if token in vocab:
                df[vocab[token]] += 1.0

    if np.any(df == 0):
        raise ValueError("Every vocabulary term must appear in at least one document.")

    return np.log(n_docs / df).astype(np.float32)


def tfidf_vectorize(
    tokenised_docs: list[list[str]],
    vocab: dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """Build a length-normalised TF-IDF matrix.

    Formula:
        tf(t, d) = count(t, d) / |d|
        w(t, d) = tf(t, d) * idf(t)

    Output shape: (len(tokenised_docs), len(vocab)).
    Cosine similarity later handles vector-length normalisation.
    """
    X = np.zeros((len(tokenised_docs), len(vocab)), dtype=np.float32)

    for row, tokens in enumerate(tokenised_docs):
        if not tokens:
            continue

        for token in tokens:
            if token in vocab:
                X[row, vocab[token]] += 1.0

        X[row] = (X[row] / float(len(tokens))) * idf

    return X


def mean_pool_glove(
    tokens: list[str],
    glove: dict[str, np.ndarray],
    dim: int = 50,
) -> np.ndarray:
    """Average word vectors into one document vector.

    Output shape: (dim,).
    """
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    unk = glove.get("<unk>", np.zeros(dim, dtype=np.float32))
    vectors = [glove.get(token, unk) for token in tokens]
    return np.mean(vectors, axis=0).astype(np.float32)


def max_pool_glove(
    tokens: list[str],
    glove: dict[str, np.ndarray],
    dim: int = 50,
) -> np.ndarray:
    """Element-wise max pooling over word vectors.

    Output shape: (dim,).
    """
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    unk = glove.get("<unk>", np.zeros(dim, dtype=np.float32))
    vectors = np.vstack([glove.get(token, unk) for token in tokens])
    return np.max(vectors, axis=0).astype(np.float32)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using NumPy."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    norm_x = float(np.linalg.norm(x))
    norm_y = float(np.linalg.norm(y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return float(np.dot(x, y) / (norm_x * norm_y))
