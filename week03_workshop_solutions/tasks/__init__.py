"""Tutor reference functions for Week 3."""

from .text_representation import (
    compute_idf,
    tfidf_vectorize,
    mean_pool_glove,
    max_pool_glove,
    cosine_similarity,
)

__all__ = [
    "compute_idf",
    "tfidf_vectorize",
    "mean_pool_glove",
    "max_pool_glove",
    "cosine_similarity",
]
