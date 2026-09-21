"""Week 6 Workshop: Evaluation activity tasks."""

from __future__ import annotations

from typing import Sequence


def average_precision(relevance_at_rank: Sequence[int], total_relevant: int) -> float:
    """Compute Average Precision (AP) for one ranked list."""
    if total_relevant <= 0:
        return 0.0

    found_relevant = 0
    precision_sum = 0.0

    for rank, is_relevant in enumerate(relevance_at_rank, start=1):
        if is_relevant:
            # Record precision only at ranks where a relevant item is found.
            found_relevant += 1
            precision_sum += found_relevant / rank

    # Include known relevant items that were not retrieved in the denominator.
    return precision_sum / total_relevant
