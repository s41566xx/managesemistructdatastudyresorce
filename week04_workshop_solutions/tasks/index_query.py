"""Week 4 Workshop: Indexing and Query Processing."""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "build_tiny_term_index",
    "save_index",
    "load_index",
    "lookup_posting_list",
    "eval_single_operator_query",
    "eval_postfix_and_or",
    "post_filter_wildcard_candidates",
]


def build_tiny_term_index(
    tokenised_docs: list[list[str]],
    doc_ids: list[int],
) -> dict[str, list[int]]:
    if len(tokenised_docs) != len(doc_ids):
        raise ValueError("tokenised_docs and doc_ids must have the same length")

    tmp: dict[str, set[int]] = {}
    # A document appears once in a posting list, even when the term repeats.
    for doc_id, tokens in zip(doc_ids, tokenised_docs):
        for term in set(tokens):
            tmp.setdefault(term, set()).add(doc_id)

    return {term: sorted(ids) for term, ids in sorted(tmp.items())}


def save_index(index: dict[str, list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def load_index(path: str | Path) -> dict[str, list[int]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {term: list(doc_ids) for term, doc_ids in data.items()}


def lookup_posting_list(term: str, index: dict[str, list[int]]) -> list[int]:
    return list(index.get(term, []))


def eval_single_operator_query(
    left_term: str,
    operator: str,
    right_term: str,
    index: dict[str, list[int]],
) -> set[int]:
    left = set(lookup_posting_list(left_term, index))
    right = set(lookup_posting_list(right_term, index))

    if operator == "AND":
        return left & right
    if operator == "OR":
        return left | right
    if operator == "AND NOT":
        return left - right
    raise ValueError("Workshop warm-up supports AND, OR, and AND NOT only")


def eval_postfix_and_or(
    postfix_tokens: list[str],
    index: dict[str, list[int]],
) -> set[int]:
    stack: list[set[int]] = []

    # Terms push posting sets; each operator replaces two sets with one result.
    for token in postfix_tokens:
        op = token.upper()

        if op in {"AND", "OR"}:
            right = stack.pop()
            left = stack.pop()
            if op == "AND":
                stack.append(left & right)
            else:
                stack.append(left | right)
        else:
            stack.append(set(lookup_posting_list(token, index)))

    return stack.pop()


def post_filter_wildcard_candidates(pattern: str, candidates: list[str]) -> list[str]:
    # Candidate generation is complete; this step only applies the wildcard test.
    if pattern.endswith("*") and not pattern.startswith("*"):
        prefix = pattern[:-1]
        return sorted(term for term in candidates if term.startswith(prefix))
    if pattern.startswith("*") and not pattern.endswith("*"):
        suffix = pattern[1:]
        return sorted(term for term in candidates if term.endswith(suffix))
    raise ValueError("Workshop supports prefix or suffix wildcard patterns only")
