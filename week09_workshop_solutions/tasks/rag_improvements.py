import numpy as np

from utils.retrieval_utils import dense_search, lookup_vectors


def dense_rerank_shortlist(
    bm25_shortlist: list[dict],
    query_vec: np.ndarray,
    document_ids: list[str],
    document_embeddings: np.ndarray,
    id_to_doc: dict[str, dict],
    top_k: int = 10,
) -> list[dict]:
    """Rerank the BM25 candidates using the same dense search as the notebook.

    Inputs:
        bm25_shortlist: Reviews returned by BM25, in rank order. For example:
            [{"rank": 1, "id": "R001", "score": 8.73, "content": "Slow service."}]
        query_vec: One query vector, stored as a NumPy array.
        document_ids: List of source IDs for the full collection.
        document_embeddings: Full vector matrix. Row i belongs to document_ids[i].
            The row position starts at 0 and is not a retrieval rank.
        id_to_doc: Source lookup, for example
            {"R001": {"id": "R001", "content": "Slow service."}}.
        top_k: Maximum number of results to return.

    Returns:
        A list containing only shortlisted reviews, sorted by cosine similarity:
        [{"rank": 1, "id": "R001", "score": 0.8, "content": "Slow service."}]
        id and content stay the same. score and rank reflect the new ordering.
    """
    # Find the shortlisted reviews' vectors in the full embedding store.
    candidate_vectors_by_id = lookup_vectors(bm25_shortlist, document_ids, document_embeddings)
    # Keep candidate IDs and vector rows in matching order.
    candidate_ids = [review["id"] for review in bm25_shortlist]
    candidate_vectors = np.asarray(
        [candidate_vectors_by_id[doc_id] for doc_id in candidate_ids], dtype=np.float32
    )
    return dense_search(
        query_vec, candidate_ids, candidate_vectors, id_to_doc, top_k=top_k,
        batch_size=128,
    )


# Shared classification rules. Each function below adds its own output format.
BASE_JUDGE_INSTRUCTIONS = """
You are a strict sentiment judge for coffee shop reviews.
Classify sentiment toward waiting time and service speed.
Allowed labels: positive, neutral, negative, non_relevant.
Use non_relevant when the review does not discuss waiting time or service speed.
Use neutral for a relevant review without clearly positive or negative sentiment.
Copy the new review's document id exactly into your answer.
""".strip()


def build_zero_shot_messages(review: dict) -> tuple[str, str]:
    """Build prompts from a review such as {"id": "R001", "content": "Slow service."}.

    Return (system_prompt, user_prompt), two strings. The user prompt contains
    only the document ID and text. Gold labels stay out of the prompt.
    This function makes no API call.
    """
    system_prompt = (
        BASE_JUDGE_INSTRUCTIONS
        + '\nReturn only JSON in this format: {"id": "<document_id>", "label": "negative"}'
    )
    user_prompt = f"Document id: {review['id']}\n\nReview:\n{review['content']}"
    return system_prompt, user_prompt


def build_few_shot_messages(review: dict, examples: list[dict]) -> tuple[str, str]:
    """Build a few-shot prompt for one new review.

    Inputs:
        review: {"id": "R001", "content": "Slow service."}.
        examples: [{"id": "EXAMPLE_1", "content": "Quick service.", "label": "positive"}].
    Return (system_prompt, user_prompt), two strings. Labelled examples come
    before the new review's ID and text. This function makes no API call.
    """
    system_prompt = (
        BASE_JUDGE_INSTRUCTIONS
        + '\nReturn only JSON in this format: {"id": "<document_id>", "label": "negative"}'
    )
    blocks = []
    # Each worked example shows the same ID in its input and expected answer.
    for example in examples:
        blocks.append(
            f"Example document id: {example['id']}\nReview:\n{example['content']}\n"
            f'Expected JSON:\n{{"id": "{example["id"]}", "label": "{example["label"]}"}}'
        )
    # The new review has an ID and text, but no supplied answer.
    blocks.append(f"Document id: {review['id']}\n\nReview:\n{review['content']}")
    return system_prompt, "\n\n".join(blocks)


def build_cot_messages(review: dict) -> tuple[str, str]:
    """Build a prompt requesting a short explanation before the JSON judgement.

    Input: {"id": "R001", "content": "Slow service."}. No examples are supplied.
    Return (system_prompt, user_prompt), two strings. This makes no API call.
    """
    # Start from the base rules, without the zero-shot JSON-only instruction.
    system_prompt = BASE_JUDGE_INSTRUCTIONS + """

Let's think step by step.
Explain the label in one or two sentences, starting with "Reasoning:".
Then output one JSON object on the final line.
Keep the explanation outside JSON.

Final JSON format:
{"id": "<document_id>", "label": "negative"}
"""
    user_prompt = f"Document id: {review['id']}\n\nReview:\n{review['content']}"
    return system_prompt.strip(), user_prompt
