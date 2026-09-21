from __future__ import annotations

import json

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}


def build_system_prompt() -> str:
    """
    Build the system prompt for sentiment classification.

    As in the lecture: the system message carries the stable instructions
    (role, allowed labels, output format). The comment itself is sent
    separately as the user message.

    Expected model output:
        {"sentiment": "neutral"}

    Requirements:
    - exactly one JSON object
    - exactly one key: sentiment
    - sentiment must be one of: positive, neutral, negative
    - no markdown
    - no explanation
    - no extra keys
    """
    return """
You are a strict sentiment classifier.

Classify the comment in the user message as exactly one of:
positive, neutral, negative

Return only one valid JSON object in this exact format:
{"sentiment": "neutral"}

Rules:
- The JSON object must have exactly one key: sentiment.
- The sentiment value must be exactly one of: positive, neutral, negative.
- Do not include markdown.
- Do not include explanation.
- Do not include extra keys.
""".strip()

def extract_first_json_object(text: str) -> dict:
    """
    Extract the first complete JSON object from model output.

    The text argument is raw model output. JSONDecoder.raw_decode(...) parses
    one complete JSON value and stops before any later explanation or object.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON object in model output.") from exc
    return obj


def validate_sentiment_json(obj: dict) -> str:
    """
    Validate a parsed JSON object and return the sentiment label.

    The obj argument is the parsed Python dictionary. This checks format
    validity, not whether the model's label is semantically correct.
    """
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    if set(obj.keys()) != {"sentiment"}:
        raise ValueError('JSON must have exactly one key: "sentiment".')

    label = obj["sentiment"]

    if label not in ALLOWED_SENTIMENTS:
        raise ValueError(f"Invalid sentiment label: {label}")

    return label
