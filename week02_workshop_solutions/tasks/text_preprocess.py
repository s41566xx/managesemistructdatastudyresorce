"""IR Week 2 Workshop - Text preprocessing (reference solution)

This file contains one possible solution for the workshop preprocessing tasks.

Design goals:
- Robust to noisy HTML snippets (tags/scripts/styles/emojis/entities)
- Deterministic (same input → same tokens)
- Keeps token order

"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import List

from bs4 import BeautifulSoup
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import wordpunct_tokenize

__all__ = ["preprocess"]

_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;")

# Optional lemmatiser
try:
    from nltk.stem import WordNetLemmatizer
    _LEMMATIZER = WordNetLemmatizer()
except Exception:  # pragma: no cover
    _LEMMATIZER = None


def _strip_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    # separator avoids joining words across tags
    return soup.get_text(separator=" ")


def _normalise(text: str) -> str:
    text = html.unescape(text)                 # &amp; → &, &quot; → "
    text = _ENTITY_PATTERN.sub(" ", text)      # drop unknown entities
    text = unicodedata.normalize("NFKC", text) # canonical forms
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    toks = wordpunct_tokenize(text)
    toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
    return toks


def _remove_stopwords(tokens: List[str]) -> List[str]:
    sw = set(nltk_stopwords.words("english"))
    return [t for t in tokens if t not in sw]


def _lemmatize(tokens: List[str]) -> List[str]:
    if _LEMMATIZER is None:
        return tokens
    try:
        return [_LEMMATIZER.lemmatize(t) for t in tokens]
    except LookupError:
        # WordNet resource not downloaded. Skip lemmatisation gracefully.
        return tokens



def preprocess(text: str, variant: str = "raw") -> List[str]:
    variant = variant.strip().lower()

    if variant == "raw":
        return text.lower().split()

    clean_text = _normalise(_strip_html(text))
    tokens = _tokenize(clean_text)

    if variant == "basic_clean":
        return tokens

    if variant == "basic_clean_stop":
        return _remove_stopwords(tokens)

    if variant == "basic_clean_stop_lemma":
        tokens = _remove_stopwords(tokens)
        return _lemmatize(tokens)

    raise ValueError(f"Unknown variant: {variant}")
