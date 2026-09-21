from pathlib import Path
import re
from typing import Any

import numpy as np
from openai import OpenAI
from whoosh import scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import FileIndex, create_in, open_dir
from whoosh.qparser import SimpleParser


Document = dict[str, str]
SearchResult = dict[str, str | int | float]


def build_whoosh_index(documents: list[Document], index_dir: str = "indexdir") -> FileIndex:
    """Build a searchable Whoosh index from a list of document dictionaries.

    documents example: [{"id": "R001", "content": "Slow service."}]
    index_dir is the folder in which to save the index.
    Return a Whoosh index object to pass to bm25_search(...).
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # ID keeps the source identifier as one exact value. TEXT creates searchable
    # tokens using stemming. stored=True also keeps the original values for hits.
    schema = Schema(
        id=ID(stored=True, unique=True),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

    ix = create_in(index_path, schema)
    writer = ix.writer()

    for doc in documents:
        writer.add_document(id=doc["id"], content=doc["content"])

    writer.commit()
    return open_dir(index_path)


def bm25_search(ix: FileIndex, query_text: str, top_k: int = 10) -> list[SearchResult]:
    """Search the Whoosh index and return up to top_k reviews, best first.

    Inputs:
        ix: Index returned by build_whoosh_index(...).
        query_text: A string, for example "waiting time at Aromas".
        top_k: Maximum number of results.

    Returns a list of dictionaries in this format:
        [{"rank": 1, "id": "R001", "score": 8.73, "content": "Slow service."}]
    rank starts at 1. id is the source document ID. score is the BM25 score.
    """
    # This path accepts ordinary natural-language queries, not query syntax.
    # Replace punctuation with spaces before SimpleParser combines terms with OR.
    clean_query = re.sub(r"[\W_]+", " ", query_text).strip()
    parser = SimpleParser("content", schema=ix.schema)
    query = parser.parse(clean_query)

    rows = []
    # Week 5 k1 and b, written explicitly with Whoosh's default values.
    ranker = scoring.BM25F(
        K1=1.2,  # Higher: repeated terms keep adding value for longer.
        B=0.75,  # Higher: stronger document-length normalisation.
    )
    # Lower K1 makes the extra gain from repetitions level off sooner.
    # B=0 ignores length. B=1 applies full length normalisation.
    # Change one value here, then rerun the search using the same index.
    with ix.searcher(weighting=ranker) as searcher:
        results = searcher.search(query, limit=top_k)
        for rank, hit in enumerate(results, start=1):
            rows.append({
                "rank": rank,
                "id": hit["id"],
                "score": float(hit.score),
                "content": hit["content"],
            })
    return rows


def embed_query(
    client: OpenAI,
    query_text: str,
    model: str = "text-embedding-3-small",
) -> tuple[np.ndarray, Any]:
    """Generate one query embedding using the course API.

    query_text is a string. model must match the supplied document vectors.
    Return (query_vec, usage). query_vec is a NumPy array with one
    value per embedding dimension. usage is the API token-usage object.
    """
    # The API accepts a list of inputs. This list contains one query, so data[0]
    # holds its vector. Use the model recorded in the saved document metadata.
    response = client.embeddings.create(model=model, input=[query_text])
    query_vec = np.asarray(response.data[0].embedding, dtype=np.float32)
    return query_vec, response.usage
