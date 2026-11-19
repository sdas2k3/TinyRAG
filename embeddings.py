from functools import lru_cache
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = Path(__file__).resolve().parent / ".cache" / "models"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    """The local embedding model or tokenizer could not be loaded."""


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    options = {"cache_dir": MODEL_CACHE, "token": False}
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, **options)
    except OSError:
        pass
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, **options)
    except OSError as error:
        raise EmbeddingError("Could not load the BGE tokenizer. Check the model cache or connect to Hugging Face.") from error


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    options = {"device": "cpu", "cache_folder": str(MODEL_CACHE), "token": False}
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True, **options)
    except (OSError, ValueError):
        pass
    try:
        return SentenceTransformer(MODEL_NAME, **options)
    except (OSError, ValueError) as error:
        raise EmbeddingError("Could not load BGE Small. Check the model cache or connect to Hugging Face.") from error


@lru_cache(maxsize=1)
def passage_vectors(passages: tuple[str, ...]):
    """Reuse the batch while its exact text and order stay the same."""
    return get_model().encode(list(passages), normalize_embeddings=True, show_progress_bar=False)


def cosine_scores(question: str, passages: tuple[str, ...]) -> list[float]:
    vectors = passage_vectors(passages)
    query = get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    # The dot product of two unit vectors is their cosine similarity.
    return (vectors * query).sum(axis=1).tolist()
