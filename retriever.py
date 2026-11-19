import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from embeddings import EmbeddingError, cosine_scores
from loader import Chunk, load_chunks, search_text


STOP_WORDS = set(
    "a an and are at be by can do does for from how i if in is it me much my "
    "of on or per that the this to was we what when where which who will with "
    "you your".split()
)
SEARCH_MODES = ("bm25", "semantic", "hybrid")
RESULT_LIMIT = 3
SEMANTIC_MIN_SCORE = 0.60
RRF_K = 60


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    semantic_rank: Optional[int]
    cosine: Optional[float]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold().replace("’", "'"))
    return [word for word in words if word not in STOP_WORDS]


def bm25_scores(question: str, passages: tuple[str, ...]) -> list[float]:
    """Score exact matches using rarity, term frequency, and passage length."""
    term_counts = [Counter(tokenize(passage)) for passage in passages]
    lengths = [sum(counts.values()) for counts in term_counts]
    average_length = sum(lengths) / len(lengths) if lengths else 0
    if not average_length:
        return [0.0] * len(passages)

    k1, b = 1.2, 0.75
    document_frequency = Counter(term for counts in term_counts for term in counts)
    rarity = {
        term: math.log1p((len(passages) - document_frequency[term] + 0.5) /
                        (document_frequency[term] + 0.5))
        for term in dict.fromkeys(tokenize(question))
        if term in document_frequency
    }

    scores = []
    for counts, length in zip(term_counts, lengths):
        normalization = k1 * (1 - b + b * length / average_length)
        score = sum(
            weight * counts[term] * (k1 + 1) / (counts[term] + normalization)
            for term, weight in rarity.items()
        )
        scores.append(score)
    return scores


def rank_scores(scores: list[float], minimum: float) -> dict[int, int]:
    eligible = (index for index, score in enumerate(scores) if score > minimum)
    ordered = sorted(eligible, key=scores.__getitem__, reverse=True)
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def retrieve(question: str, chunks: list[Chunk], mode: str = "hybrid") -> list[SearchHit]:
    """Return up to three passages, identified by their position in the corpus."""
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unknown search mode: {mode}")
    if not chunks or not any(character.isalnum() for character in question):
        return []

    passages = tuple(search_text(chunk) for chunk in chunks)
    lexical_scores = bm25_scores(question, passages) if mode != "semantic" else []
    semantic_scores = cosine_scores(question, passages) if mode != "bm25" else []
    lexical_ranks = rank_scores(lexical_scores, 0.0)
    semantic_ranks = rank_scores(semantic_scores, SEMANTIC_MIN_SCORE)

    hits = []
    for index in sorted(lexical_ranks.keys() | semantic_ranks.keys()):
        bm25_rank = lexical_ranks.get(index)
        semantic_rank = semantic_ranks.get(index)
        bm25_score = lexical_scores[index] if lexical_scores else None
        cosine = semantic_scores[index] if semantic_scores else None
        if mode == "hybrid":
            score = sum(1 / (RRF_K + rank) for rank in (bm25_rank, semantic_rank) if rank is not None)
        elif mode == "bm25":
            score = lexical_scores[index]
        else:
            score = semantic_scores[index]
        hits.append(SearchHit(chunks[index], score, bm25_rank, bm25_score, semantic_rank, cosine))

    # With equal fusion scores, prefer the better eligible semantic rank.
    hits.sort(key=lambda hit: (-hit.score, hit.semantic_rank or math.inf))
    return hits[:RESULT_LIMIT]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find relevant TinyRAG passages.")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    args = parser.parse_args()

    docs_dir = Path(__file__).resolve().parent / "docs"
    try:
        matches = retrieve(args.question, load_chunks(docs_dir), mode=args.mode)
    except EmbeddingError as error:
        parser.exit(1, f"{error}\n")

    if not matches:
        print("No matching passages found.")
    for rank, hit in enumerate(matches, start=1):
        bm25_score = f"{hit.bm25_score:.3f}" if hit.bm25_score is not None else "—"
        cosine = f"{hit.cosine:.3f}" if hit.cosine is not None else "—"
        print(f"#{rank} {hit.chunk.filename} / {hit.chunk.heading} ({args.mode} score {hit.score:.4f})")
        print(f"BM25 rank {hit.bm25_rank or '—'} · BM25 score {bm25_score} · "
              f"Semantic rank {hit.semantic_rank or '—'} · Cosine {cosine}")
        print(f"{hit.chunk.text}\n")


if __name__ == "__main__":
    main()
