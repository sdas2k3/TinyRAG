import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from embeddings import get_tokenizer


MAX_CHUNK_TOKENS = 300
MAX_HEADING_TOKENS = 80
CHUNK_BOUNDARIES = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)


@dataclass(frozen=True)
class Chunk:
    filename: str
    heading: str
    text: str


@lru_cache(maxsize=256)
def _indexed_heading(heading: str) -> str:
    # Preserve the full heading as metadata even for unusually long titles.
    offsets = get_tokenizer()(
        heading, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    if len(offsets["input_ids"]) > MAX_HEADING_TOKENS:
        return heading[:offsets["offset_mapping"][MAX_HEADING_TOKENS - 1][1]]
    return heading


def search_text(chunk: Chunk) -> str:
    return f"{_indexed_heading(chunk.heading)}\n{chunk.text}".strip()


def split_section(text: str, heading: str) -> list[str]:
    tokenizer = get_tokenizer()
    heading = _indexed_heading(heading)

    def fits(body: str) -> bool:
        tokens = tokenizer(f"{heading}\n{body}", verbose=False)["input_ids"]
        return len(tokens) <= MAX_CHUNK_TOKENS

    def split(body: str, level: int) -> list[str]:
        if fits(body):
            return [body]
        if level == len(CHUNK_BOUNDARIES):
            # A single long string can still exceed the budget without whitespace.
            middle = len(body) // 2
            return split(body[:middle], level) + split(body[middle:], level)

        boundary = CHUNK_BOUNDARIES[level]
        parts = []
        previous = 0
        for match in boundary.finditer(body):
            parts.append(body[previous:match.end()])
            previous = match.end()
        parts.append(body[previous:])

        pieces = []
        current = ""
        for part in parts:
            if current and not fits(current + part):
                pieces.extend(split(current, level + 1))
                current = ""
            current += part
        if current:
            pieces.extend(split(current, level + 1))
        return pieces

    return [piece.strip() for piece in split(text, 0) if piece.strip()]


def chunk_document(filename: str, markdown: str) -> list[Chunk]:
    """Keep each level-two Markdown section together with its source."""
    chunks: list[Chunk] = []
    heading = ""
    section_lines: list[str] = []

    def save_section() -> None:
        text = "\n".join(section_lines).strip()
        if text:
            chunks.extend(Chunk(filename, heading, piece) for piece in split_section(text, heading))

    for line in markdown.splitlines():
        if line.startswith("## "):
            save_section()
            heading = line[3:].strip()
            section_lines.clear()
        elif line.startswith("# ") and not heading and not section_lines:
            heading = line[2:].strip()
        else:
            section_lines.append(line)

    save_section()
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        chunks.extend(chunk_document(path.name, markdown))
    return chunks


if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent / "docs"
    for chunk in load_chunks(docs_dir):
        print(f"{chunk.filename} / {chunk.heading}")
        print(f"{chunk.text}\n")
