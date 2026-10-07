"""Load the help center articles and split them into chunks.

Each article is a Markdown file in docs/. The file name (without .md) is the
article id that the assistant cites, and every "## " section becomes one chunk.
Sections are short and each covers one topic, which keeps retrieval precise.
"""
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@dataclass(frozen=True)
class Chunk:
    doc_id: str   # article id, e.g. "returns"
    title: str    # article title, e.g. "Returns"
    heading: str  # section heading, e.g. "Return window and condition"
    text: str     # section body

    @property
    def search_text(self) -> str:
        """Text the retrievers index. The title and heading carry a lot of meaning."""
        return f"{self.title}. {self.heading}. {self.text}"


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        doc_id = path.stem
        title, heading, body = "", "", []

        def flush() -> None:
            text = " ".join(line.strip() for line in body if line.strip())
            if heading and text:
                chunks.append(Chunk(doc_id, title, heading, text))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                heading, body = line[3:].strip(), []
            else:
                body.append(line)
        flush()
    return chunks
