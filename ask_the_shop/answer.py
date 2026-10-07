"""Answer a question with Claude, grounded in the retrieved help center sections.

The prompt gives Claude only the top k sections, asks it to cite each fact with
the article id in square brackets, and to say a fixed sentence when the sources
don't contain the answer. The fixed sentence makes abstentions easy to detect
and score in the evaluation.
"""
import re
from dataclasses import dataclass

from .corpus import Chunk

DEFAULT_MODEL = "claude-haiku-5-5"
ABSTAIN_TEXT = "I don't know based on our help center."
SUPPORT_EMAIL = "help@fernwoodsupply.example"

SYSTEM_PROMPT = f"""You are the customer support assistant for Fernwood Supply, a small online apparel shop.
Answer the customer's question using only the help center sources you are given.

Rules:
1. Cite every fact with its source id in square brackets, for example [returns].
2. Answer in 1 to 3 short, friendly sentences.
3. If the sources don't answer the question, reply with exactly "{ABSTAIN_TEXT}" and then suggest emailing {SUPPORT_EMAIL}. Don't guess.
4. Never invent prices, dates, or policies that aren't in the sources."""

CITATION_RE = re.compile(r"\[([a-z0-9-]+)\]")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Chunk]   # the sections Claude was given
    citations: list[str]   # article ids Claude cited, in order
    abstained: bool


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f'<source id="{c.doc_id}" title="{c.title}" section="{c.heading}">\n{c.text}\n</source>'
        for c in chunks
    )


def build_user_message(question: str, chunks: list[Chunk]) -> str:
    return f"<sources>\n{format_sources(chunks)}\n</sources>\n\nCustomer question: {question}"


def parse_citations(text: str) -> list[str]:
    """Unique article ids cited in the answer, in the order they first appear."""
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def is_abstention(text: str) -> bool:
    normalized = text.replace("’", "'").lower()
    return ABSTAIN_TEXT.lower() in normalized


def answer_question(question: str, retriever, client, model: str = DEFAULT_MODEL, k: int = 5) -> Answer:
    """Retrieve the top k sections, then ask Claude to answer from them.

    `client` is an anthropic.Anthropic() instance, or any object with the same
    messages.create() method (the tests pass in a fake one).
    """
    chunks = [chunk for chunk, _ in retriever.search(question, k=k)]
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text").strip()
    return Answer(question, text, chunks, parse_citations(text), is_abstention(text))
