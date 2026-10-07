"""Ask the help center a question from the command line.

    python -m ask_the_shop "Can I return a hat I wore once?"
    python -m ask_the_shop "Do you ship to Toronto?" --no-llm      # just show the retrieved sections
"""
import argparse

from .answer import DEFAULT_MODEL, answer_question
from .corpus import load_chunks
from .retrieval import RETRIEVER_NAMES, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Fernwood Supply help center a question.")
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid")
    parser.add_argument("-k", type=int, default=5, help="number of sections to retrieve")
    parser.add_argument("--no-llm", action="store_true", help="only show retrieved sections; no API key needed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    chunks = load_chunks()
    retriever = build_retriever(args.retriever, chunks)

    if args.no_llm:
        for rank, (chunk, score) in enumerate(retriever.search(args.question, k=args.k), start=1):
            print(f"{rank}. [{chunk.doc_id}] {chunk.title} > {chunk.heading}  (score {score:.3f})")
            print(f"   {chunk.text}\n")
        return

    import anthropic

    answer = answer_question(args.question, retriever, anthropic.Anthropic(), model=args.model, k=args.k)
    print(answer.text)
    titles = {c.doc_id: c.title for c in chunks}
    if answer.citations:
        print("\nSources:")
        for doc_id in answer.citations:
            print(f"  [{doc_id}] {titles.get(doc_id, 'unknown article')}")


if __name__ == "__main__":
    main()
