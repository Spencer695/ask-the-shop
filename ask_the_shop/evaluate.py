"""Score the retrievers, and optionally Claude's answers, on the labeled questions.

    python -m ask_the_shop.evaluate              # retrieval only, no API key needed
    python -m ask_the_shop.evaluate --answers    # also scores Claude's answers (needs ANTHROPIC_API_KEY)

Each line of eval/questions.jsonl has a question, the article ids that answer it
("gold"), and key facts a correct answer must mention. Questions with no gold
articles aren't covered by the help center, so the right answer is "I don't know".
Results are printed and written to eval/results.md.
"""
import argparse
import json
import re
from pathlib import Path

from .answer import DEFAULT_MODEL, answer_question
from .corpus import load_chunks
from .retrieval import RETRIEVER_NAMES, build_retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.md"
RETRIEVERS = RETRIEVER_NAMES


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- Retrieval ----------

def ranked_docs(retriever, question: str) -> list[str]:
    """Article ids in rank order. An article's position is the rank of its best section."""
    results = retriever.search(question, k=10_000)
    return list(dict.fromkeys(chunk.doc_id for chunk, _ in results))


def retrieval_metrics(retriever, questions: list[dict], context_k: int = 5) -> dict:
    """Recall@1 and Recall@3: share of questions with a gold article in the top 1 or 3 articles.
    MRR: average of 1 / (rank of the first gold article).
    Context hit: share of questions where a gold article is in the top k sections sent to Claude."""
    answerable = [q for q in questions if q["gold"]]
    r1 = r3 = rr = context = 0.0
    misses = []
    for q in answerable:
        docs = ranked_docs(retriever, q["question"])
        rank = next((i for i, d in enumerate(docs, start=1) if d in q["gold"]), None)
        r1 += rank == 1
        r3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        top_sections = retriever.search(q["question"], k=context_k)
        context += any(chunk.doc_id in q["gold"] for chunk, _ in top_sections)
        if rank != 1:
            misses.append({"id": q["id"], "question": q["question"], "top": docs[0], "gold_rank": rank})
    n = len(answerable)
    return {"n": n, "recall@1": r1 / n, "recall@3": r3 / n, "mrr": rr / n, "context": context / n, "misses": misses}


# ---------- Answers ----------

def normalize(text: str) -> str:
    """Lowercase, straighten apostrophes, and turn "3-7" or "3–7" into "3 to 7"."""
    text = text.lower().replace("’", "'")
    return re.sub(r"(\d)\s*[-–—]\s*(\d)", r"\1 to \2", text)


def facts_present(text: str, facts: list[list[str]]) -> bool:
    """Every fact group must match; a group matches if any of its phrasings appears."""
    t = normalize(text)
    return all(any(normalize(option) in t for option in group) for group in facts)


def score_answer(answer, q: dict) -> dict:
    if not q["gold"]:
        return {"answerable": False, "correct": answer.abstained}
    given = {chunk.doc_id for chunk in answer.sources}
    return {
        "answerable": True,
        "correct": (not answer.abstained) and facts_present(answer.text, q["facts"]),
        "cited_gold": any(c in q["gold"] for c in answer.citations),
        "grounded_citations": bool(answer.citations) and all(c in given for c in answer.citations),
        "abstained": answer.abstained,
    }


def answer_metrics(scores: list[dict]) -> dict:
    answerable = [s for s in scores if s["answerable"]]
    unanswerable = [s for s in scores if not s["answerable"]]
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows) if rows else 0.0
    return {
        "Correct answer (answerable questions)": mean(answerable, "correct"),
        "Cites a correct article": mean(answerable, "cited_gold"),
        "Every citation is a source it was given": mean(answerable, "grounded_citations"),
        "Wrongly said I don't know": mean(answerable, "abstained"),
        "Correctly said I don't know (unanswerable questions)": mean(unanswerable, "correct"),
    }


# ---------- Report ----------

def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="also score Claude's answers (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid", help="retriever used for answers")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-k", type=int, default=5, help="sections sent to Claude")
    args = parser.parse_args()

    chunks = load_chunks()
    questions = load_questions()
    n_un = sum(1 for q in questions if not q["gold"])
    lines = [
        "# Evaluation results",
        "",
        f"{len(questions)} questions in `eval/questions.jsonl`: {len(questions) - n_un} answerable, {n_un} not covered by the help center. "
        f"The help center has {len({c.doc_id for c in chunks})} articles split into {len(chunks)} sections.",
        "",
        "## Retrieval",
        "",
        f"| Retriever | Recall@1 | Recall@3 | MRR | Right article in top {args.k} sections |",
        "|---|---|---|---|---|",
    ]
    all_misses = {}
    for name in RETRIEVERS:
        m = retrieval_metrics(build_retriever(name, chunks), questions, context_k=args.k)
        all_misses[name] = m["misses"]
        lines.append(f"| {name} | {pct(m['recall@1'])} | {pct(m['recall@3'])} | {m['mrr']:.2f} | {pct(m['context'])} |")

    lines += ["", "### Questions where the right article wasn't ranked first", ""]
    for name, misses in all_misses.items():
        lines.append(f"**{name}** ({len(misses)})")
        lines.append("")
        for miss in misses:
            where = f"rank {miss['gold_rank']}" if miss["gold_rank"] else "not found"
            lines.append(f"- {miss['id']}: \"{miss['question']}\" ranked `{miss['top']}` first; right article at {where}")
        lines.append("")

    if args.answers:
        import anthropic

        client = anthropic.Anthropic()
        retriever = build_retriever(args.retriever, chunks)
        scores, wrong = [], []
        for q in questions:
            ans = answer_question(q["question"], retriever, client, model=args.model, k=args.k)
            s = score_answer(ans, q)
            scores.append(s)
            if not s["correct"]:
                wrong.append((q, ans))
            print(f"{q['id']} {'ok ' if s['correct'] else 'MISS'} {ans.text[:90]!r}")
        lines += [
            f"## Answers ({args.model}, {args.retriever} retrieval, top {args.k} sections)",
            "",
            "| Metric | Score |",
            "|---|---|",
        ]
        lines += [f"| {k} | {pct(v)} |" for k, v in answer_metrics(scores).items()]
        lines += ["", "### Answers scored as wrong", ""]
        for q, ans in wrong:
            lines.append(f"- {q['id']}: \"{q['question']}\"  \n  Answer: {ans.text}")
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
