"""Fine-tune the embedding search with a small PyTorch model.

The spaCy word vectors are general English. They don't know that in this shop a
"cap" is the dad hat or that "send it back" means a return. This module learns
a small correction to the vectors that pulls each training question toward its
article, and applies it to both questions and sections.

* The adapter is x -> x + x V U^T (a rank 32 residual). It starts with V = 0,
  so an untrained adapter changes nothing, and a penalty on the size of the
  correction keeps it small. Both limit overfitting, since there are only
  about 170 training examples. A full 300 x 300 matrix overfit within a few
  steps in cross validation.
* Training data: 105 handwritten questions in eval/train_questions.jsonl, plus
  every section heading used as a short pseudo question. The 70 questions in
  eval/questions.jsonl are the test set and are never used for training or
  for choosing settings.
* Settings were chosen with 5 fold cross validation on the handwritten
  questions. The final adapter is trained on all of them for the average best
  epoch count from cross validation.
* Loss: score every section by cosine similarity, combine each article's
  sections with logsumexp into an article score, and apply cross entropy
  against the correct article. The other 20 articles act as negatives, so
  this is a contrastive objective.

Train with:  python -m ask_the_shop.adapter
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .corpus import Chunk, load_chunks
from .retrieval import EmbeddingRetriever

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "eval" / "train_questions.jsonl"
ADAPTER_PATH = ROOT / "models" / "adapter.pt"

# Chosen with 5 fold cross validation (python -m ask_the_shop.adapter prints it).
SETTINGS = {"rank": 32, "temperature": 0.05, "l2": 0.1, "lr": 0.003, "max_epochs": 300}
FOLDS = 5
SEED = 0


class Adapter(torch.nn.Module):
    """x -> normalize(x + x V U^T), a low rank correction to the word vector space."""

    def __init__(self, dim: int = 300, rank: int = SETTINGS["rank"]):
        super().__init__()
        self.u = torch.nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.v = torch.nn.Parameter(torch.zeros(dim, rank))  # zero, so training starts from the original vectors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + (x @ self.v) @ self.u.T, dim=-1)

    def correction_size(self) -> torch.Tensor:
        return ((self.v @ self.u.T) ** 2).sum()


def load_training_questions(path: Path = TRAIN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ArticleScorer:
    """Turns question vectors into one score per article, given the section vectors."""

    def __init__(self, chunks: list[Chunk], chunk_vectors: torch.Tensor, temperature: float):
        self.doc_ids = sorted({c.doc_id for c in chunks})
        index = {d: i for i, d in enumerate(self.doc_ids)}
        self.masks = [torch.tensor([index[c.doc_id] == i for c in chunks]) for i in range(len(self.doc_ids))]
        self.chunk_vectors = chunk_vectors
        self.temperature = temperature

    def logits(self, adapter: Adapter, question_vectors: torch.Tensor) -> torch.Tensor:
        sims = adapter(question_vectors) @ adapter(self.chunk_vectors).T / self.temperature
        return torch.stack([torch.logsumexp(sims[:, mask], dim=1) for mask in self.masks], dim=1)

    def gold_mask(self, golds: list[list[str]]) -> torch.Tensor:
        return torch.tensor([[d in gold for d in self.doc_ids] for gold in golds])


def contrastive_loss(logits: torch.Tensor, gold_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy that allows more than one correct article: -log P(any correct article)."""
    gold_logits = logits.masked_fill(~gold_mask, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(gold_logits, dim=1)).mean()


def top1_accuracy(logits: torch.Tensor, gold_mask: torch.Tensor) -> float:
    return gold_mask[torch.arange(len(gold_mask)), logits.argmax(dim=1)].float().mean().item()


def fit(base: EmbeddingRetriever, train_qs: list[dict], val_qs: list[dict] | None = None,
        epochs: int | None = None, seed: int = SEED, **overrides):
    """Train on questions plus section headings.

    With val_qs, keeps the epoch with the lowest validation loss (early stopping).
    Without, trains for exactly `epochs` epochs.
    """
    s = {**SETTINGS, **overrides}
    torch.manual_seed(seed)

    def embed(texts):
        return torch.tensor(np.array([base.embed(t) for t in texts]), dtype=torch.float32)

    scorer = ArticleScorer(base.chunks, torch.tensor(base.matrix, dtype=torch.float32), s["temperature"])
    x_train = embed([q["question"] for q in train_qs] + [c.heading for c in base.chunks])
    y_train = scorer.gold_mask([q["gold"] for q in train_qs] + [[c.doc_id] for c in base.chunks])
    if val_qs:
        x_val, y_val = embed([q["question"] for q in val_qs]), scorer.gold_mask([q["gold"] for q in val_qs])

    adapter = Adapter(x_train.shape[1], s["rank"])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=s["lr"])
    snapshot = lambda: {k: v.clone() for k, v in adapter.state_dict().items()}
    history = []
    best = {"epoch": 0, "val_loss": float("inf"), "val_acc": 0.0, "state": snapshot()}
    if val_qs:
        with torch.no_grad():
            start_logits = scorer.logits(adapter, x_val)
        best.update(val_loss=contrastive_loss(start_logits, y_val).item(), val_acc=top1_accuracy(start_logits, y_val))
    start = dict(best)

    for epoch in range(1, (epochs or s["max_epochs"]) + 1):
        optimizer.zero_grad()
        task_loss = contrastive_loss(scorer.logits(adapter, x_train), y_train)
        (task_loss + s["l2"] * adapter.correction_size()).backward()
        optimizer.step()
        row = {"epoch": epoch, "train_loss": task_loss.item()}
        if val_qs:
            with torch.no_grad():
                val_logits = scorer.logits(adapter, x_val)
            row.update(val_loss=contrastive_loss(val_logits, y_val).item(), val_acc=top1_accuracy(val_logits, y_val))
            if row["val_loss"] < best["val_loss"]:
                best = {"epoch": epoch, "val_loss": row["val_loss"], "val_acc": row["val_acc"], "state": snapshot()}
        history.append(row)

    if val_qs:
        adapter.load_state_dict(best["state"])
    adapter.eval()
    return adapter, {"history": history, "best": best, "start": start}


def cross_validate(base: EmbeddingRetriever, questions: list[dict], folds: int = FOLDS, seed: int = SEED, **overrides) -> dict:
    """Average validation loss and top 1 accuracy, before and after training, over k folds."""
    shuffled = questions[:]
    random.Random(seed).shuffle(shuffled)
    rows = []
    for k in range(folds):
        val = shuffled[k::folds]
        train = [q for i, q in enumerate(shuffled) if i % folds != k]
        _, info = fit(base, train, val, seed=seed, **overrides)
        rows.append(info)
    mean = lambda f: float(np.mean([f(r) for r in rows]))
    return {
        "start_val_loss": mean(lambda r: r["start"]["val_loss"]),
        "start_val_acc": mean(lambda r: r["start"]["val_acc"]),
        "val_loss": mean(lambda r: r["best"]["val_loss"]),
        "val_acc": mean(lambda r: r["best"]["val_acc"]),
        "best_epoch": mean(lambda r: r["best"]["epoch"]),
    }


class TunedEmbeddingRetriever(EmbeddingRetriever):
    """The embedding retriever with the trained adapter applied to questions and sections."""

    name = "embeddings-tuned"

    def __init__(self, chunks: list[Chunk], adapter: Adapter | None = None, nlp=None):
        super().__init__(chunks, nlp)
        if adapter is None:
            adapter = Adapter(self.matrix.shape[1])
            adapter.load_state_dict(torch.load(ADAPTER_PATH, weights_only=True))
            adapter.eval()
        self.adapter = adapter
        self.matrix = self._apply(self.matrix)

    def _apply(self, vectors):
        with torch.no_grad():
            return self.adapter(torch.as_tensor(np.asarray(vectors), dtype=torch.float32)).numpy()

    def search(self, question: str, k: int = 5):
        scores = self.matrix @ self._apply(self.embed(question)[None, :])[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def main() -> None:
    chunks = load_chunks()
    base = EmbeddingRetriever(chunks)
    questions = load_training_questions()

    cv = cross_validate(base, questions)
    print(f"{FOLDS} fold cross validation on {len(questions)} training questions:")
    print(f"  before training: val loss {cv['start_val_loss']:.3f}, val top 1 accuracy {cv['start_val_acc']:.0%}")
    print(f"  after training:  val loss {cv['val_loss']:.3f}, val top 1 accuracy {cv['val_acc']:.0%} "
          f"(best epoch {cv['best_epoch']:.0f} on average)")

    epochs = max(1, round(cv["best_epoch"]))
    adapter, _ = fit(base, questions, epochs=epochs)
    ADAPTER_PATH.parent.mkdir(exist_ok=True)
    torch.save(adapter.state_dict(), ADAPTER_PATH)
    print(f"Trained on all {len(questions)} questions plus {len(chunks)} headings for {epochs} epochs; "
          f"saved {ADAPTER_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
