"""Tests for the PyTorch embedding adapter."""
import json

import numpy as np
import pytest
import torch

from ask_the_shop.adapter import (
    SETTINGS,
    Adapter,
    ArticleScorer,
    TunedEmbeddingRetriever,
    contrastive_loss,
    fit,
    load_training_questions,
)
from ask_the_shop.corpus import load_chunks
from ask_the_shop.evaluate import QUESTIONS_PATH
from ask_the_shop.retrieval import EmbeddingRetriever, load_vectors


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def base(chunks):
    return EmbeddingRetriever(chunks, load_vectors())


def normalize(text):
    return " ".join(text.lower().replace("?", "").replace(".", "").split())


def test_training_questions_never_appear_in_the_test_set():
    train = {normalize(q["question"]) for q in load_training_questions()}
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        test = {normalize(json.loads(line)["question"]) for line in f if line.strip()}
    assert train.isdisjoint(test)


def test_untrained_adapter_matches_the_base_retriever(chunks, base):
    tuned = TunedEmbeddingRetriever(chunks, adapter=Adapter(300, SETTINGS["rank"]).eval(), nlp=load_vectors())
    for question in ["Will the cap fit a big head?", "How long do refunds take?"]:
        assert [c.doc_id for c, _ in base.search(question, k=5)] == [c.doc_id for c, _ in tuned.search(question, k=5)]


def test_contrastive_loss_allows_several_correct_articles():
    logits = torch.tensor([[2.0, 2.0, -5.0]])
    one_gold = contrastive_loss(logits, torch.tensor([[True, False, False]]))
    two_gold = contrastive_loss(logits, torch.tensor([[True, True, False]]))
    assert two_gold < one_gold
    expected = -np.log(2 * np.exp(2) / (2 * np.exp(2) + np.exp(-5)))
    assert two_gold.item() == pytest.approx(expected, abs=1e-6)  # float32 precision


def test_article_scorer_gives_one_score_per_article(chunks, base):
    scorer = ArticleScorer(chunks, torch.tensor(base.matrix, dtype=torch.float32), temperature=0.05)
    question = torch.tensor(np.array([base.embed("refund")]), dtype=torch.float32)
    assert scorer.logits(Adapter(300), question).shape == (1, 21)


def test_training_lowers_the_training_loss(base):
    _, info = fit(base, load_training_questions()[:60], epochs=30)
    losses = [row["train_loss"] for row in info["history"]]
    assert len(losses) == 30 and losses[-1] < losses[0]


def test_early_stopping_keeps_the_best_validation_epoch(base):
    questions = load_training_questions()
    _, info = fit(base, questions[:80], questions[80:], max_epochs=60)
    best_val = min([info["start"]["val_loss"]] + [row["val_loss"] for row in info["history"]])
    assert info["best"]["val_loss"] == pytest.approx(best_val)


def test_saved_adapter_loads_and_searches(chunks):
    tuned = TunedEmbeddingRetriever(chunks, nlp=load_vectors())
    results = tuned.search("Do you deliver to Toronto?", k=3)
    assert len(results) == 3 and "international-shipping" in [c.doc_id for c, _ in results]
