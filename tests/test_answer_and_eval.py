"""Tests for the answer step and scoring. A fake client stands in for Claude, so no API key is needed."""
from types import SimpleNamespace

from ask_the_shop.answer import ABSTAIN_TEXT, SYSTEM_PROMPT, answer_question, is_abstention, parse_citations
from ask_the_shop.corpus import Chunk
from ask_the_shop.evaluate import answer_metrics, facts_present, normalize, retrieval_metrics, score_answer

RETURNS = Chunk("returns", "Returns", "Return window", "You can return most items within 30 days of delivery.")
REFUNDS = Chunk("refunds", "Refunds", "Refund timing", "We process your refund within 3 business days.")


class FakeRetriever:
    chunks = [RETURNS, REFUNDS]

    def search(self, question, k=5):
        return [(c, 1.0) for c in self.chunks[:k]]


class FakeClient:
    """Records the request and replies with a canned answer, like anthropic.Anthropic().messages."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def test_prompt_contains_sources_and_question():
    client = FakeClient("You have 30 days [returns].")
    answer_question("How long do I have?", FakeRetriever(), client, k=2)
    request = client.requests[0]
    assert request["system"] == SYSTEM_PROMPT
    user = request["messages"][0]["content"]
    assert '<source id="returns"' in user and '<source id="refunds"' in user
    assert user.endswith("Customer question: How long do I have?")


def test_answer_parses_citations_and_abstention():
    answer = answer_question("q", FakeRetriever(), FakeClient("30 days [returns], refund in 3 days [refunds] [returns]."))
    assert answer.citations == ["returns", "refunds"]
    assert not answer.abstained

    unsure = answer_question("q", FakeRetriever(), FakeClient(f"{ABSTAIN_TEXT} Please email us."))
    assert unsure.abstained and unsure.citations == []


def test_abstention_handles_curly_apostrophes():
    assert is_abstention("I don’t know based on our help center. Email us.")
    assert not is_abstention("We don't sell gift wrap.")


def test_parse_citations_ignores_non_ids():
    assert parse_citations("See [returns] and [Click here] and [gift-cards].") == ["returns", "gift-cards"]


def test_normalize_turns_ranges_into_words():
    assert normalize("Arrives in 3–7 business days") == "arrives in 3 to 7 business days"
    assert normalize("2-3 days") == "2 to 3 days"


def test_facts_need_every_group_but_any_option():
    facts = [["$6.95"], ["free", "no charge"]]
    assert facts_present("Shipping is $6.95, or no charge over $75.", facts)
    assert not facts_present("Shipping is $6.95.", facts)


def test_score_answer_and_metrics():
    q_answerable = {"gold": ["returns"], "facts": [["30 days"]]}
    q_unanswerable = {"gold": [], "facts": []}
    good = answer_question("q", FakeRetriever(), FakeClient("You have 30 days [returns]."))
    invented = answer_question("q", FakeRetriever(), FakeClient("You have 30 days [shipping-options]."))
    unsure = answer_question("q", FakeRetriever(), FakeClient(ABSTAIN_TEXT))

    s_good = score_answer(good, q_answerable)
    assert s_good["correct"] and s_good["cited_gold"] and s_good["grounded_citations"]
    s_invented = score_answer(invented, q_answerable)
    assert s_invented["correct"] and not s_invented["cited_gold"] and not s_invented["grounded_citations"]
    assert not score_answer(unsure, q_answerable)["correct"]
    assert score_answer(unsure, q_unanswerable)["correct"]

    metrics = answer_metrics([s_good, s_invented, score_answer(unsure, q_unanswerable)])
    assert metrics["Correct answer (answerable questions)"] == 1.0
    assert metrics["Cites a correct article"] == 0.5
    assert metrics["Correctly said I don't know (unanswerable questions)"] == 1.0


def test_retrieval_metrics_on_known_ranking():
    questions = [
        {"id": "a", "question": "x", "gold": ["returns"]},   # rank 1
        {"id": "b", "question": "y", "gold": ["refunds"]},   # rank 2
        {"id": "c", "question": "z", "gold": ["missing"]},   # not found
        {"id": "u", "question": "w", "gold": []},            # unanswerable, skipped
    ]
    m = retrieval_metrics(FakeRetriever(), questions, context_k=1)
    assert m["n"] == 3
    assert m["recall@1"] == 1 / 3
    assert m["recall@3"] == 2 / 3
    assert m["mrr"] == (1 + 0.5 + 0) / 3
    assert m["context"] == 1 / 3
