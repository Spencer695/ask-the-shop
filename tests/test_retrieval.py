import pytest

from ask_the_shop.corpus import load_chunks
from ask_the_shop.retrieval import RETRIEVER_NAMES, HybridRetriever, build_retriever, tokenize


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def retrievers(chunks):
    return {name: build_retriever(name, chunks) for name in RETRIEVER_NAMES}


def test_every_article_loads_with_sections(chunks):
    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 21
    assert all(c.title and c.heading and c.text for c in chunks)
    assert not any(c.text.startswith("#") for c in chunks)


def test_tokenize_keeps_prices_and_drops_stopwords():
    assert tokenize("Is shipping $6.95 or 10% off?") == ["shipping", "$6.95", "10%", "off"]


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_search_returns_k_results_best_first(retrievers, name):
    results = retrievers[name].search("How long do refunds take?", k=4)
    assert len(results) == 4
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_exact_wording_finds_the_right_article(retrievers, name):
    top_chunk, _ = retrievers[name].search("gift card expire", k=1)[0]
    assert top_chunk.doc_id == "gift-cards"


def test_stemming_matches_word_forms(chunks):
    stemmed = build_retriever("bm25", chunks)
    assert stemmed.terms("costs") == stemmed.terms("cost")


def test_embeddings_match_related_words(chunks):
    # "Toronto" never appears in the articles; word vectors connect it to "Canada".
    dense = build_retriever("embeddings", chunks)
    top_docs = [c.doc_id for c, _ in dense.search("Do you deliver to Toronto?", k=3)]
    assert "international-shipping" in top_docs


def test_reciprocal_rank_fusion_math():
    class Fixed:
        def __init__(self, order):
            self.chunks = order

        def search(self, question, k=5):
            return [(c, 0.0) for c in self.chunks[:k]]

    a, b, c = "a", "b", "c"
    hybrid = HybridRetriever([Fixed([a, b, c]), Fixed([b, c, a])], c=60)
    scores = dict(hybrid.search("anything", k=3))
    assert scores[b] == pytest.approx(1 / 62 + 1 / 61)
    assert [item for item, _ in hybrid.search("anything", k=3)] == [b, a, c]
