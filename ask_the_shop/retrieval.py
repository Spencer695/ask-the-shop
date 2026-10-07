"""Three ways to find the help center sections that answer a question.

* BM25Retriever: keyword search. Strong when the question uses the same words
  as the article ("return label"), weak on paraphrases ("send it back").
  Words are stemmed so "costs" matches "cost" and "returned" matches "return".
* EmbeddingRetriever: dense search. Each text becomes a vector (an IDF weighted
  average of spaCy word vectors) and sections are ranked by cosine similarity,
  so related words like "Toronto" and "Canada" can still match.
* HybridRetriever: combines both rankings with reciprocal rank fusion (RRF).

Every retriever has the same interface: search(question, k) returns the top k
(chunk, score) pairs, best first.
"""
import math
import re
from collections import Counter

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from .corpus import Chunk

VECTORS_MODEL = "en_core_web_lg"  # spaCy English model with 300 dimension word vectors
STEMMER = snowballstemmer.stemmer("english")

STOPWORDS = set(
    "a an and are as at be but by can could do does for from get got had has have how i if "
    "in into is it its me my of on or our so than that the their them then there these they "
    "this to us was we what when where which who will with would you your".split()
)
TOKEN_RE = re.compile(r"\$?\d+(?:\.\d+)?%?|[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers (keeping "$6.95" and "10%" whole), minus stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], stem: bool = True):
        self.name = "bm25" if stem else "bm25-nostem"
        self.stem = stem
        self.chunks = chunks
        self.bm25 = BM25Okapi([self.terms(c.search_text) for c in chunks])

    def terms(self, text: str) -> list[str]:
        words = tokenize(text)
        return STEMMER.stemWords(words) if self.stem else words

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(self.terms(question))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class EmbeddingRetriever:
    name = "embeddings"

    def __init__(self, chunks: list[Chunk], nlp=None):
        if nlp is None:
            nlp = load_vectors()
        self.vocab = nlp.vocab
        self.chunks = chunks
        docs = [set(tokenize(c.search_text)) for c in chunks]
        # Inverse document frequency: rare words ("refund") say more about a section than common ones ("order").
        n = len(docs)
        df = Counter(word for words in docs for word in words)
        self.idf = {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}
        self.unseen_idf = math.log(n + 1) + 1  # words that never appear in the articles
        self.matrix = np.vstack([self.embed(c.search_text) for c in chunks])

    def embed(self, text: str) -> np.ndarray:
        """IDF weighted average of word vectors, scaled to length 1."""
        vectors, weights = [], []
        for word in tokenize(text):
            if self.vocab.has_vector(word):
                vectors.append(self.vocab.get_vector(word))
                weights.append(self.idf.get(word, self.unseen_idf))
        if not vectors:
            return np.zeros(self.vocab.vectors.shape[1], dtype=np.float32)
        vec = np.average(np.array(vectors), axis=0, weights=weights)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.matrix @ self.embed(question)  # cosine similarity, since all vectors have length 1
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (c + rank).

    RRF only uses ranks, so it doesn't matter that BM25 scores and cosine
    similarities are on different scales. c = 60 is the value from the
    original RRF paper (Cormack et al., 2009).
    """

    def __init__(self, retrievers: list, c: int = 60):
        self.retrievers = retrievers
        self.c = c
        self.name = "hybrid" + ("-tuned" if any(getattr(r, "name", "").endswith("-tuned") for r in retrievers) else "")

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[Chunk, float] = {}
        for retriever in self.retrievers:
            ranked = retriever.search(question, k=len(retriever.chunks))
            for rank, (chunk, _) in enumerate(ranked, start=1):
                fused[chunk] = fused.get(chunk, 0.0) + 1.0 / (self.c + rank)
        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return best[:k]


_VECTORS = None


def load_vectors():
    """Load the spaCy model once. Only its word vectors are used, so the rest of the pipeline is skipped."""
    global _VECTORS
    if _VECTORS is None:
        import spacy  # imported here so BM25 works without loading the vectors

        _VECTORS = spacy.load(VECTORS_MODEL, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"])
    return _VECTORS


RETRIEVER_NAMES = ["bm25-nostem", "bm25", "embeddings", "embeddings-tuned", "hybrid", "hybrid-tuned"]


def build_retriever(name: str, chunks: list[Chunk]):
    if name == "bm25-nostem":
        return BM25Retriever(chunks, stem=False)
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "embeddings":
        return EmbeddingRetriever(chunks)
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), EmbeddingRetriever(chunks)])
    if name in ("embeddings-tuned", "hybrid-tuned"):
        from .adapter import TunedEmbeddingRetriever  # needs PyTorch and models/adapter.pt

        tuned = TunedEmbeddingRetriever(chunks)
        return tuned if name == "embeddings-tuned" else HybridRetriever([BM25Retriever(chunks), tuned])
    raise ValueError(f"Unknown retriever: {name}")
