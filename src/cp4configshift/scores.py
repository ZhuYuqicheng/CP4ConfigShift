"""Nonconformity scores. Lower score = more conforming.

White-box scores (multiple-choice; need the logits of the K option labels):
    lac            s(x, y) = 1 - p(y | x)
    logit_margin   s(x, y) = max_{k != y} log p(k | x) - log p(y | x)
with p = softmax(logits / T) over the K options.

Black-box scores (any task; need M sampled answers per question). The
candidates are the K options for multiple-choice questions, and the distinct
answers that were actually sampled for open-ended questions. F is each
candidate's frequency among the valid samples (unparseable or empty samples
are dropped):
    self_consistency   s(x, y) = 1 - F_y
    lofreecp           s(x, y) = -F_y + lambda1 * H_norm - lambda2 * sim(y, y_top)
H_norm is the entropy of F divided by log M, and sim is the cosine similarity
between sentence embeddings of y and the most frequent answer y_top (0 for
y_top itself), following Su et al. (2024), "API Is Enough: Conformal Prediction 
for Large Language Models Without Logit-Access". As in the paper's
experiments, M is the number of samples drawn for multiple-choice questions
and the number of valid samples for open-ended questions.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from cp4configshift.data import Example
from cp4configshift.table import ScoreTable

# ── White-box scores ────────────────────────────────────────────────────


def _option_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T) over the option axis."""
    x = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _dense_table(matrix: np.ndarray, examples: Sequence[Example], info: dict) -> ScoreTable:
    n, K = matrix.shape
    answers = np.array([ex.answer for ex in examples])
    return ScoreTable(
        ids=[ex.id for ex in examples],
        candidates=[c for ex in examples for c in ex.choices],
        scores=matrix.ravel(),
        correct=(np.arange(K)[None, :] == answers[:, None]).ravel(),
        offsets=np.arange(n + 1) * K,
        info=info,
    )


def lac(logits: np.ndarray, examples: Sequence[Example], temperature: float = 1.0) -> ScoreTable:
    """``logits``: (n, K) logits of the option label tokens, row-aligned with ``examples``."""
    p = _option_probs(logits, temperature)
    return _dense_table(1.0 - p, examples, {"score": "lac", "temperature": temperature})


def logit_margin(logits: np.ndarray, examples: Sequence[Example], temperature: float = 1.0) -> ScoreTable:
    log_p = np.log(_option_probs(logits, temperature))
    top2 = -np.sort(-log_p, axis=1)[:, :2]
    best_other = np.where(log_p == top2[:, :1], top2[:, 1:2], top2[:, :1])
    return _dense_table(best_other - log_p, examples, {"score": "logit_margin", "temperature": temperature})


# ── Turning sampled text into candidates ────────────────────────────────


_THINK = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models (an
    unfinished block is removed up to the end of the text)."""
    return _THINK.sub("", text)


def parse_choice(text: str, labels: Sequence[str]) -> int:
    """Index of the option label that occurs first in ``text``, or -1 if none does.

    This is the parser used in the paper's experiments. It matches labels
    anywhere, including inside words (e.g. the "A" of "Answer", or the "a" of
    "reason" for lowercase labels), so it is most reliable when responses
    start with the label.
    """
    hits = [(text.find(lab), k) for k, lab in enumerate(labels) if lab in text]
    return min(hits)[1] if hits else -1


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


@lru_cache(maxsize=None)
def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: keep the first clause, lowercase, drop
    punctuation and articles, collapse whitespace."""
    s = text.strip()
    for stop in ("\n", ".", ","):
        idx = s.find(stop)
        if idx >= 0:
            s = s[:idx]
    s = _ARTICLES.sub(" ", s.lower().translate(_PUNCT))
    return re.sub(r"\s+", " ", s).strip()


def contains_alias(answer: str, example: Example) -> bool:
    """Correct iff some normalized gold alias is a substring of the (already
    normalized) answer."""
    for alias in example.answer:
        a = normalize_answer(alias)
        if a and a in answer:
            return True
    return False


@dataclass
class Pool:
    """The candidate answers of one question and how often each was sampled."""
    id: str
    answers: list[str]
    counts: np.ndarray
    correct: np.ndarray
    n_samples: int        # M in the LoFreeCP entropy term


def mc_pools(parsed: Sequence[Sequence[int]], examples: Sequence[Example]) -> list[Pool]:
    """Pools for multiple-choice questions. ``parsed[i][j]`` is the option index
    chosen by sample j of question i (-1 if unparseable). Every option is a
    candidate; options that were never sampled have count 0."""
    pools = []
    for row, ex in zip(parsed, examples):
        row = np.asarray(row, dtype=np.int64)
        K = len(ex.choices)
        pools.append(Pool(ex.id, list(ex.choices), np.bincount(row[row >= 0], minlength=K),
                          np.arange(K) == ex.answer, n_samples=len(row)))
    return pools


def open_pools(
    samples: Sequence[Sequence[str]],
    examples: Sequence[Example],
    is_correct: Callable[[str, Example], bool] = contains_alias,
) -> list[Pool]:
    """Pools for open-ended questions. Samples are normalized with
    ``normalize_answer``; candidates are the distinct non-empty normalized
    answers in order of first appearance."""
    pools = []
    for row, ex in zip(samples, examples):
        counts: dict[str, int] = {}
        for s in row:
            a = normalize_answer(s)
            if a:
                counts[a] = counts.get(a, 0) + 1
        answers = list(counts)
        pools.append(Pool(ex.id, answers, np.array(list(counts.values()), dtype=np.int64),
                          np.array([is_correct(a, ex) for a in answers], dtype=bool),
                          n_samples=sum(counts.values())))
    return pools


# ── Black-box scores ────────────────────────────────────────────────────


def _pool_table(pools: Sequence[Pool], scores: list[np.ndarray], info: dict) -> ScoreTable:
    sizes = [len(p.answers) for p in pools]
    return ScoreTable(
        ids=[p.id for p in pools],
        candidates=[a for p in pools for a in p.answers],
        scores=np.concatenate(scores) if scores else np.zeros(0),
        correct=np.concatenate([p.correct for p in pools]) if pools else np.zeros(0, bool),
        offsets=np.concatenate([[0], np.cumsum(sizes)]),
        info=info,
    )


def self_consistency(pools: Sequence[Pool]) -> ScoreTable:
    scores = [1.0 - p.counts / max(p.counts.sum(), 1) for p in pools]
    return _pool_table(pools, scores, {"score": "self_consistency"})


def lofreecp(
    pools: Sequence[Pool],
    embed: Callable[[list[str]], np.ndarray],
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> ScoreTable:
    """``embed`` maps a list of strings to an (len, d) array of embeddings,
    e.g. ``sentence_embedder()``."""
    texts = sorted({a for p in pools for a in p.answers})
    vecs = np.asarray(embed(texts), dtype=np.float64) if texts else np.zeros((0, 1))
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    row = {t: i for i, t in enumerate(texts)}

    scores = []
    for p in pools:
        if len(p.answers) == 0:
            scores.append(np.zeros(0))
            continue
        m = p.counts.sum()
        F = p.counts / max(m, 1)
        nz = F > 0
        h_norm = -(F[nz] * np.log(F[nz])).sum() / np.log(max(p.n_samples, 2))
        emb = vecs[[row[a] for a in p.answers]]
        top = int(np.argmax(p.counts))
        sim = emb @ emb[top]
        sim[top] = 0.0
        scores.append(-F + lambda1 * h_norm - lambda2 * sim)
    return _pool_table(pools, scores, {"score": "lofreecp", "lambda1": lambda1, "lambda2": lambda2})


@lru_cache(maxsize=None)
def sentence_embedder(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Callable[[list[str]], np.ndarray]:
    """Embedding function backed by sentence-transformers (``pip install CP4ConfigShift[embed]``)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return lambda texts: model.encode(texts, batch_size=256, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
