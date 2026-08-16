"""
Captioning evaluation metrics: BLEU-1..4, ROUGE-L, CIDEr-D.

Implemented from scratch (only needs Python's stdlib + math) so no extra
pip installs are required to run this alongside the existing project.
Single-reference-per-image scoring (ROCOv2 gives exactly one ground-truth
caption per image in train_captions.csv / valid_captions.csv / test_captions.csv).
"""

import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------- BLEU ----
def corpus_bleu(references, hypotheses, max_n=4):
    """Standard cumulative corpus BLEU-1..max_n (Papineni et al.), single
    reference per hypothesis. references/hypotheses: lists of token lists.
    Returns dict {'bleu1':.., 'bleu2':.., 'bleu3':.., 'bleu4':..}."""
    clipped = [0] * max_n
    total = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp, n)
            ref_ngrams = _ngrams(ref, n)
            clipped[n - 1] += sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
            total[n - 1] += max(sum(hyp_ngrams.values()), 0)

    precisions = [clipped[i] / total[i] if total[i] > 0 else 0.0 for i in range(max_n)]

    bp = 1.0 if hyp_len > ref_len else (math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0)

    out = {}
    for n in range(1, max_n + 1):
        ps = precisions[:n]
        if min(ps) > 0:
            geo_mean = math.exp(sum(math.log(p) for p in ps) / n)
        else:
            geo_mean = 0.0
        out[f"bleu{n}"] = bp * geo_mean
    return out


# -------------------------------------------------------------- ROUGE-L ---
def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference, hypothesis, beta=1.2):
    """Per-sample ROUGE-L F-measure. reference/hypothesis: token lists."""
    if not reference or not hypothesis:
        return 0.0
    lcs = _lcs_len(reference, hypothesis)
    if lcs == 0:
        return 0.0
    r = lcs / len(reference)
    p = lcs / len(hypothesis)
    return ((1 + beta ** 2) * r * p) / (r + beta ** 2 * p)


def corpus_rouge_l(references, hypotheses, beta=1.2):
    scores = [rouge_l(r, h, beta) for r, h in zip(references, hypotheses)]
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------- CIDEr ----
def _tfidf_vector(tokens, idf, n_max=4):
    """Build one TF-IDF vector per n-gram order (1..n_max) for a sentence."""
    vecs = []
    for n in range(1, n_max + 1):
        counts = _ngrams(tokens, n)
        total = sum(counts.values())
        vec = {}
        for g, c in counts.items():
            tf = c / total if total > 0 else 0.0
            vec[g] = tf * idf[n].get(g, 0.0)
        vecs.append(vec)
    return vecs


def _cos_sim(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[g] * b[g] for g in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def corpus_cider_d(references, hypotheses, n_max=4, sigma=6.0):
    """Lightweight CIDEr-D (Vedantam et al.) for single-reference captions.
    references/hypotheses: lists of token lists, one reference per image.
    Note: this is a compact reimplementation for convenience -- for a
    publication-grade number, cross-check against pycocoevalcap."""
    num_docs = len(references)
    df = [Counter() for _ in range(n_max + 1)]  # df[n][ngram] = #docs containing it
    for ref in references:
        for n in range(1, n_max + 1):
            for g in set(_ngrams(ref, n)):
                df[n][g] += 1

    idf = [dict()] + [
        {g: math.log(max(num_docs, 1) / (c + 1e-6)) for g, c in df[n].items()}
        for n in range(1, n_max + 1)
    ]

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_vecs = _tfidf_vector(ref, idf, n_max)
        hyp_vecs = _tfidf_vector(hyp, idf, n_max)
        per_n = []
        for n in range(n_max):
            sim = _cos_sim(ref_vecs[n], hyp_vecs[n])
            length_penalty = math.exp(-((len(hyp) - len(ref)) ** 2) / (2 * sigma ** 2))
            per_n.append(sim * length_penalty)
        scores.append(10.0 * sum(per_n) / n_max)  # CIDEr conventionally scaled x10
    return sum(scores) / len(scores) if scores else 0.0



if __name__ == "__main__":
    ref = "axial ct chest shows a right upper lobe nodule".split()
    hyp_perfect = ref
    hyp_partial = "axial ct chest shows a nodule".split()
    hyp_bad = "mri brain shows no abnormality".split()

    for name, hyp in [("identical", hyp_perfect), ("partial overlap", hyp_partial), ("unrelated", hyp_bad)]:
        b = corpus_bleu([ref], [hyp])
        r = corpus_rouge_l([ref], [hyp])
        c = corpus_cider_d([ref], [hyp])
        print(f"{name:16s} BLEU1={b['bleu1']:.3f} BLEU4={b['bleu4']:.3f} ROUGE-L={r:.3f} CIDEr={c:.3f}")
