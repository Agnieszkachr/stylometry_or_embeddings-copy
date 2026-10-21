import warnings

import numpy as np

from src.cv_pipeline import waterfall_cv


def _synthetic_corpus(n_authors=3, n_per_author=20, seed=0):
    rng = np.random.RandomState(seed)
    vocab = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    poems, labels = [], []
    for author in range(n_authors):
        for _ in range(n_per_author):
            text = "\n".join(" ".join(rng.choice(vocab, 5)) for _ in range(4))
            poems.append({"text": text, "linguistic_features": {}, "prosody": {}})
            labels.append(author)
    labels = np.array(labels)
    embeddings = rng.randn(len(labels), 8) + labels[:, None]
    return poems, embeddings, labels


def test_waterfall_cv_fills_every_stage():
    poems, embeddings, labels = _synthetic_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = waterfall_cv(poems, embeddings, labels, n_splits=3)

    assert set(summary) == {"baseline", "tier1", "tier1+2", "tier1+2+3",
                            "tier1+2+3+4", "tier5_kernel"}
    for stage, stats in summary.items():
        assert len(stats["scores"]) == 3, stage
        assert np.isfinite(stats["mean"]), stage
