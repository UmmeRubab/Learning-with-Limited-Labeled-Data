
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from research import (
    initial_indices,
    select_batch,
    summarize
)


def test_initial_indices_cover_classes():
    labels = np.tile(np.arange(4), 20)

    indices = initial_indices(
        labels,
        budget=12,
        seed=11
    )

    assert len(indices) == 12
    assert len(set(indices)) == 12
    assert set(labels[indices]) == {0, 1, 2, 3}


def test_random_selection_excludes_labeled():
    x = csr_matrix(np.eye(12))

    selected = select_batch(
        strategy="Random",
        model=None,
        x_pool=x,
        labeled=np.array([0, 1, 2, 3]),
        batch_size=4,
        rng=np.random.default_rng(11)
    )

    assert len(set(selected)) == 4

    assert not set(selected).intersection(
        {0, 1, 2, 3}
    )


def test_summary_means():
    results = pd.DataFrame([
        {
            "strategy": "Random",
            "labeled_examples": 20,
            "macro_f1": 0.4,
            "accuracy": 0.5
        },
        {
            "strategy": "Random",
            "labeled_examples": 20,
            "macro_f1": 0.6,
            "accuracy": 0.7
        }
    ])

    summary = summarize(results)

    assert abs(
        summary.loc[0, "mean_macro_f1"] - 0.5
    ) < 1e-10

