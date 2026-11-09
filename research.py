
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


LABELS = ["World", "Sports", "Business", "Sci/Tech"]
STRATEGIES = ["Random", "Uncertainty", "Diversity"]


def load_ag_news():
    train = load_dataset("fancyzhx/ag_news", split="train")
    test = load_dataset("fancyzhx/ag_news", split="test")
    return train, test


def balanced_pool(dataset, pool_size, seed):
    """Sample a class-balanced pool from the training split."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset["label"], dtype=int)

    if pool_size < 4 or pool_size > len(labels):
        raise ValueError("Invalid pool size.")

    quotas = [
        pool_size // 4 + (c < pool_size % 4)
        for c in range(4)
    ]

    selected = np.concatenate([
        rng.choice(
            np.flatnonzero(labels == c),
            size=quotas[c],
            replace=False
        )
        for c in range(4)
    ])

    rng.shuffle(selected)

    texts = [dataset[int(i)]["text"] for i in selected]
    return texts, labels[selected]


def initial_indices(labels, budget, seed):
    """Create the same initial labeled set for all strategies."""
    if budget < 4:
        raise ValueError("Initial budget must be at least 4.")

    rng = np.random.default_rng(seed + 1000003)

    chosen = [
        int(rng.choice(np.flatnonzero(labels == c)))
        for c in range(4)
    ]

    remaining = np.setdiff1d(
        np.arange(len(labels)),
        chosen
    )

    extra = rng.choice(
        remaining,
        size=budget - 4,
        replace=False
    )

    return np.asarray(
        chosen + extra.tolist(),
        dtype=int
    )


def select_batch(
    strategy,
    model,
    x_pool,
    labeled,
    batch_size,
    rng
):
    available = np.setdiff1d(
        np.arange(x_pool.shape[0]),
        labeled
    )

    if batch_size > len(available):
        raise ValueError("Insufficient unlabeled examples.")

    if strategy == "Random":
        return rng.choice(
            available,
            size=batch_size,
            replace=False
        )

    if strategy == "Uncertainty":
        probabilities = model.predict_proba(
            x_pool[available]
        )

        probabilities = np.clip(
            probabilities, 1e-12, 1.0
        )

        entropy = -(
            probabilities * np.log(probabilities)
        ).sum(axis=1)

        ranking = np.argsort(
            -entropy,
            kind="stable"
        )

        return available[ranking[:batch_size]]

    if strategy == "Diversity":
        similarities = cosine_similarity(
            x_pool[available],
            x_pool[labeled]
        )

        max_similarity = similarities.max(axis=1)

        ranking = np.argsort(
            max_similarity,
            kind="stable"
        )

        return available[ranking[:batch_size]]

    raise ValueError(f"Unknown strategy: {strategy}")


def run_experiment(
    train,
    test,
    pool_size=3000,
    budgets=(40, 80, 160, 320),
    seeds=(11, 23, 37),
    progress=None
):
    budgets = sorted(set(int(b) for b in budgets))

    if (
        not budgets
        or budgets[0] < 4
        or budgets[-1] > pool_size
    ):
        raise ValueError("Invalid labeling budgets.")

    if not seeds:
        raise ValueError("At least one seed is required.")

    test_texts = test["text"]
    test_labels = np.asarray(test["label"], dtype=int)

    rows = []

    total = (
        len(seeds)
        * len(STRATEGIES)
        * len(budgets)
    )
    completed = 0

    for seed in seeds:
        pool_texts, pool_labels = balanced_pool(
            train, pool_size, seed
        )

        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        # Only training-pool text is used to fit TF-IDF.
        x_pool = vectorizer.fit_transform(pool_texts)
        x_test = vectorizer.transform(test_texts)

        start = initial_indices(
            pool_labels,
            budgets[0],
            seed
        )

        for strategy in STRATEGIES:
            labeled = start.copy()

            rng = np.random.default_rng(
                seed + STRATEGIES.index(strategy) * 10000
            )

            model = None

            for budget in budgets:
                if len(labeled) < budget:
                    additions = select_batch(
                        strategy,
                        model,
                        x_pool,
                        labeled,
                        budget - len(labeled),
                        rng
                    )

                    labeled = np.concatenate(
                        (labeled, additions)
                    )

                model = LogisticRegression(
                    max_iter=400,
                    random_state=seed
                )

                model.fit(
                    x_pool[labeled],
                    pool_labels[labeled]
                )

                predictions = model.predict(x_test)

                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "labeled_examples": budget,
                    "accuracy": accuracy_score(
                        test_labels, predictions
                    ),
                    "macro_f1": f1_score(
                        test_labels,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "pool_size": pool_size
                })

                completed += 1

                if progress is not None:
                    progress(
                        completed / total,
                        f"{strategy} | Seed {seed} | "
                        f"{budget} labels"
                    )

    return pd.DataFrame(rows)


def summarize(results):
    return (
        results.groupby(
            ["strategy", "labeled_examples"],
            as_index=False
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std")
        )
    )
