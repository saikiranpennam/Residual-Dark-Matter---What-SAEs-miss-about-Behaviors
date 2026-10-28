# plot_auroc.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_PATH = "data/truthfulqa_activations.csv"
# N_SPLITS = 2
RANDOM_STATE = 42


def parse_vector(value):
    """Convert a JSON-string activation into a NumPy vector."""
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)

    return np.asarray(value, dtype=np.float32)


df = pd.read_csv(CSV_PATH)

required_columns = ["epsilon", "x_hat", "Confidence", "Hallucination Label"]
missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

df = df.dropna(subset=required_columns).reset_index(drop=True)

X_epsilon = np.stack(df["epsilon"].map(parse_vector))
X_x_hat = np.stack(df["x_hat"].map(parse_vector))
X_confidence = df[["Confidence"]].to_numpy(dtype=np.float32)

# 1 = hallucination/wrong answer, 0 = correct answer
y = df["Hallucination Label"].to_numpy(dtype=np.int64)

print(f"Samples: {len(y)}")
print(f"Correct: {(y == 0).sum()}")
print(f"Hallucinations: {(y == 1).sum()}")
print(f"Activation dimension: {X_epsilon.shape[1]}")

minority_count = np.bincount(y).min()

if len(y) < 100:
    raise ValueError(
        f"Only {len(y)} samples. This is sufficient only for testing "
        "the pipeline, not evaluating AUROC."
    )

N_SPLITS = min(5, minority_count)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def get_oof_predictions(X):
    """Generate out-of-fold hallucination probabilities."""

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )

    predictions = cross_val_predict(
        probe,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )

    # Probability of hallucination class
    return predictions[:, 1]


experiments = {
    "SAE residual ε": X_epsilon,
    "SAE reconstruction x̂": X_x_hat,
    "Confidence": X_confidence,
}

results = {}

for name, X in experiments.items():
    scores = get_oof_predictions(X)

    auc_score = roc_auc_score(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)

    results[name] = {
        "auc": auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "scores": scores,
    }

    print(f"{name}: AUROC = {auc_score:.4f}")


# Save out-of-fold probe scores
for name, result in results.items():
    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("ε", "epsilon")
        .replace("x̂", "x_hat")
    )
    df[f"probe_score_{safe_name}"] = result["scores"]

df.to_csv("results/truthfulqa_with_probe_scores.csv", index=False)


# Plot ROC curves and AUROC bars
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {
    "SAE residual ε": "tab:red",
    "SAE reconstruction x̂": "tab:blue",
    "Confidence": "tab:green",
}

for name, result in results.items():
    axes[0].plot(
        result["fpr"],
        result["tpr"],
        linewidth=2,
        color=colors[name],
        label=f"{name} (AUROC={result['auc']:.3f})",
    )

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Chance (0.500)",
)

axes[0].set_xlabel("False-positive rate")
axes[0].set_ylabel("True-positive rate")
axes[0].set_title("Hallucination detection ROC")
axes[0].legend()
axes[0].grid(alpha=0.25)

names = list(results)
auc_values = [results[name]["auc"] for name in names]

axes[1].bar(
    names,
    auc_values,
    color=[colors[name] for name in names],
)

axes[1].axhline(0.5, linestyle="--", color="black")
axes[1].set_ylim(0.0, 1.0)
axes[1].set_ylabel("Out-of-fold AUROC")
axes[1].set_title("Probe comparison")
axes[1].tick_params(axis="x", rotation=15)

for index, value in enumerate(auc_values):
    axes[1].text(
        index,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

plt.tight_layout()
plt.savefig("results/truthfulqa_auroc.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()