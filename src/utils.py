"""Shared utilities for the RAG Bot project."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")


def save_fig(name: str, fig_dir: Path = Path("artefacts")) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def plot_chunk_length_distribution(chunks: list[str]):
    lengths = [len(c) for c in chunks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lengths, bins=40, color="#216b91", edgecolor="white")
    ax.set_title(f"Chunk length distribution (n={len(chunks)})")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Number of chunks")
    return ax


def plot_retrieval_scores(scores: list[float]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(scores) + 1), scores, color="#b05b3e")
    ax.set_title("Top-k retrieval similarity scores")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cosine similarity")
    return ax
