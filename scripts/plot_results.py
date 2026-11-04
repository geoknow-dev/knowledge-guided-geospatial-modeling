"""Rebuild the principal quantitative figures from versioned CSV tables."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures" / "generated"
LEVELS = ["L1-1", "L1-2", "L2-1", "L2-2", "L3-1", "L3-2"]
TASKS = ["M1", "M2", "M3"]


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.png", dpi=330, bbox_inches="tight")
    plt.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close()


def plot_task_metric(metric: str, ylabel: str, name: str) -> None:
    data = defaultdict(dict)
    for row in rows("task_metrics.csv"):
        data[row["task"]][row["knowledge_level"]] = float(row[metric])
    plt.figure(figsize=(7.1, 4.2))
    for task, marker in zip(TASKS, ["o", "s", "^"]):
        plt.plot(LEVELS, [data[task][level] for level in LEVELS], marker=marker, label=task)
    plt.xlabel("Knowledge level")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    save(name)


def plot_accuracy() -> None:
    data = defaultdict(dict)
    for row in rows("task_metrics.csv"):
        data[(row["task"], "r2")][row["knowledge_level"]] = float(row["r2"])
        data[(row["task"], "nrmse")][row["knowledge_level"]] = float(row["nrmse"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for task, marker in zip(TASKS, ["o", "s", "^"]):
        axes[0].plot(LEVELS, [data[(task, "r2")][x] for x in LEVELS], marker=marker, label=task)
        axes[1].plot(LEVELS, [data[(task, "nrmse")][x] for x in LEVELS], marker=marker, label=task)
    axes[0].set_ylabel("Spatial R²")
    axes[1].set_ylabel("Spatial nRMSE")
    for axis in axes:
        axis.set_xlabel("Knowledge level")
        axis.grid(alpha=0.3)
        axis.legend(frameon=False)
    save("figure9_spatial_accuracy")


def plot_klg() -> None:
    data = rows("knowledge_level_gain.csv")[1:]
    plt.figure(figsize=(6.5, 4.0))
    values = [float(row["klg"]) for row in data]
    bars = plt.bar([row["knowledge_level"] for row in data], values, color="#6d9dc5")
    plt.xlabel("Newly added knowledge level")
    plt.ylabel("Knowledge-Level Gain (KLG)")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center")
    save("figure7b_knowledge_level_gain")


def main() -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.linewidth": 0.5})
    plot_task_metric("kcpi", "KCPI", "figure7a_task_kcpi")
    plot_klg()
    plot_task_metric("wqi", "Workflow Quality Index (WQI)", "figure8_wqi")
    plot_accuracy()


if __name__ == "__main__":
    main()

