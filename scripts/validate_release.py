"""Fast checks for a safe, internally consistent public release."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".csv", ".template"}
FORBIDDEN = {
    "credential": re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    "private data path": re.compile(r"[A-Za-z]:\\\\(?:土壤侵蚀数据|soi_ersion)"),
}


def validate_json() -> None:
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def validate_public_text() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
    if failures:
        raise RuntimeError("Release scan failed:\n" + "\n".join(failures))


def validate_klg() -> None:
    with (ROOT / "results" / "task_metrics.csv").open(encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    with (ROOT / "results" / "knowledge_level_gain.csv").open(encoding="utf-8", newline="") as handle:
        gain_rows = list(csv.DictReader(handle))

    levels = [row["knowledge_level"] for row in gain_rows]
    task_weights = {"M1": 1, "M2": 4, "M3": 8}
    weighted: list[float] = []
    for level in levels:
        selected = [row for row in task_rows if row["knowledge_level"] == level]
        value = sum(float(row["kcpi"]) * task_weights[row["task"]] for row in selected) / sum(task_weights.values())
        weighted.append(value)
    for index, row in enumerate(gain_rows):
        if abs(weighted[index] - float(row["weighted_kcpi"])) > 5e-7:
            raise RuntimeError(f"weighted KCPI mismatch at {row['knowledge_level']}")
        if index and abs((weighted[index] - weighted[index - 1]) - float(row["klg"])) > 5e-7:
            raise RuntimeError(f"KLG mismatch at {row['knowledge_level']}")


def main() -> None:
    validate_json()
    validate_public_text()
    validate_klg()
    print("release validation passed")


if __name__ == "__main__":
    main()

