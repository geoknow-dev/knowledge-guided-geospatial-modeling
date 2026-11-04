"""Run one controlled knowledge-injection experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import GeoAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["L1-1", "L1-2", "L2-1", "L2-2", "L3-1", "L3-2"], required=True)
    parser.add_argument("--task", required=True, help="Natural-language modeling task")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs" / "data_paths.example.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-kb", action="store_true", help="Disable external knowledge injection")
    parser.add_argument("--history", action="store_true", help="Enable conversation history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the experiment.")
    model_name = os.getenv("MODEL_NAME")
    api_base = os.getenv("OPENAI_BASE_URL")
    if not model_name or not api_base:
        raise RuntimeError("Set MODEL_NAME and OPENAI_BASE_URL to the exact experimental setup.")

    data_paths = json.loads(args.data_config.read_text(encoding="utf-8"))
    agent = GeoAgent(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        data_paths=data_paths,
        knowledge_base_dir=str(ROOT / "knowledge"),
        conversation_history_enabled=args.history,
        kb_injection_enabled=not args.no_kb,
        temperature=args.temperature,
        max_history_messages=12,
    )

    system_prompt, workflow = agent.plan(args.task, level=args.level)
    code = agent.generate_code(system_prompt, workflow)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.level}_kb{int(not args.no_kb)}_mem{int(args.history)}"
    (args.output_dir / f"workflow_{stem}.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / f"script_{stem}.py").write_text(code, encoding="utf-8")
    print(f"Saved workflow and code to {args.output_dir}")


if __name__ == "__main__":
    main()

