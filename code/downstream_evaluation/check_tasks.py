#!/usr/bin/env python3
"""
Verify that all task names defined in benchmarks.yaml are registered in lm-eval.

Usage:
    python downstream_evaluation/check_tasks.py
"""

import sys
import yaml
from pathlib import Path
from lm_eval.tasks import TaskManager


def main():
    config_path = Path(__file__).parent / "configs" / "benchmarks.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tm = TaskManager()
    available = set(tm.all_tasks)

    all_ok = True
    for benchmark, meta in config["benchmarks"].items():
        print(f"\n{benchmark}:")
        for language, task_name in meta["languages"].items():
            ok = task_name in available
            status = "OK" if ok else "MISSING"
            print(f"  [{status}] {language:12s} → {task_name}")
            if not ok:
                all_ok = False

    print()
    if all_ok:
        print("All task names are valid.")
    else:
        print("ERROR: some task names were not found in lm-eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
