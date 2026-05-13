#!/usr/bin/env python3
"""
Fetch downstream eval JSON results from the cluster to local results/eval/.

Uses `runai training exec` to cat each file — no kubectl or SSH needed.
Useful for debugging or re-running merge_results.py locally with different
parameters (e.g. a different layer). For the normal workflow, just fetch
the already-merged CSV with the one-liner printed by submit_eval.sh.

Usage:
    # Fetch one checkpoint
    python downstream_evaluation/fetch_json_results.py --checkpoint 531B

    # Fetch multiple checkpoints
    python downstream_evaluation/fetch_json_results.py --checkpoint 531B 10B 220B

    # Fetch all checkpoints listed in results/eval/ on the cluster
    python downstream_evaluation/fetch_json_results.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
GASPAR          = "jpazeved"
PROJECT         = f"course-cs-552-{GASPAR}"
CLUSTER_FOLDER  = "joao"
REPO_NAME       = "open-project-m2-jpmg"
SHELL_JOB       = "temp-shell3"   # a running sleep job to exec into
REMOTE_EVAL_DIR = f"/scratch/{CLUSTER_FOLDER}/{REPO_NAME}/results/eval"

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmarks.yaml"


def load_expected_files(config_path: Path) -> list[str]:
    """Return list of expected JSON stems: task__Language"""
    with open(config_path) as f:
        benchmarks = yaml.safe_load(f)["benchmarks"]
    stems = []
    for task, cfg in benchmarks.items():
        for language in cfg["languages"]:
            stems.append(f"{task}__{language}")
    return stems


def fetch_file(shell_job: str, project: str, remote_path: str, local_path: Path) -> bool:
    """Cat a single remote file into a local path via runai training exec."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["runai", "training", "exec", shell_job, "-p", project,
           "--", "cat", remote_path]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            print(f"  [MISS] {remote_path.split('/')[-1]} — not found on cluster")
            return False
        # Validate it's valid JSON before writing
        json.loads(result.stdout)
        local_path.write_bytes(result.stdout)
        print(f"  [OK]   {local_path}")
        return True
    except json.JSONDecodeError:
        print(f"  [ERR]  {remote_path.split('/')[-1]} — invalid JSON received")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERR]  {remote_path.split('/')[-1]} — timed out")
        return False


def fetch_checkpoint(checkpoint: str, local_eval_dir: Path, expected: list[str],
                     shell_job: str, project: str) -> tuple[int, int]:
    print(f"\nCheckpoint: {checkpoint}")
    ok = miss = 0
    for stem in expected:
        remote = f"{REMOTE_EVAL_DIR}/{checkpoint}/{stem}.json"
        local  = local_eval_dir / checkpoint / f"{stem}.json"
        if local.exists():
            print(f"  [SKIP] {stem}.json — already local")
            ok += 1
            continue
        if fetch_file(shell_job, project, remote, local):
            ok += 1
        else:
            miss += 1
    return ok, miss


def main():
    p = argparse.ArgumentParser(description="Fetch eval JSONs from cluster via runai exec.")
    p.add_argument("--checkpoint", nargs="+", help="Checkpoint name(s) to fetch (e.g. 531B 10B)")
    p.add_argument("--all", action="store_true", help="Fetch all checkpoints found on the cluster")
    p.add_argument("--shell-job", default=SHELL_JOB, help=f"Running sleep job to exec into (default: {SHELL_JOB})")
    p.add_argument("--local-eval-dir", default="results/eval", help="Local output directory")
    args = p.parse_args()

    if not args.checkpoint and not args.all:
        p.error("Specify --checkpoint <name> or --all")

    local_eval_dir = Path(args.local_eval_dir)
    expected = load_expected_files(_DEFAULT_CONFIG)
    print(f"Expected files per checkpoint: {len(expected)}")
    print(f"Exec shell job: {args.shell_job}  |  Project: {PROJECT}")

    if args.all:
        # List remote checkpoints by trying to cat a directory listing
        result = subprocess.run(
            ["runai", "training", "exec", args.shell_job, "-p", PROJECT,
             "--", "ls", REMOTE_EVAL_DIR],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            print(f"ERROR: could not list {REMOTE_EVAL_DIR}. Is {args.shell_job} running?")
            sys.exit(1)
        checkpoints = result.stdout.decode().split()
        print(f"Found {len(checkpoints)} checkpoint(s) on cluster: {checkpoints}")
    else:
        checkpoints = args.checkpoint

    total_ok = total_miss = 0
    for ckpt in checkpoints:
        ok, miss = fetch_checkpoint(ckpt, local_eval_dir, expected, args.shell_job, PROJECT)
        total_ok += ok
        total_miss += miss

    print(f"\nDone — {total_ok} fetched/skipped, {total_miss} missing.")


if __name__ == "__main__":
    main()
