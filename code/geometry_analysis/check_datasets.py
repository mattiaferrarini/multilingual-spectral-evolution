"""
Quick pre-flight check: verifies that every dataset defined in a config
actually exists on HuggingFace before submitting a cluster job.

Usage:
    python check_datasets.py --config configs/my_config.yaml
"""

import sys
import argparse
import yaml
from dotenv import load_dotenv
from datasets import load_dataset


def check_dataset(ds_config) -> tuple[bool, str]:
    name = ds_config["name"]
    path = ds_config["path"]
    subset = ds_config.get("subset", None)
    split = ds_config.get("split", "train")

    load_kwargs = dict(split=split, streaming=True)
    if subset:
        load_kwargs["name"] = subset

    try:
        ds = load_dataset(path, **load_kwargs)
        next(iter(ds))  # fetch one row to confirm data is accessible
        label = f"{path}" + (f" / {subset}" if subset else "")
        return True, f"  OK  [{name}] {label}"
    except Exception as e:
        label = f"{path}" + (f" / {subset}" if subset else "")
        return False, f"  FAIL  [{name}] {label}\n        {e}"


def main():
    parser = argparse.ArgumentParser(description="Check that config datasets exist on HuggingFace.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", [])
    if not datasets:
        print("No datasets defined in config.")
        sys.exit(0)

    print(f"Checking {len(datasets)} dataset(s)...\n")

    results = [check_dataset(ds) for ds in datasets]

    for _, msg in results:
        print(msg)

    failed = [msg for ok, msg in results if not ok]
    print(f"\n{len(datasets) - len(failed)}/{len(datasets)} datasets OK.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    main()
