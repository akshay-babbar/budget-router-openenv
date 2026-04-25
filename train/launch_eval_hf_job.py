"""
Launch cloud evaluation for a Hub-hosted trained router model.

Purpose:
    Avoid downloading multi-GB model artifacts to the laptop. The HF Job loads
    the model from the Hub, runs `train/eval_trained.py`, and prints per-seed
    scores plus the final mean directly in the job logs.

Recommended default:
    t4-small. CPU is cheaper but often too slow for 1.7B autoregressive eval.

Example:
    uv run python train/launch_eval_hf_job.py \\
        --model-path akshay4/sft-qwen3-1.7b-budget-router-smoke \\
        --n-episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from huggingface_hub import HfApi, get_token

REPO_OWNER = "akshay-babbar"
REPO_NAME = "budget-router-openenv"
DEFAULT_BRANCH = "feat/grpo-training-v5"
DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.11-bookworm"


def build_job_script(*, branch: str, model_path: str, n_episodes: int) -> str:
    archive_url = (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/archive/refs/heads/{branch}.zip"
    )
    return textwrap.dedent(
        f"""
        set -euo pipefail

        echo "Downloading repo archive: {archive_url}"
        REPO_DIR="$(python - <<'PY'
        import pathlib
        import urllib.request
        import zipfile

        url = {archive_url!r}
        target = pathlib.Path("/tmp/repo.zip")
        urllib.request.urlretrieve(url, target)
        with zipfile.ZipFile(target) as zf:
            zf.extractall("/tmp")
        candidates = sorted(
            p for p in pathlib.Path("/tmp").iterdir()
            if p.is_dir() and p.name.startswith({(REPO_NAME + "-")!r})
        )
        if not candidates:
            raise SystemExit("Could not find extracted repo directory")
        print(candidates[0])
        PY
        )"

        cd "$REPO_DIR"
        echo "Running in: $PWD"
        echo "Evaluating model: {model_path}"
        uv sync --extra grpo
        uv run python train/eval_trained.py --model-path {model_path} --n-episodes {n_episodes}
        """
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch trained-model eval on HF Jobs.")
    parser.add_argument("--model-path", required=True, help="Hub model id or local path inside the job.")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--flavor", default="t4-small")
    parser.add_argument("--timeout", default="25m")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("ERROR: No Hugging Face token found. Run `hf auth login` first.", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]
    script = build_job_script(
        branch=args.branch,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
    )
    job = api.run_job(
        image=args.image,
        command=["/bin/bash", "-lc", script],
        secrets={"HF_TOKEN": token},
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=namespace,
        labels={"task": "budget-router-eval", "model": args.model_path.replace("/", "_")},
    )

    print(f"Eval job started: {job.id}")
    print(f"View at: {job.url}")
    print(f"Logs: hf jobs logs {job.id}")


if __name__ == "__main__":
    main()
