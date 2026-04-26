"""
Launch the PPO → SFT distillation job on Hugging Face Jobs.

Same pattern as launch_sft_hf_job.py but runs sft_ppo_distill.py instead.
The PPO model (trained_models/ppo_hard_multi_100k.zip) is included in the
repo archive — no extra downloads needed.

Usage:
    uv run python train/launch_sft_ppo_job.py --smoke   # ~5 min, ~$0.10
    uv run python train/launch_sft_ppo_job.py --full    # ~30-45 min, ~$0.90
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


def build_job_script(*, branch: str, train_args: str) -> str:
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
        uv sync --extra grpo
        uv run python train/sft_ppo_distill.py {train_args}
        """
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PPO → SFT distillation on HF Jobs.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run the short smoke job.")
    mode.add_argument("--full", action="store_true", help="Run the full SFT job.")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--namespace", default=None)
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="Target model repo on the Hub.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("ERROR: No Hugging Face token found. Run `hf auth login` first.", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]
    if args.smoke:
        timeout = "30m"
        hub_repo = args.hub_repo or f"{namespace}/sft-ppo-qwen3-1.7b-budget-router-smoke"
        train_args = f"--smoke --push-to-hub {hub_repo}"
    else:
        timeout = "120m"
        hub_repo = args.hub_repo or f"{namespace}/sft-ppo-qwen3-1.7b-budget-router"
        train_args = f"--full --push-to-hub {hub_repo}"

    # Preflight: create the repo so we know auth works
    api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)

    script = build_job_script(branch=args.branch, train_args=train_args)
    job = api.run_job(
        image=args.image,
        command=["/bin/bash", "-lc", script],
        secrets={"HF_TOKEN": token},
        flavor=args.flavor,
        timeout=timeout,
        namespace=namespace,
        labels={"task": "budget-router-sft-ppo", "mode": "smoke" if args.smoke else "full"},
    )

    print(f"Job started: {job.id}")
    print(f"View at: {job.url}")
    print(f"Model will be pushed to: https://huggingface.co/{hub_repo}")


if __name__ == "__main__":
    main()
