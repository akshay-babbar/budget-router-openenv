#!/usr/bin/env bash
# Push to a Hugging Face Space as branch "main" (force).
#
# By default we do **not** push your current HEAD as-is. We build a **new** commit (no local
# branch change) that **drops** anything that would exceed the Hub / your size policy:
#   - any binary file (plain Git binaries are rejected by Hugging Face without Xet)
#   - any file (blob) > MAX_BYTES (default 10 MiB)
#   - any directory *prefix* whose **sum of kept** blob sizes would exceed MAX_BYTES; the
#     whole subtree is dropped
# This is the only way to "silently skip" content that is *already* in your git history: a
# normal `git push` would still send old blobs. See scripts/hf_export_build_commit.py.
# Default stderr prints a one-line omit count; set QUIET=1 to suppress it, VERBOSE=1
# to print each omitted path. MAX_BYTES is honored by the Python step.
#
# Environment:
#   HF_TOKEN         — required; write token
#   SNAPSHOT_COMMIT=1 — `git add -A`, commit (after unstaging single files over MAX_BYTES on
#                    disk, still respects .gitignore), then export+push
#   SKIP_HF_FILTER=1 — push **HEAD** without building the size-filtered export (same as old
#                    "push whole branch"; large blobs in history are still sent)
#   QUIET, VERBOSE, MAX_BYTES, HF_USER, PYTHON
set -euo pipefail

NAMESPACE="${1:-}"
SPACE_NAME="${2:-}"
: "${NAMESPACE:?usage: $0 <namespace> <space_name>}"
: "${SPACE_NAME:?usage: $0 <namespace> <space_name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"

HF_USER="${HF_USER:-$NAMESPACE}"
MAX_BYTES="${MAX_BYTES:-$((10 * 1024 * 1024))}"
PYTHON="${PYTHON:-python3}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: not inside a git repository" >&2
  exit 1
}
cd "$ROOT"

unstage_staged_file_over_max() {
  local f s
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    s=0
    if [ -L "$f" ] || ( [ -e "$f" ] && [ ! -d "$f" ] ); then
      s=$(stat -f%z -- "$f" 2>/dev/null) || s=0
    fi
    if [ -n "$s" ] && [ "$s" -gt "$MAX_BYTES" ] 2>/dev/null; then
      git reset -q HEAD -- "$f" 2>/dev/null || true
    fi
  done < <(git diff --cached --name-only 2>/dev/null)
}

if [[ "${SNAPSHOT_COMMIT:-0}" == "1" ]]; then
  git add -A
  unstage_staged_file_over_max
  if git diff --cached --quiet; then
    echo "SNAPSHOT_COMMIT: nothing to commit" >&2
  else
    git commit -m "chore: snapshot before push to Hugging Face Space"
  fi
fi

if [[ "${SKIP_HF_FILTER:-0}" == "1" ]]; then
  PUBLISH_REF="HEAD"
else
  if ! command -v "$PYTHON" &>/dev/null; then
    echo "error: $PYTHON is required to build the HF size-filtered export" >&2
    exit 1
  fi
  if [[ ! -f "$ROOT/scripts/hf_export_build_commit.py" ]]; then
    echo "error: missing $ROOT/scripts/hf_export_build_commit.py" >&2
    exit 1
  fi
  export MAX_BYTES
  PUBLISH_REF=$("$PYTHON" "$ROOT/scripts/hf_export_build_commit.py")
  if [ "${#PUBLISH_REF}" -ne 40 ]; then
    echo "error: export commit not produced" >&2
    exit 1
  fi
fi

REMOTE="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${NAMESPACE}/${SPACE_NAME}"
FSTATE="on"
if [[ "${SKIP_HF_FILTER:-0}" == "1" ]]; then
  FSTATE="off (HEAD as-is)"
fi
echo "Pushing ${PUBLISH_REF} (size filter: ${FSTATE}) -> huggingface.co/${NAMESPACE}/${SPACE_NAME} main"
git push --force "$REMOTE" "${PUBLISH_REF}:main"
