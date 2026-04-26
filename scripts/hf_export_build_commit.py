#!/usr/bin/env python3
"""
Build a *new* commit (orphan) from the current `HEAD` tree, omitting:
  - any binary blob (HF rejects plain-Git binaries unless Xet is used)
  - any blob with size > MAX_BYTES (default 10 MiB)
  - any subtree whose *sum* of **remaining** blob sizes under a directory prefix exceeds MAX_BYTES
    (repeatedly removes the shortest over-budget prefix first)

Prints the new commit object id (40 hex) to stdout. By default, stderr prints a one-line
omit summary; set QUIET=1 to suppress it or VERBOSE=1 to list omitted paths.

Run from the repository root (or any path inside the repo). Uses a temporary
GIT_INDEX_FILE; does not modify the working tree or the current branch.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
@dataclass(frozen=True)
class Blob:
    path: str
    size: int
    mode: str
    oid: str


def sh(args: list[str], cwd: str, env: dict[str, str] | None = None) -> str:
    menv = {**os.environ, "LC_ALL": "C"} if env is None else env
    return (
        subprocess.check_output(
            args,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            env=menv,
        ).decode("utf-8", "replace")
    ).strip()


def list_blobs(cwd: str) -> list[Blob]:
    out = sh(["git", "ls-tree", "-l", "-r", "HEAD"], cwd=cwd)
    blobs: list[Blob] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        toks = meta.split()
        if len(toks) < 4:
            continue
        mode, otype, _oid, size_s = toks[0], toks[1], toks[2], toks[3]
        if otype == "commit":
            continue
        if otype != "blob":
            continue
        blobs.append(Blob(path=path, size=int(size_s), mode=mode, oid=toks[2]))
    return blobs


def is_binary_blob(cwd: str, oid: str) -> bool:
    data = subprocess.check_output(
        ["git", "cat-file", "blob", oid],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
    )[:8192]
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def parent_prefixes(path: str) -> list[str]:
    if "/" not in path:
        return []
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def apply_rules(
    cwd: str,
    blobs: list[Blob],
    max_b: int,
    *,
    verbose: bool,
) -> tuple[dict[str, Blob], dict[str, int]]:
    """Return keep map and omitted counts by reason."""
    keep: dict[str, Blob] = {}
    omitted = {"binary": 0, "file": 0, "subtree": 0}
    for b in blobs:
        if b.size > max_b:
            omitted["file"] += 1
            if verbose:
                print(f"Omit: file {b.path!r} ({b.size} B) > {max_b} B", file=sys.stderr)
        elif is_binary_blob(cwd, b.oid):
            omitted["binary"] += 1
            if verbose:
                print(f"Omit: binary file {b.path!r}", file=sys.stderr)
        else:
            keep[b.path] = b

    while True:
        psum: dict[str, int] = defaultdict(int)
        for p, b in keep.items():
            for d in parent_prefixes(p):
                psum[d] += b.size
        bad = [d for d, t in psum.items() if t > max_b]
        if not bad:
            break
        victim = min(bad, key=lambda d: (len(d.split("/")), d))
        to_del = [p for p in keep if p == victim or p.startswith(victim + "/")]
        for p in to_del:
            del keep[p]
            omitted["subtree"] += 1
        if verbose:
            print(
                f"Omit: subtree {victim!r} (sum of kept blobs under prefix > {max_b} B)",
                file=sys.stderr,
            )

    return keep, omitted


def make_export_commit(cwd: str, paths: list[str]) -> str:
    if not paths:
        raise SystemExit("error: nothing to commit after size filter (empty tree)")

    empty = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # empty tree object
    with tempfile.TemporaryDirectory() as tdir:
        gidx = os.path.join(tdir, "i")
        env: dict[str, str] = {**os.environ, "GIT_INDEX_FILE": gidx}
        subprocess.check_call(
            ["git", "read-tree", empty], cwd=cwd, env=env, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        for p in paths:
            r = subprocess.run(
                ["git", "ls-tree", "HEAD", "--", p],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 or not r.stdout.strip():
                continue
            line = r.stdout.strip()
            mline, path_f = line.split("\t", 1)
            p_use = path_f if path_f else p
            toks = mline.split()
            if len(toks) < 3:
                continue
            mode, otype, oid = toks[0], toks[1], toks[2]
            if otype != "blob":
                continue
            subprocess.check_call(
                ["git", "update-index", "--add", "--cacheinfo", f"{mode},{oid},{p_use}"],
                cwd=cwd,
                env=env,
                stdout=subprocess.DEVNULL,
            )
        tree = subprocess.check_output(["git", "write-tree"], cwd=cwd, env=env).decode().strip()
        msg = os.environ.get("HF_EXPORT_MSG", "chore: HF Space export (size filter)")
        commit = sh(["git", "commit-tree", tree, "-m", msg], cwd=cwd, env=env)
        if len(commit) < 4:
            raise SystemExit("error: could not build export commit")
        return commit


def main() -> None:
    top = sh(["git", "rev-parse", "--show-toplevel"], cwd=os.path.abspath("."))
    if not top:
        raise SystemExit("error: not a git repository")
    cwd = top
    max_b = int(os.environ.get("MAX_BYTES", str(10 * 1024 * 1024)))
    verbose = os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes", "y")
    quiet = os.environ.get("QUIET", "0").lower() in ("1", "true", "yes", "y")

    blobs = list_blobs(cwd)
    _keep, omitted = apply_rules(cwd, blobs, max_b, verbose=verbose)
    paths = sorted(_keep)
    n_om = sum(omitted.values())
    if not quiet and n_om:
        print(
            "HF export: omitted "
            f"{n_om} path(s) "
            f"(binary={omitted['binary']}, file>{max_b}B={omitted['file']}, "
            f"subtree>{max_b}B={omitted['subtree']})",
            file=sys.stderr,
        )

    oid = make_export_commit(cwd, paths)
    print(oid, end="")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:  # pragma: no cover
        print("error: git command failed", file=sys.stderr)
        raise SystemExit(1) from e
