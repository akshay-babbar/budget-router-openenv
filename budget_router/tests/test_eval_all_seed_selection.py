import importlib.util
from pathlib import Path

import pytest


def _load_eval_all():
    path = Path(__file__).resolve().parents[2] / "eval" / "eval_all.py"
    spec = importlib.util.spec_from_file_location("eval_all", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_values_override_named_seed_set():
    eval_all = _load_eval_all()

    assert eval_all.select_seeds(
        seed_set="dev",
        seeds=3,
        seed_values="200,201,202",
    ) == [200, 201, 202]


def test_seed_values_accept_commas_and_whitespace():
    eval_all = _load_eval_all()

    assert eval_all.select_seeds(
        seed_set="heldout",
        seeds=1,
        seed_values="200, 201  202",
    ) == [200, 201, 202]


def test_seed_values_reject_empty_input():
    eval_all = _load_eval_all()

    with pytest.raises(ValueError, match="No explicit seeds"):
        eval_all.select_seeds(seed_set="dev", seeds=3, seed_values=" , ")

