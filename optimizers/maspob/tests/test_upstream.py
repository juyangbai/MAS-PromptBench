"""Pinned upstream core files: attribution header, SHA-256 pins, loading and missing dependencies."""

from __future__ import annotations

import hashlib
import sys

import pytest

from .. import native
from ..regime import CORE_SOURCE_SHA256, GENERATION_TEMPLATE_SHA256, HEADER_END, SOURCE_COMMIT, SOURCE_URL
from .fakes import requires_gnn


def test_vendored_files_are_exactly_the_four_pinned_core_files():
    vendored = sorted(str(path.relative_to(native.UPSTREAM_ROOT)) for path in native.UPSTREAM_ROOT.rglob("*.py"))
    assert vendored == sorted(CORE_SOURCE_SHA256)


@pytest.mark.parametrize("rel_path", sorted(CORE_SOURCE_SHA256))
def test_upstream_content_is_byte_identical_below_a_comment_only_attribution_header(rel_path):
    data = (native.UPSTREAM_ROOT / rel_path).read_bytes()
    header, marker, body = data.partition(HEADER_END)
    assert marker and all(line.startswith(b"#") for line in header.splitlines())
    text = header.decode("utf-8")
    for required in (SOURCE_URL, SOURCE_COMMIT, "arXiv:2603.02630", "research purposes", rel_path):
        assert required in text
    assert hashlib.sha256(body).hexdigest() == CORE_SOURCE_SHA256[rel_path]
    assert native.upstream_source(rel_path)[1] == CORE_SOURCE_SHA256[rel_path]


def test_integrity_check_fails_closed_on_a_modified_file(monkeypatch, tmp_path):
    for rel_path in CORE_SOURCE_SHA256:
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((native.UPSTREAM_ROOT / rel_path).read_bytes())
    tampered = tmp_path / "scripts/utils/training.py"
    tampered.write_bytes(tampered.read_bytes() + b"\n# local edit\n")
    monkeypatch.setattr(native, "UPSTREAM_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="integrity check failed"):
        native.verified_core_source_hashes()
    unheaded = tmp_path / "scripts/gnn_model.py"
    unheaded.write_bytes(unheaded.read_bytes().partition(HEADER_END)[2])
    with pytest.raises(RuntimeError, match="attribution header"):
        native.upstream_source("scripts/gnn_model.py")


def test_prompt_modules_load_without_torch_and_restore_the_scripts_namespace():
    before = {name: module for name, module in sys.modules.items() if name == "scripts" or name.startswith("scripts.")}
    styles, generator = native.load_prompt_modules()
    after = {name: module for name, module in sys.modules.items() if name == "scripts" or name.startswith("scripts.")}
    assert after == before
    assert hashlib.sha256(styles.ITERATIVE_GENERATE_PROMPT.encode("utf-8")).hexdigest() == GENERATION_TEMPLATE_SHA256
    assert len(styles.STYLE_DIMENSIONS) == 10
    assert all(len(options) == 10 for options in styles.STYLE_DIMENSIONS.values())
    assert len(styles.PRESET_STYLE_NAMES) == 10
    assert generator.clean_generated_prompt("```\nHere is the prompt:\nSolve it.\n```") == "Solve it."
    assert native.load_prompt_modules() == (styles, generator)


def test_missing_torch_geometric_raises_a_clear_dependency_error(monkeypatch):
    monkeypatch.setattr(native, "_MODEL_MODULES", None)
    monkeypatch.setitem(sys.modules, "torch_geometric", None)
    with pytest.raises(native.MASPOBDependencyError, match="pip install .*torch_geometric"):
        native.load_model_modules()
    assert native._MODEL_MODULES is None
    assert "torch_geometric" in native.missing_dependencies()


@requires_gnn
def test_model_modules_expose_the_upstream_gat_and_linucb_api():
    gnn, training = native.load_model_modules()
    for name in (
        "WorkflowGAT",
        "initialize_fisher",
        "update_fisher",
        "compute_prediction_and_uncertainty",
        "build_combined_embedding",
        "select_best_prompt_for_operator",
    ):
        assert callable(getattr(gnn, name))
    assert callable(training.train_with_early_stopping)
    assert gnn.GATv2Conv.__name__ == "GATv2Conv"
