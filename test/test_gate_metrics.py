"""Gate provenance checks against independent, temporary Git repositories."""

import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gate_metrics.py"


def _git(root, *args):
    return subprocess.run(
        [
            "git",
            "-c", "core.fsmonitor=false",
            "-c", "core.autocrlf=false",
            "-c", "commit.gpgsign=false",
            "-c", "user.name=Gate metrics tests",
            "-c", "user.email=gate-metrics@example.invalid",
            *args,
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def metrics(monkeypatch):
    spec = importlib.util.spec_from_file_location("gate_metrics_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(metrics, monkeypatch, tmp_path):
    # No inherited Git directory, object store, template, hooks, or user config.
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    _git(tmp_path, "init", "--quiet", "--template=")
    _write(tmp_path, ".gitignore", "uv.lock\npymdp/generated-cache/\n")
    _write(tmp_path, "gate.sh", "#!/bin/sh\nexit 0\n").chmod(0o755)
    _write(tmp_path, "pyproject.toml", '[project]\nname = "metrics-fixture"\nversion = "0.0.0"\n')
    _write(tmp_path, "pymdp/source.py", "VALUE = 1\n")
    _write(tmp_path, "uv.lock", "version = 1\nrevision = 1\n")
    doc = _write(tmp_path, metrics.DIGEST_EXCLUDE, "existing metrics record\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "Independent fixture baseline")
    monkeypatch.setattr(metrics, "ROOT", tmp_path)
    monkeypatch.setattr(metrics, "DOC", doc)
    return tmp_path


def test_inputs_digest_reads_working_bytes_at_fixed_head(metrics, repository):
    head = _git(repository, "rev-parse", "HEAD")
    before = metrics.inputs_digest()
    source = repository / "pymdp/source.py"
    original = source.stat()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert _git(repository, "rev-parse", "HEAD") == head
    assert source.stat().st_size == original.st_size
    assert source.stat().st_mtime_ns == original.st_mtime_ns
    assert metrics.inputs_digest() != before


def test_inputs_digest_reads_ignored_uv_lock(metrics, repository):
    assert _git(repository, "check-ignore", "uv.lock") == "uv.lock"
    head = _git(repository, "rev-parse", "HEAD")
    before = metrics.inputs_digest()
    _write(repository, "uv.lock", "version = 1\nrevision = 2\n")

    assert _git(repository, "rev-parse", "HEAD") == head
    assert metrics.inputs_digest() != before


def test_inputs_digest_tracks_uv_lock_presence(metrics, repository):
    before = metrics.inputs_digest()
    (repository / "uv.lock").unlink()
    assert metrics.inputs_digest() != before

    _write(repository, "uv.lock", "version = 1\nrevision = 1\n")
    assert metrics.inputs_digest() == before


@pytest.mark.parametrize(
    "relative",
    [
        "pymdp/added.py",
        "test/test_added.py",
        "scripts/added.py",
        "examples/added.txt",
        "docs-mkdocs/added.md",
    ],
    ids=["library", "test", "script", "example", "documentation"],
)
def test_inputs_digest_admits_new_inputs(metrics, repository, relative):
    head = _git(repository, "rev-parse", "HEAD")
    before = metrics.inputs_digest()
    _write(repository, relative, "# New admitted input\n")
    added = metrics.inputs_digest()

    assert added != before
    assert _git(repository, "rev-parse", "HEAD") == head
    _git(repository, "add", "--", relative)
    assert metrics.inputs_digest() == added


def test_inputs_digest_tracks_deleted_inputs(metrics, repository):
    head = _git(repository, "rev-parse", "HEAD")
    before = metrics.inputs_digest()
    (repository / "pymdp/source.py").unlink()
    deleted = metrics.inputs_digest()

    assert deleted != before
    assert _git(repository, "rev-parse", "HEAD") == head
    _git(repository, "add", "--update", "--", "pymdp/source.py")
    assert metrics.inputs_digest() == deleted


def test_inputs_digest_tracks_executable_mode(metrics, repository):
    before = metrics.inputs_digest()
    script = repository / "gate.sh"
    script.chmod(script.stat().st_mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)
    assert metrics.inputs_digest() != before


def test_inputs_digest_excludes_generated_page(metrics, repository):
    before = metrics.inputs_digest()
    metrics.DOC.write_text("replaced generated record\n", encoding="utf-8")
    assert metrics.inputs_digest() == before
    metrics.DOC.unlink()
    assert metrics.inputs_digest() == before


@pytest.mark.parametrize(
    "relative",
    ["=0.15,", "unrelated-data.json", "pymdp/generated-cache/data.json"],
    ids=["preserved-overlay", "unrelated-root-data", "ignored-output"],
)
def test_inputs_digest_excludes_unrelated_overlays(metrics, repository, relative):
    before = metrics.inputs_digest()
    path = _write(repository, relative, "unrelated bytes\n")
    assert metrics.inputs_digest() == before
    path.write_text("different unrelated bytes\n", encoding="utf-8")
    assert metrics.inputs_digest() == before
    path.unlink()
    assert metrics.inputs_digest() == before


def test_check_rejects_changed_working_input_without_rewriting_record(metrics, repository, capsys):
    text = "\n".join(f"`{arm.name}`" for arm in metrics.ARMS)
    text += f"\n| Inputs digest | `{metrics.inputs_digest()}` |\n"
    metrics.DOC.write_text(text, encoding="utf-8")
    assert metrics.check() == 0
    _write(repository, "pymdp/source.py", "VALUE = 2\n")

    assert metrics.check() == 1
    assert metrics.DOC.read_text(encoding="utf-8") == text
    assert "does not match the tree" in capsys.readouterr().err


def _prepare_main(metrics, repository, monkeypatch, command):
    _write(repository, "gate.sh", f"#!/bin/sh\nset -eu\n{command}\n").chmod(0o755)
    monkeypatch.setattr(metrics, "ARMS", (metrics.Arm("fixture", [sys.executable, "-B", "-c", "pass"]),))
    # Version probing must not require uv or a project environment in this fixture.
    monkeypatch.setattr(metrics, "python_version", lambda: sys.version.split()[0])
    monkeypatch.setattr(metrics, "uv_version", lambda: "uv fixture")
    monkeypatch.setattr(sys, "argv", ["gate_metrics.py"])


@pytest.mark.parametrize(
    "command",
    [
        "printf 'VALUE = 2\\n' > pymdp/source.py",
        "printf 'version = 1\\nrevision = 2\\n' > uv.lock",
        "printf 'VALUE = 3\\n' > pymdp/added.py",
        "rm pymdp/source.py",
        "rm pymdp/source.py; mkdir pymdp/source.py",
    ],
    ids=["source-change", "lock-change", "input-addition", "input-deletion", "nonfile-replacement"],
)
def test_main_refuses_publication_when_inputs_change(metrics, repository, monkeypatch, capsys, command):
    _prepare_main(metrics, repository, monkeypatch, command)
    original = metrics.DOC.read_bytes()

    assert metrics.main() == 1
    assert metrics.DOC.read_bytes() == original
    assert not metrics.DOC.with_suffix(".md.tmp").exists()
    captured = capsys.readouterr()
    assert "metrics not published" in captured.err
    assert "wrote " not in captured.out


def test_main_records_inputs_bound_after_metadata(metrics, repository, monkeypatch):
    _prepare_main(metrics, repository, monkeypatch, "exit 0")

    def version_after_edit():
        _write(repository, "pymdp/source.py", "VALUE = 2\n")
        return sys.version.split()[0]

    monkeypatch.setattr(metrics, "python_version", version_after_edit)
    assert metrics.main() == 0
    text = metrics.DOC.read_text(encoding="utf-8")
    assert f"| Inputs digest | `{metrics.inputs_digest()}` |" in text
    assert "| `fixture` |" in text
