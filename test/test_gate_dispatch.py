"""Exercise the real shell dispatchers with a fail-closed, inert uv executable.

These tests prove argv/environment forwarding, not pytest collection, GNN parser
behavior, notebook execution, or a real MkDocs build. All scripts run in a copied
temporary tree. PATH contains only the inert uv recorder and a small allowlist
of shell utilities; the recorder never executes any requested Python or tool.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]

_UV_RECORDER = r'''
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["GATE_TEST_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "argv": args,
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "gnn_root": os.environ.get("PYMDP_GNN_ROOT"),
        "gnn_checkout": os.environ.get("PYMDP_GNN_CHECKOUT"),
        "uv_no_sync": os.environ.get("UV_NO_SYNC"),
        "mkdocs_warning": os.environ.get("NO_MKDOCS_2_WARNING"),
    }) + "\n")

if args == ["run", "--no-sync", "python", "-c", "import mkdocs"]:
    raise SystemExit(int(os.environ.get("GATE_TEST_MKDOCS_IMPORT_EXIT", "0")))

pin_prefix = ["run", "--group", "test", "--extra", "gnn", "python", "-c"]
if args[:7] == pin_prefix and len(args) == 8:
    if "from pymdp.gnn import GNN_PINNED_RELEASE, GNN_PINNED_REVISION" not in args[7]:
        raise SystemExit(96)
    print(os.environ["GATE_TEST_PIN_SHA"])
    print(os.environ["GATE_TEST_PIN_TAG"])
    raise SystemExit(0)

export_prefix = ["run", "--group", "test", "--extra", "gnn", "python", "-"]
if args[:7] == export_prefix and len(args) == 9:
    if "gnn.export_pinned_source" not in sys.stdin.read():
        raise SystemExit(96)
    destination = Path(args[8])
    if not destination.is_dir() or any(destination.iterdir()):
        raise SystemExit(95)
    # This directory is an argv fixture, not an exported GNN implementation.
    source = destination / "src"
    source.mkdir()
    print(source.resolve())
    raise SystemExit(0)

allowed_prefixes = (
    ["run", "--group", "test", "pytest", "test"],
    ["run", "--group", "dev", "ruff", "check", "--force-exclude"],
    ["run", "--group", "test", "--extra", "nb", "python",
     "scripts/run_notebook_manifest.py"],
    ["run", "--group", "test", "--extra", "gnn", "pytest",
     "test/test_gnn_bridge.py", "-v"],
    ["run", "--group", "test", "--extra", "gnn", "python", "-m",
     "gnn.importers", "check", "--repo", ".", "--require-pointer"],
    ["run", "--group", "dev", "--group", "test", "--extra", "nb",
     "--extra", "gnn", "python", "scripts/gate_metrics.py"],
    ["run", "--no-sync", "mkdocs", "build", "--strict"],
)
if any(args[:len(prefix)] == prefix for prefix in allowed_prefixes):
    raise SystemExit(int(os.environ.get("GATE_TEST_TOOL_EXIT", "0")))

print("Refusing unexpected uv invocation: " + repr(args), file=sys.stderr)
raise SystemExit(97)
'''


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash shell")
class TestGateDispatch(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="pymdp-gate-dispatch-")
        self.addCleanup(self.workspace.cleanup)
        self.root = (Path(self.workspace.name) / "sandbox with spaces").resolve()
        self.root.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        for relative in ("gate.sh", "scripts/docs_build.sh"):
            target = self.root / relative
            shutil.copyfile(REPO_ROOT / relative, target)
            target.chmod(0o755)
        sync = scripts / "sync_docs_notebooks.sh"
        sync.write_text(
            '#!/bin/bash\nprintf "sync\\n" >> "$GATE_TEST_SYNC_LOG"\n',
            encoding="utf-8",
        )
        sync.chmod(0o755)

        binaries = self.root / "bin"
        binaries.mkdir()
        for name in ("bash", "dirname", "sed", "mkdir", "mktemp", "env"):
            utility = next(
                (Path(directory) / name for directory in ("/bin", "/usr/bin")
                 if (Path(directory) / name).is_file()),
                None,
            )
            if utility is None:
                self.skipTest(f"required shell utility unavailable: {name}")
            (binaries / name).symlink_to(utility)
        uv = binaries / "uv"
        uv.write_text(f"#!{sys.executable}\n" + _UV_RECORDER, encoding="utf-8")
        uv.chmod(0o755)

        self.log = self.root / "uv.jsonl"
        self.sync_log = self.root / "sync.log"
        temporary = self.root / "tmp"
        temporary.mkdir()
        pin = json.loads((REPO_ROOT / ".github/gnn-pin.json").read_text(encoding="utf-8"))
        # Deliberately do not inherit PATH, BASH_ENV, Python paths, or GNN roots.
        self.environment = {
            "PATH": str(binaries),
            "HOME": str(self.root),
            "TMPDIR": str(temporary),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_NO_SYNC": "1",
            "GATE_TEST_LOG": str(self.log),
            "GATE_TEST_SYNC_LOG": str(self.sync_log),
            "GATE_TEST_PIN_SHA": pin["commit"],
            "GATE_TEST_PIN_TAG": pin["tag"],
        }

    def _calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def _run(self, arm, *arguments, environment=None, direct_docs=False):
        start = len(self._calls())
        env = dict(self.environment)
        env.update(environment or {})
        if direct_docs:
            command = ["/bin/bash", str(self.root / "scripts/docs_build.sh"), *arguments]
        else:
            command = ["/bin/bash", str(self.root / "gate.sh"), arm, *arguments]
        completed = subprocess.run(
            command,
            cwd=self.workspace.name,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return completed, self._calls()[start:]

    def _assert_success(self, completed):
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_optional_separator_is_removed_for_pytest_arms(self):
        options = [
            "-n0", "--collect-only", "-q", "--cov=pymdp", "--cov-report=term",
            "-k", "TestEmission",
        ]
        prefixes = {
            "test": ["run", "--group", "test", "pytest", "test", "-n", "2",
                     "-m", "not nightly"],
            "test-full": ["run", "--group", "test", "pytest", "test", "-n", "2"],
            "test-nightly": ["run", "--group", "test", "pytest", "test", "-n", "2",
                             "-m", "nightly"],
        }
        for arm, prefix in prefixes.items():
            observed = []
            for separator in ([], ["--"]):
                with self.subTest(arm=arm, separator=separator):
                    completed, calls = self._run(arm, *separator, *options)
                    self._assert_success(completed)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0]["argv"], prefix + options)
                    self.assertEqual(calls[0]["cwd"], str(self.root))
                    self.assertEqual(calls[0]["uv_no_sync"], "1")
                    observed.append(calls[0]["argv"])
            self.assertEqual(observed[0], observed[1])

    def test_only_one_leading_separator_is_consumed(self):
        completed, calls = self._run("test", "--", "--", "-q")
        self._assert_success(completed)
        self.assertEqual(calls[0]["argv"][-2:], ["--", "-q"])
        self.assertEqual(calls[0]["argv"].count("--"), 1)

    def test_notebook_internal_separator_is_preserved(self):
        options = ["--nbval-cell-timeout=120", "notebook path with spaces.ipynb"]
        for arm, manifest, defaults in (
            ("notebooks", "test/notebooks/ci_notebooks.txt", []),
            ("notebooks-nightly", "test/notebooks/nightly_notebooks.txt",
             ["--nbval-cell-timeout=3600"]),
        ):
            for separator in ([], ["--"]):
                with self.subTest(arm=arm, separator=separator):
                    completed, calls = self._run(arm, *separator, *options)
                    self._assert_success(completed)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(
                        calls[0]["argv"],
                        ["run", "--group", "test", "--extra", "nb", "python",
                         "scripts/run_notebook_manifest.py", manifest, "--"]
                        + defaults + options,
                    )
                    self.assertEqual(calls[0]["argv"].count("--"), 1)

    def test_lint_and_bench_share_the_optional_separator_boundary(self):
        for arm, prefix, options in (
            ("lint", ["run", "--group", "dev", "ruff", "check", "--force-exclude"],
             ["pymdp/gnn.py", "--select", "F"]),
            ("bench", ["run", "--group", "dev", "--group", "test", "--extra", "nb",
                       "--extra", "gnn", "python", "scripts/gate_metrics.py"],
             ["--check"]),
        ):
            for separator in ([], ["--"]):
                with self.subTest(arm=arm, separator=separator):
                    completed, calls = self._run(arm, *separator, *options)
                    self._assert_success(completed)
                    self.assertEqual([call["argv"] for call in calls], [prefix + options])

    def test_docs_gate_forwards_spaced_site_directory_without_syncing_uv(self):
        site = self.root / "site with spaces"
        options = ["--site-dir", str(site), "--verbose"]
        for separator in ([], ["--"]):
            with self.subTest(separator=separator):
                syncs_before = (
                    len(self.sync_log.read_text(encoding="utf-8").splitlines())
                    if self.sync_log.exists() else 0
                )
                completed, calls = self._run("docs", *separator, *options)
                self._assert_success(completed)
                self.assertEqual(
                    [call["argv"] for call in calls],
                    [["run", "--no-sync", "python", "-c", "import mkdocs"],
                     ["run", "--no-sync", "mkdocs", "build", "--strict"] + options],
                )
                self.assertTrue(all(call["mkdocs_warning"] == "1" for call in calls))
                self.assertEqual(
                    len(self.sync_log.read_text(encoding="utf-8").splitlines()),
                    syncs_before + 1,
                )
                # The recorder does not build a site: this is only an argv oracle.
                self.assertFalse(site.exists())

    def test_docs_script_forwards_arguments_without_gate(self):
        options = ["--site-dir", str(self.root / "direct site with spaces")]
        completed, calls = self._run("", *options, direct_docs=True)
        self._assert_success(completed)
        self.assertEqual(
            calls[-1]["argv"],
            ["run", "--no-sync", "mkdocs", "build", "--strict"] + options,
        )
        self.assertEqual(self.sync_log.read_text(encoding="utf-8"), "sync\n")

    def test_docs_missing_prepared_tool_fails_without_installing(self):
        completed, calls = self._run(
            "docs", "--site-dir", str(self.root / "site"),
            environment={"GATE_TEST_MKDOCS_IMPORT_EXIT": "23"},
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(
            [call["argv"] for call in calls],
            [["run", "--no-sync", "python", "-c", "import mkdocs"]],
        )
        self.assertIn("mkdocs is not installed", completed.stderr)

    def test_gnn_exports_are_fresh_and_old_cache_is_preserved(self):
        old_cache = self.root / ".cache" / "gnn-pin"
        stale = old_cache / "src" / "gnn" / "stale_parser.py"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"preserved old parser module\n")
        sentinel = old_cache / "sentinel"
        sentinel.write_bytes(b"preserved old cache\n")
        checkout = str(self.root / "GNN checkout with spaces")
        exports = []
        for separator in ([], ["--"]):
            with self.subTest(separator=separator):
                completed, calls = self._run(
                    "gnn", *separator, "-n0", "-k", "TestEmission",
                    environment={"PYMDP_GNN_CHECKOUT": checkout},
                )
                self._assert_success(completed)
                self.assertEqual(len(calls), 4)
                export = calls[1]["argv"]
                self.assertEqual(
                    export[:7],
                    ["run", "--group", "test", "--extra", "gnn", "python", "-"],
                )
                self.assertEqual(export[7], checkout)
                destination = (self.root / export[8]).resolve()
                self.assertEqual(destination.parent, old_cache.parent)
                self.assertTrue(destination.name.startswith("gnn-pin."))
                self.assertNotEqual(destination, old_cache)
                exports.append(destination)
                source = str(destination / "src")
                self.assertEqual(calls[2]["pythonpath"], source)
                self.assertEqual(
                    calls[2]["argv"][-4:],
                    ["--expected-commit", self.environment["GATE_TEST_PIN_SHA"],
                     "--expected-tag", self.environment["GATE_TEST_PIN_TAG"]],
                )
                self.assertEqual(calls[3]["pythonpath"], source)
                self.assertEqual(calls[3]["gnn_root"], source)
                self.assertEqual(calls[3]["gnn_checkout"], checkout)
                self.assertEqual(calls[3]["argv"][-4:], ["-v", "-n0", "-k", "TestEmission"])
        self.assertNotEqual(exports[0], exports[1])
        self.assertEqual(stale.read_bytes(), b"preserved old parser module\n")
        self.assertEqual(sentinel.read_bytes(), b"preserved old cache\n")
        self.assertEqual(
            {path.relative_to(old_cache).as_posix() for path in old_cache.rglob("*")
             if path.is_file()},
            {"sentinel", "src/gnn/stale_parser.py"},
        )

    def test_gnn_existing_root_bypasses_export(self):
        source = str(self.root / "existing pinned source")
        completed, calls = self._run(
            "gnn", "--", "-q", environment={"PYMDP_GNN_ROOT": source}
        )
        self._assert_success(completed)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1]["pythonpath"], source)
        self.assertEqual(calls[2]["gnn_root"], source)
        self.assertEqual(calls[2]["argv"][-2:], ["-v", "-q"])
        self.assertFalse((self.root / ".cache").exists())

    def test_underlying_tool_exit_status_is_preserved(self):
        completed, calls = self._run(
            "test", "--", "-q", environment={"GATE_TEST_TOOL_EXIT": "19"}
        )
        self.assertEqual(completed.returncode, 19)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
