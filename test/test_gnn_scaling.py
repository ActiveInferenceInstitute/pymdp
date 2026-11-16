"""Independent tests of the dense GNN sweep manifest reader."""

import hashlib
import json

import pytest

from pymdp.gnn_scaling import ScalingManifestError, verify_scaling_manifest

_SOURCE = """# Dense sweep fixture
## GNNSection
ActInfPOMDP

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=int]
A[2,2,type=float]
B[2,2,2,type=float]
C[2,type=float]
D[2,type=float]

## Time
ModelTimeHorizon=3

## ModelParameters
num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 3
"""


@pytest.fixture
def sweep(tmp_path):
    fixture_dir = tmp_path / "relocated-fixtures"
    fixture_dir.mkdir()
    model = fixture_dir / "pymdp_scaling_N2_T3.md"
    model.write_bytes(_SOURCE.encode())
    fixtures = [
        {
            "path": model.name,
            "num_states": 2,
            "num_timesteps": 3,
            "source_seed": None,
        }
    ]
    manifest = {
        "schema_version": "gnn_pymdp_scaling_manifest_v1",
        "status": "generated",
        "axes": {"num_states": "count", "num_timesteps": "steps"},
        "execution_time_unit": "seconds",
        "producer": {
            "runner_sha256": "0" * 64,
            "generator_sha256": "1" * 64,
            "python": "3.14.0",
            "distribution_versions": {
                "generalized-notation-notation": None,
                "inferactively-pymdp": "1.0.5",
                "jax": None,
                "jaxlib": None,
            },
        },
        "configuration": {"n_values": "2,9", "t_values": "3"},
        "paths": {"generated_specs": "/unrelated/producer-only/path"},
        "grid_pairs": 2,
        "planned_count": 1,
        "planned_pairs": [[2, 3]],
        "skipped_pairs": [{"num_states": 9, "num_timesteps": 3, "reason": "max_n"}],
        "phases": [],
        "fixtures": fixtures,
    }
    path = tmp_path / "manifest.json"
    _bind_bytes(fixtures[0], model)
    return path, model, manifest


def _bind_bytes(fixture, path):
    data = path.read_bytes()
    fixture.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _read(sweep):
    path, model, manifest = sweep
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return verify_scaling_manifest(path, model.parent)


def test_verifies_relocated_fixture_and_explicitly_skipped_cell(sweep):
    result = _read(sweep)
    assert result.cells == ((2, 3),)
    assert result.status == "generated"


def test_completed_run_with_a_successful_phase(sweep):
    phase = {"label": "main", "status": "success", "exit_code": 0}
    sweep[2].update(status="success", phases=[phase])
    assert _read(sweep).status == "success"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "other", "Expected scaling manifest schema"),
        ("status", "failed", "Incomplete scaling run status"),
        ("status", "success", "Phase ledger"),
        ("execution_time_unit", "milliseconds", "axes or timing units"),
        ("grid_pairs", True, "Inconsistent grid"),
        ("configuration", {"n_values": "2,2", "t_values": "3"}, "unique positive"),
        ("planned_pairs", [[2, True]], "positive integer"),
        ("planned_pairs", [[9, 9]], "unrequested planned"),
        ("skipped_pairs", [], "Skip ledger"),
        (
            "skipped_pairs",
            [{"num_states": 2, "num_timesteps": 3, "reason": "max_n"}],
            "Duplicate or unrequested skipped",
        ),
        (
            "skipped_pairs",
            [{"num_states": 9, "num_timesteps": 3, "reason": "unknown"}],
            "Invalid skipped cell",
        ),
        ("fixtures", [], "one fixture per planned cell"),
    ],
)
def test_refuses_inconsistent_manifest(sweep, field, value, message):
    sweep[2][field] = value
    with pytest.raises(ScalingManifestError, match=message):
        _read(sweep)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "../pymdp_scaling_N2_T3.md", "relative fixture path"),
        ("path", "/tmp/pymdp_scaling_N2_T3.md", "relative fixture path"),
        ("num_states", 9, "unplanned fixture"),
        ("bytes", 1, "byte count"),
        ("sha256", "e" * 64, "SHA-256 differ"),
        ("source_seed", 7, "requires null source_seed"),
    ],
)
def test_refuses_inconsistent_fixture_identity(sweep, field, value, message):
    sweep[2]["fixtures"][0][field] = value
    with pytest.raises(ScalingManifestError, match=message):
        _read(sweep)


def test_refuses_symlink_even_when_it_matches_digest(sweep):
    _, model, _ = sweep
    target = model.with_name("alternate.md")
    model.rename(target)
    try:
        model.symlink_to(target)
    except OSError:
        pytest.skip("Fixture symlinks unavailable on this filesystem")
    with pytest.raises(ScalingManifestError, match="ordinary fixture file"):
        _read(sweep)


@pytest.mark.parametrize(
    ("original", "changed", "message"),
    [
        ("ModelTimeHorizon=3", "ModelTimeHorizon=8", "Time declaration"),
        ("B[2,2,2", "B[2,2,4", "StateSpaceBlock declaration B"),
        ("num_timesteps: 3", "num_timesteps: 6", "ModelParameters declaration"),
        ("num_obs: 2", "num_ obs: 2", "lacks a required axis declaration"),
        ("ActInfPOMDP", "ActInfContinuous", "must declare ActInfPOMDP"),
    ],
)
def test_refuses_semantic_mismatch_with_valid_fixture_digest(
    sweep, original, changed, message
):
    _, model, manifest = sweep
    model.write_bytes(_SOURCE.replace(original, changed).encode())
    _bind_bytes(manifest["fixtures"][0], model)
    with pytest.raises(ScalingManifestError, match=message):
        _read(sweep)
