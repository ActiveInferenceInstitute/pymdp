"""Verify fixture evidence from a dense GNN PyMDP scaling sweep."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "gnn_pymdp_scaling_manifest_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PRODUCER_FILES = {
    "runner_sha256": "scripts/experiments/run_pymdp_gnn_scaling_analysis.py",
    "generator_sha256": "scripts/pymdp_spec_generator.py",
}


class ScalingManifestError(ValueError):
    """The manifest or fixture evidence does not satisfy the dense contract."""


@dataclass(frozen=True)
class VerifiedScalingSweep:
    """Producer status and verified (state count, timestep count) cells."""

    status: str
    cells: tuple[tuple[int, int], ...]


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScalingManifestError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _axis(configuration: dict[str, Any], key: str) -> set[int]:
    values = configuration.get(key)
    try:
        if not isinstance(values, str):
            raise ValueError
        axis = [int(token) for token in values.split(",")]
    except ValueError as exc:
        raise ScalingManifestError(f"{key} must be a comma-separated axis") from exc
    if not all(map(_positive, axis)) or len(set(axis)) != len(axis):
        raise ScalingManifestError(f"{key} must contain unique positive counts")
    return set(axis)


def _cell(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(map(_positive, value)):
        raise ScalingManifestError("Each grid cell needs positive integer N and T")
    return value[0], value[1]


def _row_cell(row: dict[str, Any]) -> tuple[int, int]:
    return _cell([row.get("num_states"), row.get("num_timesteps")])


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _check_producer(producer: object, producer_root: Path | None) -> None:
    if not isinstance(producer, dict):
        raise ScalingManifestError("Missing producer metadata")
    python = producer.get("python")
    versions = producer.get("distribution_versions")
    names = ("generalized-notation-notation", "inferactively-pymdp", "jax", "jaxlib")
    if (
        not isinstance(python, str)
        or not python
        or not isinstance(versions, dict)
        or any(name not in versions for name in names)
        or any(
            value is not None and not isinstance(value, str)
            for value in versions.values()
        )
    ):
        raise ScalingManifestError("Malformed producer runtime metadata")
    for field, relative_path in _PRODUCER_FILES.items():
        digest = producer.get(field)
        if not _sha256(digest):
            raise ScalingManifestError(f"Invalid producer digest {field}")
        if producer_root is not None:
            actual = hashlib.sha256(
                (producer_root / relative_path).read_bytes()
            ).hexdigest()
            if digest != actual:
                raise ScalingManifestError(
                    f"Producer source differs for {relative_path}"
                )


def _check_fixture(path: Path, fixture: dict[str, Any], n: int, t: int) -> None:
    """Read the exact bytes and witness the generated source's declared axes."""
    if path.is_symlink() or not path.is_file():
        raise ScalingManifestError(f"{path.name} must be an ordinary fixture file")
    size = fixture.get("bytes")
    if not _positive(size) or not _sha256(fixture.get("sha256")):
        raise ScalingManifestError(f"Invalid fixture identity for {path.name}")
    if "source_seed" not in fixture or fixture["source_seed"] is not None:
        raise ScalingManifestError(f"{path.name} requires null source_seed")

    expected = {
        "GNNSection": {"ActInfPOMDP": "ActInfPOMDP"},
        "StateSpaceBlock": {
            "s": f"s[{n},1,type=float]",
            "o": f"o[{n},1,type=int]",
            "A": f"A[{n},{n},type=float]",
            "B": f"B[{n},{n},{n},type=float]",
            "C": f"C[{n},type=float]",
            "D": f"D[{n},type=float]",
        },
        "Time": {"ModelTimeHorizon": f"ModelTimeHorizon={t}"},
        "ModelParameters": {
            "num_hidden_states": f"num_hidden_states:{n}",
            "num_obs": f"num_obs:{n}",
            "num_actions": f"num_actions:{n}",
            "num_timesteps": f"num_timesteps:{t}",
        },
    }
    digest = hashlib.sha256()
    byte_count = 0
    section = ""
    seen_sections: set[str] = set()
    seen: set[tuple[str, str]] = set()
    with path.open("rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            byte_count += len(raw_line)
            if byte_count > size:
                raise ScalingManifestError(f"Wrong fixture byte count for {path.name}")
            line = raw_line.decode("utf-8").strip()
            if line.startswith("## "):
                section = line[3:].strip()
                if section in expected:
                    if section in seen_sections:
                        raise ScalingManifestError(f"{path.name} repeats {section}")
                    seen_sections.add(section)
            elif section in expected and line and not line.startswith("#"):
                value = re.sub(r"\s*([,\[\]=:])\s*", r"\1", line)
                key = re.split(r"[\[=:]", value, maxsplit=1)[0]
                expected_value = expected[section].get(key)
                if section == "GNNSection" and expected_value is None:
                    raise ScalingManifestError(f"{path.name} must declare ActInfPOMDP")
                if expected_value is not None:
                    if (section, key) in seen or value != expected_value:
                        raise ScalingManifestError(
                            f"{path.name} has conflicting {section} declaration {key}"
                        )
                    seen.add((section, key))
    if byte_count != size or digest.hexdigest() != fixture["sha256"]:
        raise ScalingManifestError(f"Fixture bytes or SHA-256 differ for {path.name}")
    required = {(group, key) for group, values in expected.items() for key in values}
    if seen != required:
        raise ScalingManifestError(f"{path.name} lacks a required axis declaration")


def verify_scaling_manifest(
    manifest_path: Path,
    fixture_root: Path,
    *,
    producer_root: Path | None = None,
) -> VerifiedScalingSweep:
    """Verify a successful or generation-only dense sweep's fixture evidence.

    The caller supplies the fixture directory. When producer_root is given,
    the two producer file hashes are also checked against its source files.
    Source checks cover the generated flat model dimensions and horizon.
    """
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_json_object
        )
        return _verify_manifest(manifest, fixture_root, producer_root)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScalingManifestError(f"Could not read scaling evidence: {exc}") from exc


def _verify_manifest(
    manifest: object, fixture_root: Path, producer_root: Path | None
) -> VerifiedScalingSweep:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
        raise ScalingManifestError(f"Expected scaling manifest schema {SCHEMA}")
    status = manifest.get("status")
    if status not in ("generated", "success"):
        raise ScalingManifestError(f"Incomplete scaling run status {status!r}")
    phases = manifest.get("phases")
    if not isinstance(phases, list) or bool(phases) != (status == "success"):
        raise ScalingManifestError("Phase ledger disagrees with scaling run status")
    if any(
        not isinstance(phase, dict)
        or phase.get("status") != "success"
        or type(phase.get("exit_code")) is not int
        or phase["exit_code"] != 0
        for phase in phases
    ):
        raise ScalingManifestError("Completed scaling run has an unsuccessful phase")
    if manifest.get("axes") != {"num_states": "count", "num_timesteps": "steps"} or (
        manifest.get("execution_time_unit") != "seconds"
    ):
        raise ScalingManifestError("Unexpected scaling axes or timing units")
    _check_producer(manifest.get("producer"), producer_root)

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ScalingManifestError("Missing scaling configuration")
    n_values = _axis(configuration, "n_values")
    t_values = _axis(configuration, "t_values")
    planned = manifest.get("planned_pairs")
    grid_pairs = manifest.get("grid_pairs")
    planned_count = manifest.get("planned_count")
    if (
        not isinstance(planned, list)
        or not _positive(planned_count)
        or planned_count != len(planned)
        or not _positive(grid_pairs)
        or grid_pairs != len(n_values) * len(t_values)
    ):
        raise ScalingManifestError("Inconsistent grid or planned count")
    cells = tuple(_cell(row) for row in planned)
    if len(set(cells)) != len(cells) or any(
        n not in n_values or t not in t_values for n, t in cells
    ):
        raise ScalingManifestError("Duplicate or unrequested planned cell")
    skipped = manifest.get("skipped_pairs")
    if not isinstance(skipped, list) or len(skipped) != grid_pairs - planned_count:
        raise ScalingManifestError("Skip ledger does not account for all cells")
    accounted = set(cells)
    for row in skipped:
        if not isinstance(row, dict) or row.get("reason") not in (
            "max_n",
            "max_size",
            "timeout_bounds",
        ):
            raise ScalingManifestError("Invalid skipped cell or reason")
        n, t = _row_cell(row)
        if (n, t) in accounted or n not in n_values or t not in t_values:
            raise ScalingManifestError("Duplicate or unrequested skipped cell")
        accounted.add((n, t))

    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != len(cells):
        raise ScalingManifestError("Expected one fixture per planned cell")
    remaining = set(cells)
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ScalingManifestError("Malformed fixture row")
        n, t = _row_cell(fixture)
        if (n, t) not in remaining:
            raise ScalingManifestError("Duplicate or unplanned fixture cell")
        remaining.remove((n, t))
        filename = f"pymdp_scaling_N{n}_T{t}.md"
        if fixture.get("path") != filename:
            raise ScalingManifestError(f"Expected the relative fixture path {filename}")
        _check_fixture(fixture_root / filename, fixture, n, t)
    return VerifiedScalingSweep(status, cells)
