"""Record parent-recovery readiness; this command executes no fleet rollouts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parent_preflight(root, config):
    root = Path(root)
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{config['required_parent_revision']}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    missing = [path for path in config["required_parent_paths"] if not (root / path).is_file()]
    missing_patterns = [
        pattern
        for pattern in config["required_parent_artifact_patterns"]
        if not any(root.glob(pattern))
    ]
    provenance_errors = []
    reconstructed = config.get("parent_mode") == "reconstructed"
    if reconstructed:
        for field in ["required_parent_manifest", "required_parent_audit"]:
            path = root / config[field]
            if (
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() != config[field + "_sha256"]
            ):
                provenance_errors.append(f"Changed pinned {field}")
        path = root / config["required_parent_manifest"]
        if path.is_file():
            manifest = json.loads(path.read_text())
            for key, base in [("source_sha256", root), ("output_sha256", root / "results/tables")]:
                for name, expected in manifest[key].items():
                    source = base / name
                    if (
                        not source.is_file()
                        or hashlib.sha256(source.read_bytes()).hexdigest() != expected
                    ):
                        provenance_errors.append(f"Changed parent artifact: {name}")
            if manifest["rollouts"] != 1600 or manifest["confirmation_run"]:
                provenance_errors.append("Unexpected parent scope")
        path = root / config["required_parent_audit"]
        if path.is_file() and not json.loads(path.read_text())["integrity_checks_pass"]:
            provenance_errors.append("Parent integrity audit failed")
    available = (
        resolved.returncode == 0 and not missing and not missing_patterns and not provenance_errors
    )
    return {
        "experiment": "10H-Q",
        "status": (
            "ready_for_frozen_diagnostic" if reconstructed else "ready_for_parent_provenance_audit"
        )
        if available
        else "blocked_missing_parent",
        "required_parent_revision": config["required_parent_revision"],
        "parent_revision_resolves": resolved.returncode == 0,
        "resolved_parent_revision": resolved.stdout.strip() if resolved.returncode == 0 else None,
        "missing_parent_paths": missing,
        "missing_parent_artifact_patterns": missing_patterns,
        "fleet_rollouts_executed": 0,
        "rollout_adapter_implemented": available and reconstructed,
        "frozen_models_refitted": False,
        "confirmation_run": False,
        "parent_reconstructed": reconstructed,
        "provenance_errors": provenance_errors,
        "remaining_work": [
            "Repeat the four frozen reconstructed P cases and verify array equality",
            "Execute the twelve locked Q cases and common-input replays",
            "Audit, report, commit and push the completed diagnostic",
        ]
        if reconstructed
        else [
            "Restore the original 93f24cc checkout and selected model/assignment artifacts",
            "Verify the original parent provenance locks",
            "Implement the rollout/replay adapter against the restored 10H-P interfaces",
            "Reproduce overlapping parent cases before executing locked diagnostics",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.repo_root / "code/config/experiment_10hq.json"
    config = json.loads(config_path.read_text())
    status = parent_preflight(args.repo_root, config)
    status["preparation_sha256"] = {
        path: hashlib.sha256((args.repo_root / path).read_bytes()).hexdigest()
        for path in (
            "code/config/experiment_10hq.json",
            "docs/experiment_10hq_protocol.md",
            "code/src/federated_lpv/feedback_diagnostics.py",
            "code/experiments/experiment_10hq_preflight.py",
            "code/tests/test_feedback_diagnostics.py",
        )
    }
    rendered = json.dumps(status, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return (
        0
        if status["parent_revision_resolves"]
        and not (
            status["missing_parent_paths"]
            or status["missing_parent_artifact_patterns"]
            or status["provenance_errors"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
