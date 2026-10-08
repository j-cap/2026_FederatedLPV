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
        cwd=root, capture_output=True, text=True, check=False,
    )
    missing = [path for path in config["required_parent_paths"] if not (root / path).is_file()]
    missing_patterns = [
        pattern for pattern in config["required_parent_artifact_patterns"]
        if not any(root.glob(pattern))
    ]
    available = resolved.returncode == 0 and not missing and not missing_patterns
    return {
        "experiment": "10H-Q",
        "status": "ready_for_parent_provenance_audit" if available else "blocked_missing_parent",
        "required_parent_revision": config["required_parent_revision"],
        "parent_revision_resolves": resolved.returncode == 0,
        "resolved_parent_revision": resolved.stdout.strip() if resolved.returncode == 0 else None,
        "missing_parent_paths": missing,
        "missing_parent_artifact_patterns": missing_patterns,
        "fleet_rollouts_executed": 0,
        "rollout_adapter_implemented": False,
        "frozen_models_refitted": False,
        "confirmation_run": False,
        "remaining_work": [
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
            "code/config/experiment_10hq.json", "docs/experiment_10hq_protocol.md",
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
    return 0 if status["parent_revision_resolves"] and not (
        status["missing_parent_paths"] or status["missing_parent_artifact_patterns"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
