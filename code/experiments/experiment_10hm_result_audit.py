"""Audit the reconstructed frozen libraries without running a new fit."""

import json

import numpy as np
import pandas as pd
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hl_validated_model_selection import job_prefix
from experiment_10hm_frozen_libraries import choose_library, runtime
from numpy.testing import assert_allclose

from federated_lpv.causal_model_selection import forecast_choices, prefix_scores
from federated_lpv.physical_coupled_fit import COUPLING_ROW


def run():
    manifest = json.loads((OUT / "experiment_10hm_complete.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in manifest["input_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in manifest["output_sha256"].items():
        assert sha(OUT / name) == expected
    libraries = json.loads((OUT / "experiment_10hm_libraries.json").read_text())
    for lib in libraries:
        seed, scale, fold = lib["seed"], lib["process_noise_scale"], lib["donor_fold"]
        _cfg, _, data, _, _, recipients, _, coords, dt, q, r = runtime(seed, scale)
        z = np.asarray(lib["log_parameters"])
        assert_allclose(z @ COUPLING_ROW, 0, atol=1e-12)
        assert (z >= coords.lower - 1e-8).all() and (z <= coords.upper + 1e-8).all()
        assert_allclose(z, [coords.effective(eta) for eta in lib["eta"]], atol=1e-12)
        a, b, c = (
            set(lib[k]) for k in ["donor_clients", "validation_clients", "recipient_clients"]
        )
        assert len(a) == len(b) == len(c) == 10 and not (a & b or a & c or b & c)
        inner = pd.read_csv(OUT / f"{job_prefix(seed, scale)}_inner_scores.csv")
        # The saved choice must minimize validation scores among eligible models.
        fits = pd.read_csv(OUT / f"{job_prefix(seed, scale)}_fold{fold}_fits.csv")
        eligible = {
            int(row.restart): bool(
                row.success and row.kkt_residual <= 1e-4 and row.constraint_violation <= 1e-8
            )
            for _, row in fits[fits.model == "K2"].iterrows()
        }
        assert choose_library(inner[inner.fold == fold], eligible) == lib["selection"]
        clients, scores = prefix_scores(
            data["heldout"], recipients, z, dt, q, r, 100, [5, 20, 50], 20
        )
        assert clients.tolist() == lib["recipient_clients"]
        assert_allclose(scores, lib["recipient_prefix_scores"], rtol=1e-12)
        choices = (
            forecast_choices(scores, lib["selection"]["margin"])
            if lib["selection"]
            else np.zeros(10, int)
        )
        assert choices.tolist() == lib["recipient_choices"]
    (OUT / "experiment_10hm_audit.json").write_text(
        json.dumps(
            {
                "reconstruction": True,
                "libraries_audited": len(libraries),
                "all_checks_pass": True,
                "library_sha256": sha(OUT / "experiment_10hm_libraries.json"),
                "audit_source_sha256": sha(
                    ROOT / "code/experiments/experiment_10hm_result_audit.py"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "10H-M audit: 20 libraries, bounds/coupling, disjoint clients and causal assignments pass"
    )


if __name__ == "__main__":
    run()
