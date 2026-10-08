"""Recompute selected N refinement gradients, locks and causal assignments."""

import json

import numpy as np
import pandas as pd
from experiment_10hi_physically_coupled_fit import subset
from experiment_10hk_joint_client_mixture import OUT, ROOT, sha
from experiment_10hl_validated_model_selection import job_prefix
from experiment_10hm_frozen_libraries import runtime
from experiment_10hn_multihorizon_libraries import select_bundle, source_hashes
from numpy.testing import assert_allclose

from federated_lpv.causal_model_selection import forecast_choices, prefix_scores
from federated_lpv.multihorizon_fit import MultihorizonLoss
from federated_lpv.physical_coupled_fit import COUPLING_ROW, constraint_audit


def run():
    manifest = json.loads((OUT / "experiment_10hn_complete.json").read_text())
    assert manifest["source_sha256"] == source_hashes()
    for name, expected in manifest["output_sha256"].items():
        assert sha(OUT / name) == expected
    for name, expected in manifest["job_manifests"].items():
        assert sha(OUT / f"{name}_complete.json") == expected
        job = json.loads((OUT / f"{name}_complete.json").read_text())
        assert job["source_sha256"] == source_hashes()
        for path, value in job["output_sha256"].items():
            assert sha(OUT / path) == value
    old = {
        (x["seed"], x["process_noise_scale"], x["donor_fold"]): x
        for x in json.loads((OUT / "experiment_10hm_libraries.json").read_text())
    }
    libraries = json.loads((OUT / "experiment_10hn_libraries.json").read_text())
    for lib in libraries:
        seed, scale, fold = lib["seed"], lib["process_noise_scale"], lib["donor_fold"]
        original = old[seed, scale, fold]
        cfg, _, data, _, pos, recipients, folds, coords, dt, q, r = runtime(seed, scale)
        mask = folds != fold
        memberships = pd.read_csv(OUT / f"{job_prefix(seed, scale)}_fold{fold}_memberships.csv")
        memberships = memberships[
            memberships.restart == original["selection"]["restart"]
        ].set_index("client_position")
        refined = []
        for j, audit in enumerate(lib["refinement_audits"]):
            weights = None if j == 0 else memberships.loc[pos[mask], f"weight_{j - 1}"].to_numpy()
            loss = MultihorizonLoss(
                subset(data["train"], mask), coords, dt, q, r, [1, 5, 20, 50], 20, weights
            )
            value, gradient = loss.value_gradient(np.asarray(audit["proposed_eta"]))
            assert_allclose(value, audit["final_loss"], rtol=1e-10)
            independently = constraint_audit(
                np.asarray(audit["proposed_eta"]), gradient / audit["initial_loss"], coords
            )
            assert_allclose(independently["kkt_residual"], audit["kkt_residual"], atol=1e-10)
            eligible = bool(
                audit["success"]
                and independently["kkt_residual"] <= 1e-4
                and independently["constraint_violation"] <= 1e-8
                and value <= audit["initial_loss"] * (1 + 1e-10)
            )
            assert eligible == audit["eligible"]
            refined.append(
                coords.effective(audit["proposed_eta"] if eligible else original["eta"][j])
            )
        selected, candidates = select_bundle(
            subset(data["train"], ~mask),
            pos[~mask],
            {"original": np.asarray(original["log_parameters"]), "refined": np.asarray(refined)},
            cfg,
            dt,
            q,
            r,
        )
        assert selected == lib["selection"] and candidates == lib["validation_candidates"]
        z = np.asarray(lib["log_parameters"])
        assert_allclose(z @ COUPLING_ROW, 0, atol=1e-12)
        assert (z >= coords.lower - 1e-8).all() and (z <= coords.upper + 1e-8).all()
        clients, scores = prefix_scores(
            data["heldout"], recipients, z, dt, q, r, 100, [5, 20, 50], 20
        )
        assert clients.tolist() == lib["recipient_clients"]
        assert_allclose(scores, lib["recipient_prefix_scores"], rtol=1e-12)
        assert forecast_choices(scores, selected["margin"]).tolist() == lib["recipient_choices"]
    (OUT / "experiment_10hn_audit.json").write_text(
        json.dumps(
            {
                "libraries_audited": len(libraries),
                "refinements_audited": 3 * len(libraries),
                "all_checks_pass": True,
                "reconstruction": True,
                "library_sha256": sha(OUT / "experiment_10hn_libraries.json"),
                "audit_source_sha256": sha(
                    ROOT / "code/experiments/experiment_10hn_result_audit.py"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "10H-N audit passed: 20 libraries, 60 gradients/KKT, validation-only selection and prefix assignment"
    )


if __name__ == "__main__":
    run()
