"""Execute the locked Q factorial without fitting or changing parent libraries."""
import argparse
import json

from experiment_10hk_joint_client_mixture import ROOT
from experiment_10hp_output_feedback import run
from experiment_10hq_preflight import parent_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads((ROOT/"code/config/experiment_10hq.json").read_text())
    status = parent_preflight(ROOT, cfg)
    if status["status"] != "ready_for_frozen_diagnostic":
        raise RuntimeError(f"Parent provenance blocked: {status}")
    if cfg["fit_or_select_models"] or cfg["confirmation_run"]:
        raise RuntimeError("The frozen development diagnostic cannot fit or confirm")
    run("10hq", args.workers)


if __name__ == "__main__":
    main()
