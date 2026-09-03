"""Phase 0: inspect speed-dependent ground-truth dynamics for all families."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from federated_lpv import continuous_bicycle_matrices, family_centers


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "phase0_vehicle_families.png"


def main() -> None:
    speeds = np.linspace(10.0, 30.0, 101)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    entries = ((0, 0, "A[0,0]"), (0, 1, "A[0,1]"), (0, 0, "B[0,0]"))

    for name, parameters in family_centers().items():
        matrices = [continuous_bicycle_matrices(speed, parameters) for speed in speeds]
        for axis, (row, column, label) in zip(axes, entries, strict=True):
            values = [a[row, column] if label.startswith("A") else b[row, column] for a, b in matrices]
            axis.plot(speeds, values, label=name)
            axis.set(xlabel="speed [m/s]", title=label)
            axis.grid(alpha=0.25)

    axes[0].set_ylabel("matrix entry")
    axes[-1].legend(frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
