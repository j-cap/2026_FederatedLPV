"""Recipient-only scheduling regression with a function-curvature penalty.

This interface accepts designs and targets, never fleet labels or reference
Jacobians. The penalty acts on the fitted function, not its QR coefficients.
"""
from __future__ import annotations

import numpy as np


def curvature_factor(points=401):
    """Factor for (1/2) integral_{-1}^1 ||f''(z)||^2 dz in the fixed L=7 basis."""
    if points < 3:
        raise ValueError("At least three quadrature points required")
    v = np.linspace(10., 30., points)
    z = (v - 20.) / 10.
    # The representation's QR reference grid is always 401 points.
    vr = np.linspace(10., 30., 401)
    zr = (vr - 20.) / 10.
    reference = np.column_stack([np.ones_like(vr), 1/vr, 1/vr**2,
                                 zr, zr**2, zr**3, zr**4])
    _, transform = np.linalg.qr(reference)
    second = np.column_stack([np.zeros_like(v), 200/v**3, 600/v**4,
                               np.zeros_like(v), 2*np.ones_like(v), 6*z, 12*z**2])
    weights = np.ones(points) / (points - 1)
    weights[[0, -1]] *= .5
    return np.sqrt(weights)[:, None] * (second @ np.linalg.inv(transform))


def fit_smooth_local(design, targets, factor, strength, epsilon=1e-16):
    """Minimize mean target loss + strength*curvature + numerical ridge.

    Two distinct speed observations constrain the curvature penalty's affine
    nullspace. Augmented least squares avoids squaring the condition number.
    """
    design, targets, factor = map(np.asarray, (design, targets, factor))
    if design.ndim != 2 or targets.ndim != 2 or len(design) != len(targets):
        raise ValueError("Design and targets must have matching matrix rows")
    if len(design) < 2 or factor.ndim != 2 or factor.shape[1] != design.shape[1]:
        raise ValueError("Invalid local design or curvature factor")
    if strength <= 0 or epsilon <= 0:
        raise ValueError("Positive curvature and numerical penalties required")
    if not all(np.isfinite(a).all() for a in (design, targets, factor)):
        raise ValueError("Nonfinite local data")
    p = design.shape[1]
    augmented = np.vstack([design/np.sqrt(len(design)), np.sqrt(strength)*factor,
                            np.sqrt(epsilon)*np.eye(p)])
    right = np.vstack([targets/np.sqrt(len(design)),
                        np.zeros((len(factor)+p, targets.shape[1]))])
    return np.linalg.lstsq(augmented, right, rcond=None)[0]


def select_one_se(fleet_scores):
    """Select largest strength within one fleet SE of the best development mean."""
    strengths = sorted(fleet_scores)
    arrays = [np.asarray(fleet_scores[a], dtype=float) for a in strengths]
    if not arrays or min(map(len, arrays)) < 2 or len({len(a) for a in arrays}) != 1:
        raise ValueError("Matched development fleets required")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite development score")
    means = np.array([a.mean() for a in arrays])
    ses = np.array([a.std(ddof=1)/np.sqrt(len(a)) for a in arrays])
    best = int(np.argmin(means))
    threshold = means[best] + ses[best]
    selected = max(a for a, m in zip(strengths, means) if m <= threshold)
    return selected, {"best_strength": strengths[best], "best_mean": float(means[best]),
                      "best_standard_error": float(ses[best]), "threshold": float(threshold)}
