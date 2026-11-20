"""Verify radial mode norms and roots of the Robin eigenvalue equation.

The roots must be positive, ordered, and returned in the requested count.
Substituting each root into the equation checks its residual.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from thermal_opd.eigenmodes import (
    radial_eigenvalue_equation,
    radial_eigenvalues,
    radial_mode_norm,
)

RADIUS = 12.5e-3
CONDUCTIVITY = 1.38
H = 10.0


@pytest.mark.parametrize("m", [0, 2, 8])
@pytest.mark.parametrize("mode_index", [0, 3])
def test_radial_norm_matches_independent_integral(m, mode_index):
    lam = radial_eigenvalues(m, RADIUS, CONDUCTIVITY, H, num_modes=4)[mode_index]
    expected, _ = quad(
        lambda r: jv(m, lam * r)**2 * r,
        0.0, RADIUS, epsabs=1e-14, epsrel=1e-12,
    )

    assert np.isclose(radial_mode_norm(lam, m, RADIUS), expected, rtol=1e-10, atol=0.0)


def test_number_of_eigenvalues():
    roots = radial_eigenvalues(
        m=0,
        radius=RADIUS,
        conductivity=CONDUCTIVITY,
        h=H,
        num_modes=5,
    )

    assert len(roots) == 5


def test_eigenvalues_are_positive_and_ordered():
    roots = radial_eigenvalues(
        m=2,
        radius=RADIUS,
        conductivity=CONDUCTIVITY,
        h=H,
        num_modes=5,
    )

    assert np.all(roots > 0)
    assert np.all(np.diff(roots) > 0)


def test_eigenvalues_satisfy_robin_boundary_condition():
    roots = radial_eigenvalues(
        m=2,
        radius=RADIUS,
        conductivity=CONDUCTIVITY,
        h=H,
        num_modes=5,
    )

    residuals = [
        radial_eigenvalue_equation(
            lam,
            m=2,
            radius=RADIUS,
            conductivity=CONDUCTIVITY,
            h=H,
        )
        for lam in roots
    ]

    assert np.allclose(residuals, 0.0, atol=1e-8)


def test_axisymmetric_insulated_roots_require_zero_mode_support():
    with pytest.raises(ValueError, match='zero mode'):
        radial_eigenvalues(0, RADIUS, CONDUCTIVITY, 0., num_modes=3)
