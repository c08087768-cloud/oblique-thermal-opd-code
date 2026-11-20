"""Checks of boundary signs, surface areas and angular cancellation."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.steady import steady_temperature


RADIUS = 12.7e-3
K = 1.38
LENGTH = 5e-3
H_FRONT, H_BACK, H_SIDE = 17.0, 7.0, 12.0


@pytest.fixture(scope="module")
def modal_case():
    roots = {m: radial_eigenvalues(m, RADIUS, K, H_SIDE, 2) for m in (0, 2)}
    coefficients = {0: np.array([800.0, 120.0]), 2: np.array([250.0, -80.0])}
    return coefficients, roots


def test_boundary_check_handles_mixed_modes_and_unequal_face_losses(modal_case):
    from thermal_opd.steady_validation import steady_boundary_residuals

    coefficients, roots = modal_case
    residuals = steady_boundary_residuals(
        np.linspace(0, RADIUS, 41), np.linspace(0, 2*np.pi, 32, endpoint=False),
        np.linspace(0, LENGTH, 25), coefficients, roots,
        RADIUS, K, LENGTH, H_FRONT, H_BACK, H_SIDE,
    )
    assert max(residuals.values()) < 1e-10


def test_boundary_check_detects_inconsistent_side_heat_transfer(modal_case):
    from thermal_opd.steady_validation import steady_boundary_residuals

    coefficients, roots = modal_case
    r = np.linspace(0, RADIUS, 41)
    phi = np.linspace(0, 2*np.pi, 32, endpoint=False)
    z = np.linspace(0, LENGTH, 25)
    delta_h = 3.0
    residuals = steady_boundary_residuals(
        r, phi, z, coefficients, roots, RADIUS, K, LENGTH,
        H_FRONT, H_BACK, H_SIDE + delta_h,
    )
    side_temperature = steady_temperature(
        RADIUS, phi[None, :], z[:, None], coefficients, roots,
        K, LENGTH, H_FRONT, H_BACK,
    )
    # The roots satisfy the original H_SIDE. Changing only h leaves delta_h*T.
    assert np.isclose(residuals["side_W_per_m2"], delta_h * np.max(np.abs(side_temperature)), rtol=1e-11)
    assert residuals["side_W_per_m2"] > 1.0


def test_energy_balance_matches_independent_surface_integrals(modal_case):
    from thermal_opd.steady_validation import steady_energy_balance

    coefficients, roots = modal_case
    actual = steady_energy_balance(
        coefficients, roots, RADIUS, K, LENGTH, H_FRONT, H_BACK, H_SIDE,
        num_r=32, num_z=24, num_phi=32,
    )
    # Exact angular integration removes m>0. Use adaptive scalar quadrature,
    # independently of the implementation's fixed Gauss surface quadrature.
    axisymmetric = {0: coefficients[0]}
    axisymmetric_roots = {0: roots[0]}
    args = (axisymmetric, axisymmetric_roots, K, LENGTH, H_FRONT, H_BACK)
    front = quad(lambda r: 2*np.pi*r*H_FRONT*float(steady_temperature(r, 0, 0, *args)), 0, RADIUS)[0]
    rear = quad(lambda r: 2*np.pi*r*H_BACK*float(steady_temperature(r, 0, LENGTH, *args)), 0, RADIUS)[0]
    side = quad(lambda z: 2*np.pi*RADIUS*H_SIDE*float(steady_temperature(RADIUS, 0, z, *args)), 0, LENGTH)[0]
    input_power = quad(lambda r: 2*np.pi*r*sum(c*jv(0, lam*r) for c, lam in zip(coefficients[0], roots[0])), 0, RADIUS)[0]
    for key, expected in (("front_loss_W", front), ("rear_loss_W", rear), ("side_loss_W", side), ("modal_input_W", input_power)):
        assert np.isclose(actual[key], expected, rtol=1e-10, atol=1e-13)
    assert actual["balance_absolute_error_W"] < 1e-12


def test_nonaxisymmetric_mode_has_zero_net_power(modal_case):
    from thermal_opd.steady_validation import steady_energy_balance

    coefficients, roots = modal_case
    actual = steady_energy_balance(
        {2: coefficients[2]}, {2: roots[2]}, RADIUS, K, LENGTH,
        H_FRONT, H_BACK, H_SIDE, num_r=32, num_z=24, num_phi=32,
    )
    assert actual["modal_input_W"] == 0.0
    assert max(abs(value) for value in actual.values()) < 1e-12


def test_zero_source_reports_finite_absolute_energy_balance(modal_case):
    from thermal_opd.steady_validation import steady_energy_balance

    _, roots = modal_case
    actual = steady_energy_balance(
        {0: np.zeros(2)}, {0: roots[0]}, RADIUS, K, LENGTH,
        H_FRONT, H_BACK, H_SIDE, num_r=16, num_z=16, num_phi=16,
    )
    assert all(value == 0.0 for value in actual.values())
