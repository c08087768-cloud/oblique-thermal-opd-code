import numpy as np
import pytest

from thermal_opd.steady import axial_coefficients, axial_temperature, steady_mode, steady_temperature
from thermal_opd.steady import axial_response, steady_temperature_gradient


CONDUCTIVITY = 1.38
THICKNESS = 6.0e-3
H_FRONT = 10.0
H_BACK = 10.0


def test_front_boundary_condition():
    lam = 300.0
    source_coefficient = 1000.0

    A, B = axial_coefficients(lam, source_coefficient, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)

    Z0 = axial_temperature(0.0, lam, A, B)
    dZ0 = lam * B

    residual = -CONDUCTIVITY * dZ0 + H_FRONT * Z0 - source_coefficient

    assert np.isclose(residual, 0.0, atol=1e-10)


def test_back_boundary_condition():
    lam = 300.0
    source_coefficient = 1000.0

    A, B = axial_coefficients(lam, source_coefficient, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)

    ZL = axial_temperature(THICKNESS, lam, A, B)
    dZL = lam * (A * np.sinh(lam * THICKNESS) + B * np.cosh(lam * THICKNESS))

    residual = CONDUCTIVITY * dZL + H_BACK * ZL

    assert np.isclose(residual, 0.0, atol=1e-10)


def test_axial_ode():
    lam = 300.0
    source_coefficient = 1000.0

    A, B = axial_coefficients(lam, source_coefficient, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)

    z = THICKNESS / 2
    Z = axial_temperature(z, lam, A, B)
    step = 1e-5
    ddZ = (
        axial_temperature(z + step, lam, A, B)
        - 2 * Z
        + axial_temperature(z - step, lam, A, B)
    ) / step**2

    assert np.isclose(ddZ, lam**2 * Z, rtol=1e-6, atol=0.0)


def test_single_mode_temperature():
    r = np.array([2.0e-3])
    phi = np.array([0.4])
    z = np.array([3.0e-3])

    m = 2
    lam = 300.0
    source_coefficient = 1000.0

    coefficients = {m: np.array([source_coefficient])}
    eigenvalues = {m: np.array([lam])}

    temperature = steady_temperature(r, phi, z, coefficients, eigenvalues, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)

    A, B = axial_coefficients(lam, source_coefficient, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)
    expected = steady_mode(r, phi, z, m, lam, A, B)

    assert np.allclose(temperature, expected)


def test_temperature_scales_linearly_with_source():
    r = np.array([2.0e-3])
    phi = np.array([0.4])
    z = np.array([3.0e-3])

    eigenvalues = {0: np.array([300.0])}
    coefficients_1 = {0: np.array([1000.0])}
    coefficients_2 = {0: np.array([2000.0])}

    T1 = steady_temperature(r, phi, z, coefficients_1, eigenvalues, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)
    T2 = steady_temperature(r, phi, z, coefficients_2, eigenvalues, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)

    assert np.allclose(T2, 2.0 * T1)


def test_high_radial_mode_retains_small_positive_back_temperature():
    # With an insulated rear face, the exact rear amplitude is known.
    lam = 6000.0
    source = 1000.0
    expected = source / (
        CONDUCTIVITY * lam * np.sinh(lam * THICKNESS)
        + H_FRONT * np.cosh(lam * THICKNESS)
    )
    actual = steady_temperature(
        0.0, 0.0, THICKNESS,
        {0: np.array([source])}, {0: np.array([lam])},
        CONDUCTIVITY, THICKNESS, H_FRONT, 0.0,
    )

    assert np.isclose(actual, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("lam", [40.0, 300.0, 6000.0])
def test_stable_axial_response_satisfies_unequal_face_boundaries(lam):
    source = 1000.0
    front_h, back_h = 15.0, 5.0
    front, front_gradient = axial_response(
        0.0, lam, source, CONDUCTIVITY, THICKNESS, front_h, back_h
    )
    back, back_gradient = axial_response(
        THICKNESS, lam, source, CONDUCTIVITY, THICKNESS, front_h, back_h
    )

    assert np.isclose(front_h * front - CONDUCTIVITY * front_gradient, source)
    assert np.isclose(back_h * back + CONDUCTIVITY * back_gradient, 0.0, atol=1e-10)


def test_temperature_gradients_match_independent_finite_differences():
    r, phi, z = 2e-3, 0.4, 3e-3
    source = {0: np.array([1000.0]), 2: np.array([200.0])}
    roots = {0: np.array([80.0]), 2: np.array([300.0])}
    args = (source, roots, CONDUCTIVITY, THICKNESS, H_FRONT, H_BACK)
    dr, dz = steady_temperature_gradient(r, phi, z, *args)
    step = 1e-7
    dr_fd = (steady_temperature(r + step, phi, z, *args)
             - steady_temperature(r - step, phi, z, *args)) / (2 * step)
    dz_fd = (steady_temperature(r, phi, z + step, *args)
             - steady_temperature(r, phi, z - step, *args)) / (2 * step)

    assert np.isclose(dr, dr_fd, rtol=1e-7, atol=0.0)
    assert np.isclose(dz, dz_fd, rtol=1e-7, atol=0.0)
