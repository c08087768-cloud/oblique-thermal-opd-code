import numpy as np
from thermal_opd import projection


def test_batched_projection_matches_independent_scalar_integrals():
    r = np.linspace(0., 0.7, 301)
    phi = np.linspace(0., 2*np.pi, 40, endpoint=False)
    q = np.exp(-r[:, None]**2)*(2.+np.cos(2*phi))
    roots = np.array([1., 3., 8.])
    actual = projection.fourier_bessel_coefficients(q, r, phi, 2, roots)
    expected = [projection.fourier_bessel_coefficient(q, r, phi, 2, lam) for lam in roots]
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
from scipy.special import jv
from scipy.integrate import quad

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.projection import fourier_bessel_coefficient


RADIUS = 12.5e-3
CONDUCTIVITY = 1.38
H = 10.0


def test_coarse_projection_uses_continuous_mode_norm():
    m = 0
    lam = radial_eigenvalues(m, RADIUS, CONDUCTIVITY, H, num_modes=1)[0]
    r = np.array([0.0, RADIUS / 2, RADIUS])
    phi = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
    q = np.zeros((3, len(phi)))
    q[1, :] = 1.0
    norm, _ = quad(lambda radius: jv(m, lam * radius)**2 * radius, 0, RADIUS)
    # For these three radial samples, the trapezoidal numerator is known.
    expected = (RADIUS**2 / 4) * jv(m, lam * RADIUS / 2) / norm

    actual = fourier_bessel_coefficient(q, r, phi, m, lam)

    assert np.isclose(actual, expected, rtol=1e-10, atol=0.0)


def test_single_mode_projection():
    m = 2
    lam = radial_eigenvalues(m, RADIUS, CONDUCTIVITY, H, num_modes=1)[0]

    r = np.linspace(0.0, RADIUS, 400)
    phi = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    R, Phi = np.meshgrid(r, phi, indexing="ij")

    q = 3.5 * jv(m, lam * R) * np.cos(m * Phi)
    coefficient = fourier_bessel_coefficient(q, r, phi, m, lam)

    assert np.isclose(coefficient, 3.5, rtol=1e-3)


def test_orthogonal_radial_mode_projection():
    m = 2
    roots = radial_eigenvalues(m, RADIUS, CONDUCTIVITY, H, num_modes=2)

    r = np.linspace(0.0, RADIUS, 400)
    phi = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    R, Phi = np.meshgrid(r, phi, indexing="ij")

    q = jv(m, roots[0] * R) * np.cos(m * Phi)
    coefficient = fourier_bessel_coefficient(q, r, phi, m, roots[1])

    assert np.isclose(coefficient, 0.0, atol=1e-3)
