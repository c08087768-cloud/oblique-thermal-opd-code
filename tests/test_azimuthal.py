import numpy as np

from thermal_opd.azimuthal import azimuthal_coefficients
from thermal_opd.heat_source import oblique_gaussian_heat_flux


def test_known_fourier_coefficients():
    phi = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    q_phi = 3.0 + 2.0 * np.cos(2 * phi)

    a, b = azimuthal_coefficients(q_phi, phi, max_mode=4)

    assert np.isclose(a[0], 3.0)
    assert np.isclose(a[2], 2.0)

    assert np.allclose(a[[1, 3, 4]], 0.0, atol=1e-12)
    assert np.allclose(b, 0.0, atol=1e-12)

def test_oblique_gaussian_symmetry():
    w0 = 6e-3
    theta = np.deg2rad(30.0)
    absorbed_power = 1.0
    r = w0

    phi = np.linspace(0, 2 * np.pi, 256, endpoint=False)

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    q_phi = oblique_gaussian_heat_flux(
        x, y, w0, theta, absorbed_power
    )

    a, b = azimuthal_coefficients(q_phi, phi, max_mode=6)

    assert np.allclose(b, 0.0, atol=1e-10)
    assert np.allclose(a[[1, 3, 5]], 0.0, atol=1e-10)