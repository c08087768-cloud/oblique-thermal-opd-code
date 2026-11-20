import numpy as np

from thermal_opd.heat_source import gaussian_disk_power, oblique_gaussian_heat_flux
from thermal_opd.parameters import PhysicalParameters


def test_total_absorbed_power():
    w0 = 6e-3
    theta = np.deg2rad(30.0)
    absorbed_power = 1.0

    x = np.linspace(-5 * w0, 5 * w0, 1001)
    y = np.linspace(-5 * w0, 5 * w0, 1001)
    X, Y = np.meshgrid(x, y)

    q = oblique_gaussian_heat_flux(X, Y, w0, theta, absorbed_power)

    total_power = np.trapezoid(np.trapezoid(q, x=x, axis=1), x=y)

    assert np.isclose(total_power, absorbed_power, rtol=1e-4)

def test_normal_incidence_is_axisymmetric():
    w0 = 6e-3
    absorbed_power = 1.0

    r = np.linspace(0.0, 2 * w0, 100)

    q_x = oblique_gaussian_heat_flux(r, 0.0, w0, 0.0, absorbed_power)
    q_y = oblique_gaussian_heat_flux(0.0, r, w0, 0.0, absorbed_power)

    assert np.allclose(q_x, q_y)

def test_oblique_incidence_stretches_y_direction():
    w0 = 6e-3
    theta = np.deg2rad(30.0)
    absorbed_power = 1.0

    q_x = oblique_gaussian_heat_flux(w0, 0.0, w0, theta, absorbed_power)
    q_y = oblique_gaussian_heat_flux(0.0, w0, w0, theta, absorbed_power)

    assert q_y > q_x


def test_projected_one_over_e_squared_radii():
    p = PhysicalParameters()
    theta = np.deg2rad(60.0)
    q0 = oblique_gaussian_heat_flux(
        0.0, 0.0, p.beam_radius, theta, p.absorbed_power
    )
    qx = oblique_gaussian_heat_flux(
        p.beam_radius, 0.0, p.beam_radius, theta, p.absorbed_power
    )
    qy = oblique_gaussian_heat_flux(
        0.0, p.beam_radius / np.cos(theta),
        p.beam_radius, theta, p.absorbed_power
    )

    assert np.isclose(qx / q0, np.exp(-2))
    assert np.isclose(qy / q0, np.exp(-2))


def test_normal_incidence_power_on_finite_disk():
    p = PhysicalParameters()
    radius = p.beam_radius
    r = np.linspace(0.0, radius, 2001)
    q = oblique_gaussian_heat_flux(
        r, 0.0, p.beam_radius, 0.0, p.absorbed_power
    )
    actual = 2 * np.pi * np.trapezoid(q * r, x=r)
    expected = p.absorbed_power * (
        1 - np.exp(-2 * radius**2 / p.beam_radius**2)
    )

    assert np.isclose(actual, expected, rtol=1e-6)


def test_gaussian_disk_quadrature_recovers_exact_normal_incidence_power():
    p = PhysicalParameters()
    expected = p.absorbed_power * (-np.expm1(-2 * (p.radius / p.beam_radius)**2))
    nominal = gaussian_disk_power(p.radius, p.beam_radius, 0., p.absorbed_power,
                                  num_r=80, num_phi=180)
    underresolved = gaussian_disk_power(p.radius, p.beam_radius, 0., p.absorbed_power,
                                        num_r=1, num_phi=180)
    assert np.isclose(nominal, expected, rtol=1e-13, atol=0)
    assert not np.isclose(underresolved, expected, rtol=1e-3, atol=0)
