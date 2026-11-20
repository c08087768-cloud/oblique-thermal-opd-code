"""Independent checks of Snell geometry and refracted-ray temperature integrals."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

import thermal_opd.opd as opd_module
import thermal_opd.steady as steady_module
from thermal_opd.model import ModalResolution, ThermalModel
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.steady import steady_temperature


def test_snell_angle_for_known_pair_of_sines():
    from thermal_opd.opd import refracted_angle
    assert np.isclose(refracted_angle(np.arcsin(0.75), 1., 1.5), np.pi/6, rtol=1e-13)
    assert np.isclose(refracted_angle(-np.arcsin(0.75), 1., 1.5), -np.pi/6)


@pytest.mark.parametrize("angle,n_in,n_sub", [(np.pi/2, 1., 1.5), (0.2, 0., 1.5), (0.2, 1., np.nan), (np.pi/3, 1.5, 1.)])
def test_snell_rejects_nonphysical_or_nontransmitting_inputs(angle, n_in, n_sub):
    from thermal_opd.opd import refracted_angle
    with pytest.raises(ValueError):
        refracted_angle(angle, n_in, n_sub)


def test_transmission_mask_requires_entry_and_exit_inside_cylinder():
    from thermal_opd.opd import transmitted_ray_mask
    x = np.array([0., 0., 0., 0.011])
    y = np.array([0., 0.009, -0.009, 0.])
    actual = transmitted_ray_mask(x, y, 0.01, 0.003, np.pi/4)
    np.testing.assert_array_equal(actual, [True, False, True, False])
    assert transmitted_ray_mask(0., 0.007, 0.01, 0.003, np.pi/4)


@pytest.mark.parametrize("angle_deg", [0., 30., 60.])
def test_optical_inputs_change_only_ray_geometry_and_opd_scale(angle_deg):
    current = PhysicalParameters()
    old_index = replace(current, substrate_refractive_index=1.44963)
    old_slope = replace(current, dn_dT=9.6e-6)
    prepared = ThermalModel(current, angle_deg, ModalResolution(2, 3, 80, 16, 32)).prepare()
    temperature = prepared.solve_steady().temperature

    def ray(index):
        sine = np.sin(np.deg2rad(angle_deg)) / index
        cosine = np.sqrt(1 - sine**2)
        return sine, cosine, sine / cosine

    _, _, current_slope = ray(current.substrate_refractive_index)
    _, _, previous_slope = ray(old_index.substrate_refractive_index)
    edge_y = current.radius - current.thickness * (current_slope + previous_slope) / 2
    test_y = [0., edge_y, current.radius * 1.01]
    np.testing.assert_array_equal(
        opd_module.transmitted_ray_mask(0., test_y, current.radius, current.thickness,
                                        prepared.refraction_angle),
        [True, True, False],
    )
    previous = replace(prepared, parameters=old_index)
    np.testing.assert_array_equal(
        opd_module.transmitted_ray_mask(0., test_y, current.radius, current.thickness,
                                        previous.refraction_angle),
        [True, angle_deg == 0., False],
    )

    x = np.array([.001, -.002])
    y = np.array([-.002, .003])
    values = []
    for parameters in (current, old_index, old_slope):
        model = replace(prepared, parameters=parameters)
        steady = model.solve_steady()
        sine, cosine, slope = ray(parameters.substrate_refractive_index)
        assert model.refraction_angle == pytest.approx(np.arctan2(sine, cosine), abs=1e-15)
        np.testing.assert_array_equal(steady.temperature(.002, .3, .001),
                                      temperature(.002, .3, .001))

        def integrate(x0, y0):
            def integrand(z):
                ray_y = y0 + z * slope
                return float(temperature(np.hypot(x0, ray_y), np.arctan2(ray_y, x0), z))

            return parameters.dn_dT / cosine * quad(integrand, 0., parameters.thickness)[0]

        result = steady.opd(x, y)
        np.testing.assert_allclose(result, [integrate(x0, y0) for x0, y0 in zip(x, y)],
                                   rtol=1e-11, atol=1e-20)
        values.append(result)

    new_opd, old_index_opd, old_slope_opd = values
    np.testing.assert_allclose(new_opd, old_slope_opd * current.dn_dT / old_slope.dn_dT,
                               rtol=1e-13, atol=1e-20)
    if angle_deg == 0.:
        np.testing.assert_array_equal(new_opd, old_index_opd)
    else:
        assert not np.allclose(new_opd, old_index_opd, rtol=1e-8, atol=1e-20)


def test_nearly_uniform_temperature_uses_actual_ray_length():
    from thermal_opd.opd import steady_refracted_opd
    source, h_front, length, dn = 1000., 15., 0.005, 1e-5
    theta = 0.4
    actual = steady_refracted_opd(
        0., 0., {0: np.array([source])}, {0: np.array([1e-8])},
        1.38, length, h_front, 0., dn, radius=0.0127,
        refraction_angle=theta, num_ray_points=32, point_chunk_size=10,
    )
    assert np.isclose(actual, source/h_front*dn*length/np.cos(theta), rtol=1e-12, atol=0.)


@pytest.mark.parametrize("theta", [0.4, -0.4])
def test_refracted_opd_matches_independent_moving_coordinate_integral(theta):
    from thermal_opd.opd import steady_refracted_opd
    x = np.array([-0.002, 0.001])
    y = np.array([-0.002, 0.003])
    c = {0: np.array([1000.]), 2: np.array([200.]), 4: np.array([-50.])}
    roots = {0: np.array([80.]), 2: np.array([300.]), 4: np.array([600.])}
    args = (c, roots, 1.38, 0.005, 15., 5.)
    dn = 0.96e-5
    actual = steady_refracted_opd(
        x[:, None], y[None, :], *args, dn, radius=0.0127,
        refraction_angle=theta, num_ray_points=48, point_chunk_size=2,
    )
    expected = np.empty((2, 2))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            def integrand(z):
                ray_y = yj + z*np.tan(theta)
                return float(steady_temperature(np.hypot(xi, ray_y), np.arctan2(ray_y, xi), z, *args))
            expected[i, j] = dn/np.cos(theta)*quad(integrand, 0., 0.005, epsabs=1e-13)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-20)
    wrong = dn/np.cos(theta)*quad(
        lambda z: float(steady_temperature(np.hypot(x[1], y[1]), np.arctan2(y[1], x[1]), z, *args)), 0., 0.005
    )[0]
    assert abs(actual[1, 1] - wrong)/abs(actual[1, 1]) > 1e-3


def test_zero_incidence_high_mode_obeys_integrated_energy_identity():
    from thermal_opd.opd import steady_refracted_opd
    lam, source, k, dn = 150000., 1000., 1.38, 1e-5
    actual = steady_refracted_opd(
        0., 0., {0: np.array([source])}, {0: np.array([lam])},
        k, 0.005, 0., 0., dn, radius=0.0127,
        refraction_angle=0., num_ray_points=256, point_chunk_size=1,
    )
    assert np.isclose(actual, source/(k*lam**2)*dn, rtol=1e-10, atol=0.)


def test_refracted_opd_preserves_sign_and_chunking():
    from thermal_opd.opd import steady_refracted_opd
    x = np.array([0., 0.001, 0.002])
    args = ({0: np.array([1000.])}, {0: np.array([40.])}, 1.38, 0.005, 15., 5.)
    opts = dict(radius=0.0127, refraction_angle=0.3, num_ray_points=32)
    first = steady_refracted_opd(x, 0., *args, 1e-5, **opts, point_chunk_size=1)
    second = steady_refracted_opd(x, 0., *args, -1e-5, **opts, point_chunk_size=3)
    np.testing.assert_allclose(first, -second, rtol=1e-13)


def test_opd_rejects_side_exiting_rays_instead_of_extrapolating_temperature():
    from thermal_opd.opd import steady_refracted_opd
    with pytest.raises(ValueError, match="rear"):
        steady_refracted_opd(
            0., 0.012, {0: np.array([1000.])}, {0: np.array([40.])},
            1.38, 0.005, 15., 5., 1e-5, radius=0.0127,
            refraction_angle=0.4, num_ray_points=32, point_chunk_size=1,
        )


def test_removed_opd_api_and_axial_only_integral_are_absent():
    assert not hasattr(opd_module, "steady_opd")
    assert not hasattr(steady_module, "axial_integral")
