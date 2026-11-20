import numpy as np
import pytest

pytest.importorskip('skfem')
from thermal_opd import fem


def test_fem_robin_signs_and_front_load_reproduce_linear_axial_solution():
    # Front sign, rear sign, or a missing boundary mass causes this to fail.
    a, k, h0, hL, L = 3., 1.4, 2., 4., 0.5
    b = -hL*a/(k+hL*L)
    solution = fem.solve_steady_fem(1., L, k, h0, hL, 0.,
        lambda x, y: np.zeros_like(x) + h0*a-k*b,
        disk_refinements=2, axial_layers=3, quadrature_order=4, solver='direct', rtol=1e-10)
    x, y, z = np.array([0.13, -0.21]), np.array([0.17, 0.22]), np.array([0.0, L])
    np.testing.assert_allclose(solution.temperature_at(x, y, z), a+b*z, rtol=1e-12)
    assert solution.diagnostics['relative_linear_residual'] < 1e-12
    assert solution.diagnostics['relative_energy_imbalance'] < 1e-12
    assert solution.diagnostics['front_loss_W'] > 0
    assert solution.diagnostics['rear_loss_W'] > 0


@pytest.mark.parametrize('angle', [0., np.arctan(0.25), -np.arctan(0.25)])
def test_segmented_fem_ray_integral_of_bilinear_field_is_exact(angle):
    # A slanted path makes P1(xy)*P1(z) quadratic along a ray. A midpoint
    # integral or fixed-coordinate path misses its z^2 contribution.
    solution = fem.solve_steady_fem(1., 0.5, 1.4, 2., 4., 0., lambda x, y: 1.+0*x,
        disk_refinements=2, axial_layers=3, quadrature_order=4, solver='direct', rtol=1e-10)
    xy = solution.disk_mesh.p
    z = solution.z_m
    solution.temperature_K[:] = 1 + 2*xy[0][None, :] + 3*xy[1][None, :] + 4*z[:, None] + 5*xy[1][None, :]*z[:, None]
    x0, y0, L = np.array([0.13, -0.21]), np.array([0.17, -0.22]), 0.5
    slope = np.tan(angle)
    expected = ((1+2*x0+3*y0)*L + (3*slope+4+5*y0)*L**2/2 + 5*slope*L**3/3)/np.cos(angle)
    actual = solution.refracted_opd(x0, y0, angle, dn_dT=1., point_chunk_size=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fem_does_not_extrapolate_outside_polygon_or_accept_side_exit():
    solution = fem.solve_steady_fem(1., 0.5, 1., 1., 1., 1., lambda x, y: 1.+0*x,
        disk_refinements=1, axial_layers=2, quadrature_order=4, solver='direct', rtol=1e-10)
    with pytest.raises(ValueError):
        solution.temperature_at(1.1, 0., 0.2)
    with pytest.raises(ValueError):
        solution.temperature_at(0., 0., -0.1)
    assert not solution.transmitted_mask(0., 0.9, np.arctan(0.4))
    with pytest.raises(ValueError):
        solution.refracted_opd(0., 0.9, np.arctan(0.4), 1., point_chunk_size=3)


def test_fem_nonaxisymmetric_mode_error_decreases_with_mesh_refinement():
    from scipy.special import jv
    from thermal_opd.eigenmodes import radial_eigenvalues
    from thermal_opd.steady import axial_response
    R, L, k, h = 1., 0.4, 1.3, 2.
    lam = radial_eigenvalues(2, R, k, h, 1)[0]
    def flux(x, y):
        return jv(2, lam*np.hypot(x, y))*np.cos(2*np.arctan2(y, x))
    x, y = np.meshgrid(np.linspace(-0.6, 0.6, 9), np.linspace(-0.6, 0.6, 9))
    expected = flux(x, y)*axial_response(0.1, lam, 1., k, L, h, h)[0]
    errors = []
    for level, layers in ((2, 4), (3, 8)):
        sol = fem.solve_steady_fem(R, L, k, h, h, h, flux, disk_refinements=level,
                                 axial_layers=layers, quadrature_order=6, solver='cg', rtol=1e-11)
        errors.append(np.linalg.norm(sol.temperature_at(x, y, 0.1)-expected)/np.linalg.norm(expected))
        assert sol.diagnostics['relative_linear_residual'] < 1e-9
    assert errors[1] < 0.4*errors[0]
    assert errors[1] < 0.05
