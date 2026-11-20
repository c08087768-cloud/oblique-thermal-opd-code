"""Physical checks: Robin signs, boundary forcing, convolution and ray geometry."""
import numpy as np
import pytest
from scipy.integrate import quad

from thermal_opd.transient import AxialModes, PowerHistory
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.steady import steady_temperature
from thermal_opd.opd import steady_refracted_opd


def test_axial_modes_satisfy_asymmetric_robin_and_are_orthogonal():
    # Reversing the front Robin sign or omitting a root breaks this check.
    modes = AxialModes(0.4, 1.3, 2., 5., 7)
    assert np.all(np.diff(modes.beta) > 0)
    for j in range(7):
        b = modes.beta[j]
        def f(z):
            return np.cos(b*z) + 2/(1.3*b)*np.sin(b*z)
        def df(z):
            return -b*np.sin(b*z) + 2/1.3*np.cos(b*z)
        assert abs(2*f(0)-1.3*df(0)) < 1e-11
        assert abs(5*f(.4)+1.3*df(.4)) < 1e-9
        assert modes.norm[j] == pytest.approx(quad(lambda z: f(z)**2, 0, .4)[0], rel=1e-11)
    z = np.linspace(0, .4, 20001)
    values = modes.values(z)
    assert abs(np.trapezoid(values[0]*values[1], x=z)) < 1e-9


@pytest.mark.parametrize('kind', ['step', 'linear', 'exponential'])
@pytest.mark.parametrize('rate', [.2, 1., 1.+1e-10, 10.])
def test_time_response_matches_independent_boundary_load_convolution(kind, rate):
    history = PowerHistory(kind, 1.)
    t = 2.3
    expected = rate*quad(lambda s: np.exp(-rate*(t-s))*history.value(s), 0, t, points=[1.])[0]
    actual = history.value(t)-history.deficit(rate, t)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_zero_initial_and_long_time_reuse_steady_temperature_and_refracted_opd(modal_problem):
    p = PhysicalParameters()
    c, roots = {0: np.array([1000.]), 2: np.array([200.])}, {0: np.array([80.]), 2: np.array([300.])}
    prepared=modal_problem(p,c,roots,refraction_angle=.4)
    model = prepared.solve_transient(num_axial_modes=32)
    args = (c, roots, p.conductivity, p.thickness, p.h_eff, p.h_eff)
    r, phi, z = np.array([.002, .003]), np.array([.1, .4]), np.array([0., .005])
    np.testing.assert_array_equal(model.temperature(r, phi, z, 0.), [0., 0.])
    np.testing.assert_allclose(model.temperature(r, phi, z, 1e5), steady_temperature(r, phi, z, *args), rtol=1e-13)
    kw = dict(num_ray_points=48, point_chunk_size=3)
    x, y = np.array([-.002, .001]), np.array([-.002, .003])
    np.testing.assert_array_equal(model.opd(x, y, 0., **kw), [0., 0.])
    np.testing.assert_allclose(model.opd(x, y, 1e5, **kw),
        steady_refracted_opd(x, y, *args, p.dn_dT, radius=p.radius,refraction_angle=.4,**kw), rtol=1e-13)
    expected = []
    for xx, yy in zip(x, y):
        expected.append(p.dn_dT/np.cos(.4)*quad(lambda zz: float(model.temperature(
            np.hypot(xx, yy+zz*np.tan(.4)), np.arctan2(yy+zz*np.tan(.4), xx), zz, .625)),
            0., p.thickness, epsabs=1e-12)[0])
    np.testing.assert_allclose(model.opd(x, y, .625, **kw), expected, rtol=1e-10)


def test_early_front_temperature_matches_semi_infinite_flux_solution(modal_problem):
    # Boundary forcing normalization and rho*cp both enter this independent limit.
    from dataclasses import replace
    p = replace(PhysicalParameters(), h_conv=0., emissivity=0., thickness=.1)
    q, t = 1000., .1
    model = modal_problem(p,{0:np.array([q])},{0:np.array([1.])}).solve_transient(num_axial_modes=1200)
    expected = 2*q/p.conductivity*np.sqrt(p.thermal_diffusivity*t/np.pi)
    assert model.temperature(0., 0., 0., t) == pytest.approx(expected, rel=2e-6)


def test_temperature_orders_and_their_ray_opds_add_without_pv_additivity(modal_problem):
    p = PhysicalParameters()
    c, roots = {0: np.array([1000.]), 2: np.array([200.])}, {0: np.array([80.]), 2: np.array([300.])}
    model = modal_problem(p,c,roots,refraction_angle=.4).solve_transient(num_axial_modes=24)
    kw = dict(num_ray_points=32, point_chunk_size=4)
    x, y = np.array([-.002, .003]), np.array([.001, -.002])
    separate = [modal_problem(p,{m:c[m]},{m:roots[m]},refraction_angle=.4).solve_transient(num_axial_modes=24) for m in c]
    np.testing.assert_allclose(sum(s.temperature(np.hypot(x,y),np.arctan2(y,x),0.,2.5) for s in separate),
                               model.temperature(np.hypot(x,y),np.arctan2(y,x),0.,2.5), rtol=1e-13)
    np.testing.assert_allclose(sum(s.opd(x,y,2.5,**kw) for s in separate),
                               model.opd(x,y,2.5,**kw), rtol=1e-13)


def test_step_boundary_forcing_balances_storage_and_all_three_heat_losses(modal_problem):
    # Missing X(0)/norm in the boundary load or a wrong diffusion rate violates
    # the integrated PDE even if long-time and zero-time shortcuts pass.
    from thermal_opd.eigenmodes import radial_eigenvalues
    from scipy.special import jv
    from numpy.polynomial.legendre import leggauss
    p = PhysicalParameters()
    lam = radial_eigenvalues(0,p.radius,p.conductivity,p.h_eff,1)[0]
    q = 1000.
    model = modal_problem(p,{0:np.array([q])},{0:np.array([lam])}).solve_transient(num_axial_modes=64)
    nodes,w = leggauss(100)
    z = (nodes+1)*p.thickness/2
    wz = w*p.thickness/2
    radial_integral = 2*np.pi*p.radius*jv(1,lam*p.radius)/lam
    t,eps = .625,1e-4
    T = model.temperature(0.,0.,z,t)
    derivative = (model.temperature(0.,0.,z,t+eps)-model.temperature(0.,0.,z,t-eps))/(2*eps)
    storage = p.density*p.specific_heat*radial_integral*np.dot(wz,derivative)
    face_loss = p.h_eff*radial_integral*(model.temperature(0.,0.,0.,t)+model.temperature(0.,0.,p.thickness,t))
    side_loss = p.h_eff*2*np.pi*p.radius*jv(0,lam*p.radius)*np.dot(wz,T)
    assert storage+face_loss+side_loss == pytest.approx(q*radial_integral,rel=1e-8)


def test_axial_truncation_converges_at_very_early_positive_time(modal_problem):
    p = PhysicalParameters()
    c,roots = {0:np.array([1000.])},{0:np.array([80.])}
    prepared=modal_problem(p,c,roots)
    reference = prepared.solve_transient(num_axial_modes=256).temperature(0.,0.,0.,.001)
    values = [prepared.solve_transient(num_axial_modes=j).temperature(0.,0.,0.,.001) for j in (12,48,128)]
    errors = np.abs(np.array(values)-reference)
    assert errors[1]<errors[0]/10
    assert errors[2]<errors[1]/100


@pytest.mark.parametrize('angle',[0.,.4,-.4])
def test_time_series_ray_integral_preserves_time_and_point_axes(angle,modal_problem):
    # A misplaced time dimension or discarded negative incidence fails these
    # integrals, derived independently along moving coordinates.
    p = PhysicalParameters()
    model = modal_problem(p,{0:np.array([1000.]),2:np.array([200.])},
                           {0:np.array([80.]),2:np.array([300.])},refraction_angle=angle).solve_transient(num_axial_modes=32)
    x,y = np.array([-.002,.003])[:,None],np.array([.001,-.002])[None,:]
    times = [0.,.625,2.5]
    actual = model.opd_series(x,y,times,num_ray_points=48,point_chunk_size=2)
    prepared = model.prepare_opd(.625,num_ray_points=48,point_chunk_size=2)
    np.testing.assert_allclose(prepared(x,y),actual[1],rtol=1e-13)
    assert actual.shape==(3,2,2)
    assert np.all(actual[0]==0)
    for k,time in enumerate(times[1:],1):
        for i,xx in enumerate(x[:,0]):
            for j,yy in enumerate(y[0]):
                expected = p.dn_dT/np.cos(angle)*quad(lambda z:float(model.temperature(
                    np.hypot(xx,yy+z*np.tan(angle)),np.arctan2(yy+z*np.tan(angle),xx),z,time)),
                    0,p.thickness,epsabs=1e-12)[0]
                assert actual[k,i,j]==pytest.approx(expected,rel=1e-10,abs=1e-20)


@pytest.mark.parametrize('time', [-1., np.nan, np.inf])
def test_invalid_time_is_rejected(time,modal_problem):
    model = modal_problem(PhysicalParameters(),{0:np.array([1.])},{0:np.array([80.])}).solve_transient(num_axial_modes=8)
    with pytest.raises(ValueError):
        model.temperature(0., 0., 0., time)
