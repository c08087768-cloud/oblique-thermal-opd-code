import numpy as np
import pytest
pytest.importorskip('skfem')
from thermal_opd.fem import solve_transient_fem, solve_steady_fem
from thermal_opd.transient import PowerHistory


def test_transient_fem_zero_initial_energy_and_timestep_refinement():
    # A missing consistent capacity matrix, incorrect front input, or stepping
    # past observation times breaks the independent axial-series comparison.
    args = (1., .5, 1.4, 2., 2., 0., lambda x,y: 6.+0*x)
    kw = dict(disk_refinements=1,axial_layers=64,quadrature_order=4,solver='direct',rtol=1e-11)
    from scipy.optimize import brentq
    from scipy.integrate import quad
    beta = np.array([brentq(lambda x:x-2*np.arctan2(2*.5/1.4,x)-j*np.pi,
                           j*np.pi,(j+1)*np.pi)/.5 for j in range(80)])
    def X(j,z):
        return np.cos(beta[j]*z)+2/(1.4*beta[j])*np.sin(beta[j]*z)
    norms = np.array([quad(lambda z:X(j,z)**2,0,.5)[0] for j in range(80)])
    expected = sum(6/(1.4*b*b*norms[j])*X(j,.25)*(-np.expm1(-1.4*b*b/12*.37))
                   for j,b in enumerate(beta))
    errors = []
    for dt in (.02,.01):
        result = solve_transient_fem(*args,density=3.,specific_heat=4.,times=[0.,.37,40.],
                                    time_step=dt,history=PowerHistory(),**kw)
        np.testing.assert_array_equal(result.frames[0].temperature_K,0.)
        np.testing.assert_array_equal(result.times_s,[0.,.37,40.])
        errors.append(abs(float(result.frames[1].temperature_at(.1,.2,.25))-expected))
        for step in result.steps:
            assert abs(step['storage_rate_W']+step['heat_loss_W']-step['input_W']) < 1e-9
        steady = solve_steady_fem(*args,**kw)
        np.testing.assert_allclose(result.frames[-1].temperature_K,steady.temperature_K,rtol=1e-9)
    assert errors[1] < .65*errors[0]


def test_transient_fem_ramp_load_and_zero_input_keep_energy_balance():
    args = (1., .5, 1.4, 2., 2., 0.)
    kw = dict(density=3.,specific_heat=4.,times=[0.,.1,.3],time_step=.07,
              disk_refinements=1,axial_layers=3,quadrature_order=4,solver='cg',rtol=1e-11)
    for flux in (0.,6.):
        result = solve_transient_fem(*args,lambda x,y:flux+0*x,history=PowerHistory('linear',.2),**kw)
        for row in result.steps:
            assert row['input_W'] == pytest.approx(result.diagnostics['full_input_W']*min(row['time_s']/.2,1.),abs=1e-12)
            assert abs(row['storage_rate_W']+row['heat_loss_W']-row['input_W']) < 1e-9
        if flux == 0:
            np.testing.assert_array_equal(result.frames[-1].temperature_K,0.)


@pytest.mark.parametrize('times,dt', [([.1,.1],.1),([-.1,.2],.1),([0.,np.nan],.1),([0.,.1],0.)])
def test_transient_fem_rejects_invalid_schedule(times,dt):
    with pytest.raises(ValueError):
        solve_transient_fem(1.,.5,1.4,2.,2.,0.,lambda x,y:1.+0*x,
            density=3.,specific_heat=4.,times=times,time_step=dt,history=PowerHistory(),
            disk_refinements=1,axial_layers=3,quadrature_order=4,solver='direct',rtol=1e-10)
