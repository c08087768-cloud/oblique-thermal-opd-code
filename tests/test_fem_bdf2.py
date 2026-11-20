"""BDF2 must preserve the spatial problem and expose its own time evidence."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from thermal_opd import fem
from thermal_opd.transient import PowerHistory
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.run_io import create_run, data_directory, write_summary

ARGS = (1., .5, 1.4, 2., 2., 0., lambda x,y: 6.+0*x)
KW = dict(disk_refinements=1, axial_layers=6, quadrature_order=4)


def response(dt, times=(0.,.4), **extra):
    return fem.solve_transient_fem(*ARGS, **KW, density=3., specific_heat=4.,
        times=times, time_step=dt, history=PowerHistory(), solver='direct', rtol=1e-12,
        time_integrator='bdf2', **extra)


def test_bdf2_is_second_order_against_independent_matrix_exponential():
    # Shares the assembled spatial problem, but not the time recurrence.
    system=fem.assemble_thermal_fem(*ARGS,**KW)
    capacity=12.*system.mass.toarray(); stiffness=system.matrix.toarray()
    equilibrium=np.linalg.solve(stiffness,system.rhs)
    exact=equilibrium-expm(-np.linalg.solve(capacity,stiffness)*.4)@equilibrium
    errors=[]
    for dt in (.02,.01,.005):
        result=response(dt)
        errors.append(np.linalg.norm(result.frames[-1].temperature_K.ravel()-exact))
        np.testing.assert_array_equal(result.frames[0].temperature_K,0.)
        assert result.diagnostics['time_integrator']=='bdf2'
        assert max(s['bdf_balance_absolute_error_W'] for s in result.steps) < 1e-9
    assert errors[0]/errors[1]>3.4
    assert errors[1]/errors[2]>3.4


def test_variable_step_formula_and_restart_are_recorded():
    result=response(.06,times=(0.,.02,.08,.11),time_step_by_interval_s=[.01,.03,.015])
    steps=result.steps
    assert [s['time_order'] for s in steps]==[1,2,1,2,2,2]
    assert steps[2]['bdf_alpha0']==1.  # Growth ratio 3 triggers a BE restart.
    assert steps[4]['bdf_alpha0']==pytest.approx(4/3)
    assert steps[4]['bdf_alpha1']==pytest.approx(-1.5)
    assert steps[4]['bdf_alpha2']==pytest.approx(1/6)
    energies=np.r_[0.,[s['stored_energy_J'] for s in steps]]
    for i,s in enumerate(steps):
        assert s['storage_rate_W']==pytest.approx((energies[i+1]-energies[i])/s['dt_s'],abs=1e-11)
        older=energies[max(0,i-1)]
        derivative=(s['bdf_alpha0']*(energies[i+1]-energies[i])+s['bdf_alpha2']*(older-energies[i]))/s['dt_s']
        assert s['bdf_storage_rate_W']==pytest.approx(derivative,abs=1e-11)
        # Existing field retains the actual secant energy-rate definition.
        assert s['balance_absolute_error_W']==pytest.approx(abs(s['storage_rate_W']+s['heat_loss_W']-s['input_W']))
        assert s['bdf_balance_absolute_error_W'] < 1e-9


def test_variable_schedule_matches_exact_response_after_refinement():
    system=fem.assemble_thermal_fem(*ARGS,**KW)
    equilibrium=np.linalg.solve(system.matrix.toarray(),system.rhs)
    exact=equilibrium-expm(-np.linalg.solve(12*system.mass.toarray(),system.matrix.toarray())*.4)@equilibrium
    errors=[]
    for scale in (1.,.5,.25):
        result=response(.03*scale,times=(0.,.04,.25,.4),time_step_by_interval_s=np.array([.005,.008,.03])*scale)
        errors.append(np.linalg.norm(result.frames[-1].temperature_K.ravel()-exact))
    assert errors[0]/errors[1]>3.
    assert errors[1]/errors[2]>3.


def test_invalid_method_is_rejected():
    with pytest.raises(ValueError,match='integrator'):
        fem.solve_transient_fem(*ARGS,**KW,density=3.,specific_heat=4.,times=[0.,.01],time_step=.01,
            history=PowerHistory(),solver='direct',rtol=1e-12,time_integrator='unknown')


def test_uniform_flux_matches_independent_finite_volume_reference():
    # Independent spatial discretization and exact-in-time tridiagonal response;
    # no prism matrices, Bessel roots or semi-analytical axial modes are shared.
    from thermal_opd.transient_validation import transient_uniform_flux_1d_reference
    times=np.array([0.,.05,.15,.4]);depths=np.array([0.,.25,.5])
    references=[]
    for n in (512,1024):
        z,T=transient_uniform_flux_1d_reference(.5,1.4,12.,6.,2.,2.,times,num_intervals=n)
        references.append(np.stack([np.interp(depths,z,row) for row in T]))
    assert np.max(abs(references[1]-references[0]))/np.max(references[1]) < 1e-4
    result=fem.solve_transient_fem(*ARGS,disk_refinements=1,axial_layers=128,quadrature_order=4,
        density=3.,specific_heat=4.,times=times,time_step=.0005,history=PowerHistory(),
        solver='direct',rtol=1e-12,time_integrator='bdf2')
    actual=np.stack([f.temperature_at(.1,.2,depths) for f in result.frames])
    assert np.max(abs(actual-references[-1]))/np.max(references[-1]) < 5e-4


def write_evidence(root, dt, method='bdf2'):
    p=PhysicalParameters(); times=np.array([0.,.001,.0025])
    result=fem.solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
        lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,0.,p.absorbed_power),
        density=p.density,specific_heat=p.specific_heat,times=times,time_step=dt,
        history=PowerHistory(),disk_refinements=1,axial_layers=12,quadrature_order=6,
        solver='direct',rtol=1e-12,time_integrator=method)
    source=create_run(root,'fem_comparison'); keys=list(result.steps[0]);frame=result.frames[0]
    np.savez_compressed(data_directory(source)/'fem_run0.npz',disk_nodes_xy_m=frame.disk_mesh.p,
        disk_triangles=frame.disk_mesh.t,axial_nodes_z_m=frame.z_m,times_s=times,
        temperature_K=np.stack([f.temperature_K for f in result.frames]),step_keys=np.array(keys),
        steps=np.array([[s[k] for k in keys] for s in result.steps]))
    write_summary(source,dict(kind='transient_fem_joint_case',status='COMPLETE',angle_deg=0.,
        physical_parameters_SI=asdict(p),history=asdict(PowerHistory()),times_s=times.tolist(),
        cases=[dict(index=0,diagnostics=result.diagnostics)]))
    return source


@pytest.fixture
def saved_pair(tmp_path):
    a,b=write_evidence(tmp_path,.001/8),write_evidence(tmp_path,.001/16)
    out=create_run(tmp_path,'fem_convergence_validation')
    manifest=dict(angles_deg=[0.],times_s=[0.,.001,.0025],temperature_xy_points=9,opd_xy_points=9,
        cases=[dict(angle_deg=0.,candidate=dict(source_run=str(a),case_index=0),
            time_reference=dict(source_run=str(b),case_index=0))])
    return out,manifest


def test_saved_bdf2_diagnostics_are_independently_recomputed(saved_pair,monkeypatch):
    from thermal_opd.transient_validation import transient_fem_convergence_report
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Saved validation reran FEM'))
    report,_=transient_fem_convergence_report(*saved_pair)
    named={c['name']:c for c in report['checks']}
    for name in ('initial_condition','energy_balance','weak_equation_residual','robin_loss_diagnostics','prescribed_load'):
        assert named['theta0_'+name]['status']=='passed'
    assert not report['fem_convergence_established']  # Missing spatial and joint evidence.


@pytest.mark.parametrize('field', ['bdf_alpha2','bdf_storage_rate_W','bdf_balance_absolute_error_W'])
def test_corrupted_bdf2_evidence_is_rejected(saved_pair,field):
    from thermal_opd.transient_validation import transient_fem_convergence_report
    out,manifest=saved_pair
    file=data_directory(Path(manifest['cases'][0]['candidate']['source_run']))/'fem_run0.npz'
    with np.load(file) as old:
        data={k:old[k] for k in old.files}
    j=data['step_keys'].tolist().index(field)
    data['steps'][2,j]+=1.
    np.savez_compressed(file,**data)
    with pytest.raises(ValueError):
        transient_fem_convergence_report(out,manifest)


def test_different_integrators_cannot_claim_isolated_time_convergence(saved_pair):
    from thermal_opd.transient_validation import transient_fem_convergence_report
    out,manifest=saved_pair
    source=write_evidence(out.parent.parent,.001/32,method='backward_euler')
    manifest['cases'][0]['time_reference']['source_run']=str(source)
    report,_=transient_fem_convergence_report(out,manifest)
    check=next(c for c in report['checks'] if c['name']=='theta0_time_convergence')
    assert check['metrics']['isolated_parameter_change'] is False
    assert check['status']=='failed'
