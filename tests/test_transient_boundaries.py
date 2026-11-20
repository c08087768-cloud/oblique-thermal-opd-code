"""Boundary checks must detect wrong signs/roots, not just small field changes."""
from dataclasses import replace
import json
import numpy as np
import pytest

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.transient import PowerHistory
from thermal_opd.model import PreparedModel,ModalResolution


def make_model(kind='step', wrong_side=False):
    p = PhysicalParameters()
    roots = {m: radial_eigenvalues(m,p.radius,p.conductivity,
                                  p.h_eff*(2 if wrong_side else 1),3) for m in (0,2)}
    c = {0:np.array([1000.,-120.,30.]),2:np.array([200.,50.,-10.])}
    return PreparedModel(p,60.,ModalResolution(2,3,80,16,32),c,roots).solve_transient(num_axial_modes=12,history=PowerHistory(kind,5.))


@pytest.mark.parametrize('kind',['step','linear','exponential'])
def test_transient_analytic_gradients_match_independent_finite_differences(kind):
    model = make_model(kind)
    r,phi,z = np.array([.002,.006]),np.array([.2,1.1]),np.array([.001,.003])
    dr,dz = model.temperature_gradient(r,phi,z,.625)
    eps = 1e-7
    radial = (model.temperature(r+eps,phi,z,.625)-model.temperature(r-eps,phi,z,.625))/(2*eps)
    axial = (model.temperature(r,phi,z+eps,.625)-model.temperature(r,phi,z-eps,.625))/(2*eps)
    np.testing.assert_allclose(dr,radial,rtol=2e-7,atol=1e-7)
    np.testing.assert_allclose(dz,axial,rtol=2e-7,atol=1e-7)
    zero = model.temperature_gradient(r,phi,z,0.)
    np.testing.assert_array_equal(zero,np.zeros((2,2)))


@pytest.mark.parametrize('kind',['step','linear','exponential'])
def test_all_three_transient_boundaries_include_time_dependent_front_load(kind):
    from thermal_opd.transient_validation import transient_boundary_fields
    model = make_model(kind)
    p = model.prepared.parameters
    times = np.array([0.,.001,.625,2.5,75.,2500.])
    r,phi,z = np.linspace(0,p.radius,11),np.arange(16)*2*np.pi/16,np.linspace(0,p.thickness,13)
    f = transient_boundary_fields(model,times,r,phi,z)
    for key in ('front_modal_residual_W_per_m2','rear_residual_W_per_m2','side_residual_W_per_m2'):
        assert np.max(np.abs(f[key]))<1e-8
        np.testing.assert_array_equal(f[key][0],np.zeros_like(f[key][0]))
    np.testing.assert_allclose(f['front_load_W_per_m2'][1],
                               model.history.value(times[1])*f['modal_source_W_per_m2'])
    wrong_sign = p.h_eff*f['front_temperature_K']+p.conductivity*f['front_dT_dz_K_per_m']-f['front_load_W_per_m2']
    assert np.max(np.abs(wrong_sign))>10.
    bad = transient_boundary_fields(make_model(kind,wrong_side=True),times,r,phi,z)
    assert np.max(np.abs(bad['side_residual_W_per_m2']))>1e-2


def test_saved_boundary_checker_reloads_derivatives_and_rejects_missing_or_corrupted_report(tmp_path):
    from thermal_opd.transient_validation import write_transient_boundary_check, validate_transient_boundary_check
    from dataclasses import asdict
    model = make_model('linear')
    p = model.prepared.parameters
    times,r,phi,z = np.array([0.,.625,75.]),np.linspace(0,p.radius,9),np.arange(12)*2*np.pi/12,np.linspace(0,p.thickness,7)
    fields = dict(times_s=times,r_m=r,phi_rad=phi,orders=np.array(list(model.prepared.coefficients)))
    fields['temperature_sa_K'] = np.stack([np.stack([model.temperature(r[:,None],phi[None,:],zz,t)
        for zz in (0.,p.thickness/2,p.thickness)]) for t in times])
    for m in model.prepared.coefficients:
        fields[f'source_coefficients_m{m}_W_per_m2'] = model.prepared.coefficients[m]
        fields[f'radial_roots_m{m}_per_m'] = model.prepared.roots[m]
    np.savez(tmp_path/'fields.npz',**fields)
    summary = dict(kind='transient_modal_fem_study',times_s=times.tolist(),physical_parameters_SI=asdict(p),
        history=asdict(model.history),axial_counts=[12],angle_deg=60.,modal_settings=asdict(model.prepared.resolution),boundary_validation_required=True)
    with pytest.raises(ValueError,match='Missing.*boundary'):
        validate_transient_boundary_check(tmp_path,summary)
    write_transient_boundary_check(tmp_path,model,times,r,phi,z,relative_tolerance=1e-10,absolute_tolerance=1e-9)
    assert validate_transient_boundary_check(tmp_path,summary)['status']=='PASS'
    with np.load(tmp_path/'boundary_checks.npz') as saved:
        boundary = {key:saved[key] for key in saved.files}
    boundary['front_dT_dz_K_per_m'][1,2,3] += 1.
    np.savez(tmp_path/'boundary_checks.npz',**boundary)
    with pytest.raises(ValueError,match='boundary|Boundary'):
        validate_transient_boundary_check(tmp_path,summary)
    write_transient_boundary_check(tmp_path,model,times,r,phi,z,relative_tolerance=1e-10,absolute_tolerance=1e-9)
    report = json.loads((tmp_path/'boundary_summary.json').read_text())
    report['max_absolute_residuals_W_per_m2'][1][0] += .1
    (tmp_path/'boundary_summary.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='boundary|Boundary'):
        validate_transient_boundary_check(tmp_path,summary)


def test_legacy_missing_boundary_files_is_explicitly_not_run(tmp_path):
    from thermal_opd.transient_validation import validate_transient_boundary_check
    assert validate_transient_boundary_check(tmp_path,{})['status']=='NOT_RUN'


def test_boundary_history_includes_every_snapshot_and_rejects_missing_history_time(tmp_path):
    from thermal_opd.transient_validation import write_transient_boundary_check,validate_transient_boundary_check
    from dataclasses import asdict
    model=make_model();p=model.prepared.parameters
    times=np.array([0.,.625,75.]);history_times=np.array([0.,.001,.2,.625,2.5,75.])
    r,phi,z=np.linspace(0,p.radius,5),np.arange(8)*2*np.pi/8,np.linspace(0,p.thickness,7)
    fields=dict(times_s=times,r_m=r,phi_rad=phi,orders=np.array(list(model.prepared.coefficients)))
    fields['temperature_sa_K']=np.stack([np.stack([model.temperature(r[:,None],phi[None,:],zz,t)
        for zz in (0.,p.thickness/2,p.thickness)]) for t in times])
    for m in model.prepared.coefficients:
        fields[f'source_coefficients_m{m}_W_per_m2']=model.prepared.coefficients[m]
        fields[f'radial_roots_m{m}_per_m']=model.prepared.roots[m]
    np.savez(tmp_path/'fields.npz',**fields)
    np.savez(tmp_path/'history.npz',times_s=history_times)
    summary=dict(kind='transient_modal_fem_study',times_s=times.tolist(),physical_parameters_SI=asdict(p),
        history=asdict(model.history),axial_counts=[12],angle_deg=60.,modal_settings=asdict(model.prepared.resolution),
        boundary_validation_required=True,boundary_history_required=True)
    write_transient_boundary_check(tmp_path,model,history_times,r,phi,z,relative_tolerance=1e-10,absolute_tolerance=1e-9)
    assert validate_transient_boundary_check(tmp_path,summary)['status']=='PASS'
    np.savez(tmp_path/'history.npz',times_s=np.array([0.,.001,.625,2.5,75.]))
    with pytest.raises(ValueError,match='history times'):
        validate_transient_boundary_check(tmp_path,summary)


def test_zero_source_boundary_normalization_stays_finite(tmp_path):
    from thermal_opd.transient_validation import write_transient_boundary_check
    model = make_model()
    prepared=model.prepared
    model = PreparedModel(replace(prepared.parameters,absorbed_power=0.),prepared.angle_deg,prepared.resolution,
        {m:np.zeros_like(c) for m,c in prepared.coefficients.items()},prepared.roots).solve_transient(num_axial_modes=12)
    p = model.prepared.parameters
    report = write_transient_boundary_check(tmp_path,model,np.array([0.,.625]),np.linspace(0,p.radius,5),
        np.arange(8)*2*np.pi/8,np.linspace(0,p.thickness,5),relative_tolerance=1e-10,absolute_tolerance=1e-9)
    assert report['status']=='PASS'
    assert np.all(np.isfinite(report['max_residuals_over_full_power_peak_flux']))
