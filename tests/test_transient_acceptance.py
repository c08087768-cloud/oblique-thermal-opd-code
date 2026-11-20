"""Reject a numerically converged field when its physical constraints fail."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from thermal_opd.model import ThermalModel, ModalResolution, PreparedModel
from thermal_opd.parameters import PhysicalParameters
from thermal_opd import transient_validation as validation
from thermal_opd.convergence import field_errors


ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(boundary_tolerance_W_per_m2=1e-7, basis_tolerance=1e-10,
              pde_tolerance=1e-4, energy_tolerance=1e-5, limit_tolerance=1e-3,
              space_step_ratio=2e-4, time_step_s=1e-4)


def load_script(relative):
    spec = importlib.util.spec_from_file_location('_acceptance_'+Path(relative).stem, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_early_zero_depths_use_declared_front_scale_without_changing_old_norms():
    T = np.array([[1.,2.], [0.,0.], [0.,0.]])
    changed = T.copy(); changed[2,0] = .01
    error = field_errors(changed,T,np.ones(2),np.array([0.,1.]),np.array([0.,1.]),
                         temperature_normalization='global')
    assert error['temperature_relative_linf'] == [0.,0.,.005]
    with pytest.raises(ValueError,match='nonzero'):
        field_errors(changed,T,np.ones(2),np.array([0.,1.]),np.array([0.,1.]))


def test_acceptance_checks_initial_volume_and_time_independent_basis_constraints():
    check = getattr(validation,'transient_solution_acceptance',None)
    assert callable(check), 'Physical acceptance must be part of transient selection'
    prepared = ThermalModel(PhysicalParameters(),0.,ModalResolution(0,3,80,8,8)).prepare()
    report, fields = check(prepared.solve_transient(num_axial_modes=256), [.001,.625], CONFIG)
    checks = {c['name']:c for c in report['checks']}
    assert checks['initial_condition']['status'] == 'passed'
    assert checks['boundary_basis']['status'] == 'passed'
    assert checks['thermal_boundaries']['status'] == 'passed'
    assert np.all(fields['initial_temperature_K']==0)
    assert fields['boundary_times_s'][0]==0
    assert .001 in fields['boundary_times_s']
    # A wrong radial root breaks the all-time Robin relation even if the
    # caller supplies an apparently well-formed prepared model.
    bad = PreparedModel(prepared.parameters,0.,prepared.resolution,prepared.coefficients,
                        {0:prepared.roots[0]*1.01}).solve_transient(num_axial_modes=256)
    bad_report,_ = check(bad,[.001,.625],CONFIG)
    assert bad_report['status']=='failed'
    assert next(c for c in bad_report['checks'] if c['name']=='boundary_basis')['status']=='failed'


def test_full_batch_binds_independent_validation_to_this_batch_selection():
    suite = load_script('run_all.py')
    plan = suite.build_plan('full',[60.])
    names = [s['name'] for s in plan]
    stage = next(s for s in plan if s['name']=='semi_reference_theta60')
    assert names.index('transient_selection') < names.index(stage['name'])
    assert stage['needs']==['transient_selection']
    assert stage['args'][-2:]==['--selection','@transient_selection']


def test_miniature_selection_searches_J_and_reloads_physical_evidence(tmp_path,monkeypatch):
    script = load_script('scripts/convergence/select_transient_truncation.py')
    assert hasattr(script,'candidate_axial_counts'), 'J must participate in selection'
    monkeypatch.setattr(script,'repository_root',tmp_path)
    monkeypatch.setattr(script,'provenance',lambda *args: {'test_source':'miniature'})
    monkeypatch.setattr(script,'angles_deg',(0.,))
    monkeypatch.setattr(script,'times_s',(.001,.625))
    resolution = ModalResolution(0,3,80,8,8)
    monkeypatch.setattr(script,'reference',resolution)
    monkeypatch.setattr(script,'enlarged_reference',resolution)
    monkeypatch.setattr(script,'candidates',(resolution,))
    monkeypatch.setattr(script,'reference_J',256)
    monkeypatch.setattr(script,'enlarged_reference_J',512)
    monkeypatch.setattr(script,'candidate_axial_counts',(12,128,256))
    for name,value in dict(temperature_r_points=6,temperature_phi_points=8,
                           common_xy_points=9,verification_xy_points=13,figure_xy_points=17,
                           pv_seed_radius=3,pv_seed_phi=8).items():
        monkeypatch.setattr(script,name,value)
    import sys
    directory=tmp_path/'outputs/convergence/miniature'
    monkeypatch.setattr(sys,'argv',['selection','--output-dir',str(directory)])
    script.main()
    import json
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['selected_settings']['J']>12
    assert summary['acceptance_version']==1
    assert summary['transient_truncation_established'] is True
    assert validation.validate_saved_transient_run(directory)['status']=='PASS'
    verifier=load_script('scripts/verification/verify_semi_analytical.py')
    monkeypatch.setattr(verifier,'repository_root',tmp_path)
    monkeypatch.setattr(verifier,'provenance',lambda *args: {'test_source':'miniature'})
    verification_directory=tmp_path/'outputs/semi_analytical_verification/miniature'
    monkeypatch.setattr(sys,'argv',['verification','--angle','0','--selection',str(directory),
                                  '--output-dir',str(verification_directory)])
    verifier.main()
    recorded=json.loads((verification_directory/'summary.json').read_text())
    assert recorded['config']['num_axial_modes']==summary['selected_settings']['J']
    assert recorded['config']['times_s']==[.001,.625]
    report=json.loads((verification_directory/'verification.json').read_text())
    assert next(c for c in report['checks'] if c['name']=='initial_neighborhood_selected_J')['status']=='passed'
    from thermal_opd.validation_common import validate_saved_evidence_run
    assert validate_saved_evidence_run(verification_directory)['status']=='PASS'
    # Stop at the first field evaluation: exercise the real loader/model and
    # inspect what would be sent to FEM, without starting an expensive solve.
    sibling=load_script('scripts/cases/transient_fields.py')
    monkeypatch.setitem(sys.modules,'transient_fields',sibling)
    fem=load_script('scripts/cases/run_transient_case.py')
    monkeypatch.setattr(fem,'repository_root',tmp_path)
    monkeypatch.setattr(fem,'provenance',lambda *args: {'test_source':'miniature'})
    class ReachedFields(Exception):
        pass
    observed=[]
    def stop_at_fields(model,times,*args,**kwargs):
        observed.append((len(model.axial.beta),times.tolist()))
        raise ReachedFields
    monkeypatch.setattr(fem,'evaluate',stop_at_fields)
    monkeypatch.setattr(sys,'argv',['fem','--mode','smoke','--angle','0','--selection',str(directory),
                                  '--output-dir',str(tmp_path/'outputs/fem_comparison/miniature')])
    with pytest.raises(ReachedFields):fem.main()
    assert observed==[(summary['selected_settings']['J'],[0.,.001,.625])]
    path=directory/'data/acceptance.npz'
    with np.load(path) as saved:
        fields={key:saved[key] for key in saved.files}
    fields['theta0_initial_temperature_K'].flat[0]=1.
    np.savez_compressed(path,**fields)
    with pytest.raises(ValueError,match='acceptance'):
        validation.validate_saved_transient_run(directory)
