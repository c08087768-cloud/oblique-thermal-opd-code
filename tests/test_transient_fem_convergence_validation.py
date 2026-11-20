"""Saved FEM refinement evidence must be recomputed without another PDE solve."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import csv
from pathlib import Path

import numpy as np
import pytest

from thermal_opd.fem import solve_transient_fem
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import create_run,data_directory,write_summary,write_json
from thermal_opd.transient import PowerHistory


@pytest.fixture
def evidence(tmp_path,request):
    p=PhysicalParameters();times=np.array(getattr(request,'param',[0.,.001,.0025]))
    roles={}
    for role,xy,z,dt in (
        ('candidate',1,8,.001/8),('xy_reference',2,8,.001/8),
        ('z_reference',1,16,.001/8),('time_reference',1,8,.001/16),
        ('joint_reference',2,16,.001/16)):
        result=solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
            lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,0.,p.absorbed_power),
            density=p.density,specific_heat=p.specific_heat,times=times,time_step=dt,
            history=PowerHistory(),disk_refinements=xy,axial_layers=z,quadrature_order=6,
            solver='direct',rtol=1e-12)
        source=create_run(tmp_path,'fem_comparison')
        keys=list(result.steps[0]);frame=result.frames[0]
        np.savez_compressed(data_directory(source)/'fem_run0.npz',disk_nodes_xy_m=frame.disk_mesh.p,
            disk_triangles=frame.disk_mesh.t,axial_nodes_z_m=frame.z_m,times_s=times,
            temperature_K=np.array([f.temperature_K for f in result.frames]),
            step_keys=np.array(keys),steps=np.array([[s[k] for k in keys] for s in result.steps]))
        write_summary(source,dict(kind='transient_modal_fem_study',status='COMPLETE',angle_deg=0.,
            physical_parameters_SI=asdict(p),history=asdict(PowerHistory()),times_s=times.tolist(),
            fem_runs=[dict(index=0,diagnostics=result.diagnostics)]))
        roles[role]=dict(source_run=str(source),case_index=0)
    out=create_run(tmp_path,'fem_convergence_validation')
    manifest=dict(angles_deg=[0.],times_s=times.tolist(),temperature_xy_points=15,opd_xy_points=13,
        cases=[dict(angle_deg=0.,**roles)])
    return out,manifest


def evaluate(evidence):
    from thermal_opd.transient_validation import transient_fem_convergence_report
    return transient_fem_convergence_report(*evidence)


def named(report,name):
    return next(c for c in report['checks'] if c['name']==name)


@pytest.mark.parametrize('failing_index,expected',[(0,'passed'),(1,'failed')])
def test_diagnostic_1ms_is_recorded_but_only_other_times_gate(evidence,monkeypatch,failing_index,expected):
    from thermal_opd import transient_validation as validation
    out,manifest=evidence;manifest['diagnostic_only_times_s']=[.001]
    calls=[]
    def planted_errors(*args,**kwargs):
        index=len(calls)%2;calls.append(index)
        value=.5 if index==failing_index else 0.
        return dict(temperature_relative_l2=[value]*10,temperature_relative_linf=[value]*10,
            opd_piston_removed_rms_over_reference_pv=value,
            opd_piston_removed_linf_over_reference_pv=value,opd_relative_pv=value)
    monkeypatch.setattr(validation,'field_errors',planted_errors)
    report,_=evaluate((out,manifest))
    for direction in ('xy','z','time','joint'):
        check=named(report,f'theta0_{direction}_convergence');m=check['metrics']
        assert check['status']==expected
        assert m['times_s']==[.001,.0025]
        assert m['used_for_gate_by_time']==[False,True]
        assert m['errors_by_time'][failing_index]['opd_relative_pv']==.5


def test_only_diagnostic_frame_cannot_establish_convergence(evidence):
    out,manifest=evidence;manifest.update(times_s=[0.,.001],diagnostic_only_times_s=[.001])
    report,_=evaluate((out,manifest))
    for direction in ('xy','z','time','joint'):
        check=named(report,f'theta0_{direction}_convergence')
        assert check['status']=='not_run'
        assert check['metrics']['used_for_gate_by_time']==[False]
        assert len(check['metrics']['errors_by_time'])==1
    assert not report['fem_convergence_established']


def test_unapproved_diagnostic_exclusions_are_rejected(evidence):
    out,manifest=evidence;manifest['diagnostic_only_times_s']=[.001,.0025]
    with pytest.raises(ValueError,match='diagnostic'):
        evaluate((out,manifest))


def test_diagnostic_time_still_has_all_step_physical_checks(evidence):
    out,manifest=evidence;manifest['diagnostic_only_times_s']=[.001]
    source=Path(manifest['cases'][0]['candidate']['source_run'])
    path=source/'data/fem_run0.npz'
    with np.load(path) as saved:values={key:saved[key] for key in saved.files}
    keys=values['step_keys'].tolist()
    index=np.flatnonzero(np.isclose(values['steps'][:,keys.index('time_s')],.001,rtol=0,atol=1e-15))
    assert index.size==1
    values['steps'][index[0],keys.index('relative_linear_residual')]=1e-3
    np.savez_compressed(path,**values)
    report,_=evaluate((out,manifest))
    assert named(report,'theta0_weak_equation_residual')['status']=='failed'
    assert not report['fem_convergence_established']


@pytest.mark.parametrize('evidence',[[0.,.0025]],indirect=True)
def test_missing_1ms_is_missing_diagnostic_evidence_not_a_numerical_failure(evidence):
    out,manifest=evidence;manifest['diagnostic_only_times_s']=[.001]
    report,_=evaluate((out,manifest))
    check=named(report,'theta0_diagnostic_recording')
    assert check['status']=='not_run'
    assert check['threshold'] is None
    assert all(times==[.001] for times in check['metrics']['missing_times_by_role_s'].values())
    assert not report['fem_convergence_established']


def test_recomputes_all_four_directions_and_exact_initial_nodes(evidence,monkeypatch):
    from thermal_opd import fem
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Validation reran FEM'))
    report,fields=evaluate(evidence)
    assert named(report,'theta0_initial_condition')['status']=='passed'
    assert named(report,'requested_scope')['status']=='passed'
    for direction in ('xy','z','time','joint'):
        check=named(report,f'theta0_{direction}_convergence')
        assert check['threshold']==1e-3
        assert check['metrics']['isolated_parameter_change'] is True
        assert check['metrics']['times_s']==[.001,.0025]
        assert len(check['metrics']['errors_by_time'])==2
    assert 'theta0_candidate_temperature_K' in fields
    assert np.all(fields['theta0_candidate_temperature_K'][0]==0.)
    assert report['fem_convergence_established'] is False  # These deliberately coarse meshes do fail.
    assert named(report,'theta0_robin_loss_diagnostics')['status']=='passed'


def test_missing_references_and_time_coverage_are_not_run(evidence):
    out,manifest=evidence
    manifest['cases'][0]={k:v for k,v in manifest['cases'][0].items() if k in ('angle_deg','candidate')}
    manifest['times_s'].append(.625)
    report,_=evaluate((out,manifest))
    assert named(report,'requested_scope')['status']=='not_run'
    assert all(named(report,f'theta0_{d}_convergence')['status']=='not_run' for d in ('xy','z','time','joint'))
    assert not report['fem_convergence_established']


def test_extra_saved_times_cover_requested_subset_without_extending_claim(evidence):
    out,manifest=evidence;manifest['times_s']=[0.,.001]
    report,_=evaluate((out,manifest))
    assert named(report,'requested_scope')['status']=='passed'
    assert named(report,'theta0_time_convergence')['metrics']['times_s']==[.001]


@pytest.mark.parametrize('evidence',[[0.,.001,.625]],indirect=True)
def test_late_rear_error_uses_its_own_small_temperature_scale(evidence):
    report,fields=evaluate(evidence)
    check=named(report,'theta0_z_convergence')
    assert check['metrics']['temperature_normalization_by_time']==['global','per_plane']
    reference=fields['theta0_z_reference_temperature_K'][-1]
    delta=fields['theta0_candidate_temperature_K'][-1]-reference
    own_scale=np.linalg.norm(reference[2])
    global_scale=np.max(np.linalg.norm(reference,axis=1))
    actual=check['metrics']['errors_by_time'][-1]['temperature_relative_l2'][2]
    assert actual==pytest.approx(np.linalg.norm(delta[2])/own_scale,rel=1e-12)
    assert global_scale/own_scale>100  # Global scaling would conceal this cold rear plane.


def test_joint_refinement_cannot_be_mislabeled_as_xy_only(evidence):
    out,manifest=evidence
    manifest['cases'][0]['xy_reference']=manifest['cases'][0]['joint_reference']
    report,_=evaluate((out,manifest))
    check=named(report,'theta0_xy_convergence')
    assert check['status']=='failed'
    assert check['metrics']['isolated_parameter_change'] is False


def test_nonzero_initial_nodes_fail_even_if_sampled_points_miss_them(evidence):
    out,manifest=evidence
    path=Path(manifest['cases'][0]['candidate']['source_run'])/'data/fem_run0.npz'
    with np.load(path) as f: values={k:f[k] for k in f.files}
    values['temperature_K'][0,-1,-1]=1e-4
    np.savez_compressed(path,**values)
    report,_=evaluate((out,manifest))
    assert named(report,'theta0_initial_condition')['status']=='failed'


def test_historical_early_final_and_initial_array_format_is_readable(evidence):
    out,manifest=evidence
    source=Path(manifest['cases'][0]['candidate']['source_run'])
    with np.load(source/'data/fem_run0.npz') as f: values={k:f[k] for k in f.files}
    values['temperature_initial_K']=values['temperature_K'][0]
    values['temperature_K']=values['temperature_K'][-1]
    values.pop('times_s')
    np.savez_compressed(source/'data/fem_run0.npz',**values)
    s=json.loads((source/'summary.json').read_text());s['times_s']=[0.,.0025]
    s['kind']='early_transverse_fem_refinement';write_summary(source,s)
    manifest['cases'][0]={k:v for k,v in manifest['cases'][0].items() if k in ('angle_deg','candidate')}
    manifest['times_s']=[0.,.0025]
    report,_=evaluate((out,manifest))
    assert named(report,'theta0_initial_condition')['status']=='passed'


def test_saved_validation_detects_modified_source_data(evidence):
    from thermal_opd.transient_validation import validate_saved_transient_run
    out,manifest=evidence;report,fields=evaluate(evidence)
    np.savez_compressed(data_directory(out)/'comparison_fields.npz',**fields)
    write_json(out/'verification.json',report)
    write_summary(out,dict(kind='transient_fem_convergence_validation',status='COMPLETE',config=manifest,
        fem_convergence_established=report['fem_convergence_established']))
    result=validate_saved_transient_run(out)
    assert result['status']=='PASS' and result['verification']['status']==report['status']
    path=Path(manifest['cases'][0]['candidate']['source_run'])/'data/fem_run0.npz'
    with np.load(path) as f: values={k:f[k] for k in f.files}
    values['temperature_K'][-1,0,0]+=1.
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='evidence|metric|source|field'):
        validate_saved_transient_run(out)


def test_invalid_requested_scope_and_relaxed_threshold_are_rejected(evidence):
    out,manifest=evidence
    for updates in (dict(tolerance=.01),dict(times_s=[.001,.0025]),dict(angles_deg=[0.,0.])):
        altered=deepcopy(manifest);altered.update(updates)
        with pytest.raises(ValueError): evaluate((out,altered))


def test_new_spatial_policy_keeps_time_physics_and_numerical_fields_unchanged(evidence):
    out,manifest=evidence
    old_report,old_fields=evaluate(evidence)
    current=deepcopy(manifest)
    current.update(space_tolerance=.005,joint_tolerance=.005,time_tolerance=.001)
    report,fields=evaluate((out,current))
    for direction in ('xy','z','joint'):
        assert named(report,f'theta0_{direction}_convergence')['threshold']==.005
        assert named(report,f'theta0_{direction}_convergence')['metrics']==named(old_report,f'theta0_{direction}_convergence')['metrics']
    assert named(report,'theta0_time_convergence')['threshold']==.001
    for name in ('initial_condition','energy_balance','weak_equation_residual','robin_loss_diagnostics','prescribed_load'):
        assert named(report,'theta0_'+name)==named(old_report,'theta0_'+name)
    assert old_fields.keys()==fields.keys()
    for key in fields:
        np.testing.assert_array_equal(fields[key],old_fields[key])
    assert not report['fem_convergence_established']  # These test meshes remain too coarse even at .5%.


def test_spatial_policy_does_not_relax_time_or_allow_larger_unapproved_thresholds(evidence):
    out,manifest=evidence
    for updates in (dict(space_tolerance=.005,joint_tolerance=.005,time_tolerance=.005),
                    dict(space_tolerance=.01,joint_tolerance=.01,time_tolerance=.001)):
        current=deepcopy(manifest);current.update(updates)
        with pytest.raises(ValueError): evaluate((out,current))


def test_validation_does_not_allocate_whole_large_npz_for_hashing(evidence,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Hashing large FEM files must stream bytes instead of allocating the entire archive')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    evaluate(evidence)


def test_cli_writes_standard_artifacts_and_strict_exit(evidence,monkeypatch):
    out,manifest=evidence
    source=Path(__file__).resolve().parents[1]/'scripts/verification/verify_transient_fem_convergence.py'
    # Keep the real entry's normal repository-relative provenance contract.
    entry=out.parents[2]/'scripts/verification'/source.name
    entry.parent.mkdir(parents=True);entry.write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('_transient_fem_validation_cli',entry)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    path=out.parents[2]/'manifest.json';write_json(path,manifest)
    result=cli.main(['--manifest',str(path)])
    for file in ('config.json','metrics.json','verification.json','check_report.json',
        'data/comparison_fields.npz','data/convergence_by_time.csv','figures/convergence_by_time.png'):
        assert (result/file).is_file()
    s=json.loads((result/'summary.json').read_text())
    assert s['status']=='COMPLETE' and s['fem_convergence_established'] is False
    assert s['config']['space_tolerance']==.005 and s['config']['joint_tolerance']==.005
    assert s['config']['time_tolerance']==.001
    assert s['config']['diagnostic_only_times_s']==[.001]
    with (result/'data/convergence_by_time.csv').open(encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        assert float(row['threshold_fraction'])==(.001 if row['check'].endswith('_time_convergence') else .005)
        assert row['used_for_gate']==('False' if float(row['time_s'])==.001 else 'True')
    diagnostic=json.loads((result/'data/diagnostic_0p001s.json').read_text())
    assert len(diagnostic)==4 and all(row['used_for_gate'] is False for row in diagnostic)
    assert not Path(s['config']['cases'][0]['candidate']['source_run']).is_absolute()
    with pytest.raises(SystemExit,match='failed|incomplete'):
        cli.main(['--manifest',str(path),'--require-scientific-pass'])
    # Explicit old manifests retain the original interpretation on replay.
    manifest['tolerance']=.001;write_json(path,manifest)
    legacy=cli.main(['--manifest',str(path)])
    report=json.loads((legacy/'verification.json').read_text())
    assert all(c['threshold']==.001 for c in report['checks'] if c['name'].endswith('_convergence'))


def test_failed_source_and_corrupt_robin_diagnostic_are_preserved_as_failed(evidence):
    out,manifest=evidence
    source=Path(manifest['cases'][0]['candidate']['source_run'])
    s=json.loads((source/'summary.json').read_text());s['status']='FAILED';write_summary(source,s)
    path=source/'data/fem_run0.npz'
    with np.load(path) as f: values={k:f[k] for k in f.files}
    column=values['step_keys'].tolist().index('heat_loss_W')
    values['steps'][-1,column]+=1e-5
    np.savez_compressed(path,**values)
    report,_=evaluate((out,manifest))
    assert named(report,'theta0_source_completion')['status']=='failed'
    assert named(report,'theta0_robin_loss_diagnostics')['status']=='failed'
    assert not report['fem_convergence_established']
