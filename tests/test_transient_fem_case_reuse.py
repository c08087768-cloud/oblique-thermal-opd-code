"""The formal case consumes reviewed nodes and never substitutes a coarse solve."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from thermal_opd import transient_validation as validation
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.model import ModalResolution
from thermal_opd.run_io import file_sha256, write_summary, write_json, data_directory, read_json
from thermal_opd.transient import PowerHistory
from test_transient_fem_p2_saved import saved_case


ROOT=Path(__file__).resolve().parents[1]
SOURCES=('fem.py','parameters.py','heat_source.py','transient.py')


def certificate(tmp_path):
    source,header,out,manifest=saved_case(tmp_path)
    manifest['solver_source_sha256']={f'src/thermal_opd/{name}':file_sha256(ROOT/'src/thermal_opd'/name) for name in SOURCES}
    report,fields=validation.transient_fem_convergence_report(out,manifest)
    np.savez_compressed(data_directory(out)/'comparison_fields.npz',**fields)
    write_json(out/'verification.json',report)
    summary=dict(kind='transient_fem_convergence_validation',status='COMPLETE',config=manifest,
        fem_convergence_established=False)
    write_summary(out,summary)
    return source,header,out,summary


def loader():
    value=getattr(validation,'load_transient_fem_certificate',None)
    assert callable(value),'Missing formal FEM certificate loader'
    return value


def test_full_mode_requires_certificate_before_creating_output(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('_transient_reuse_case',ROOT/'scripts/cases/run_transient_case.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    output=tmp_path/'case'
    monkeypatch.setattr(sys,'argv',['run_transient_case.py','--mode','full','--output-dir',str(output)])
    monkeypatch.setattr(entry,'create_run',lambda *a,**k:pytest.fail('Full mode started work before certificate check'))
    with pytest.raises(ValueError,match='fem-convergence'):
        entry.main()
    assert not output.exists()


def test_consistent_failed_certificate_is_not_formal_evidence(tmp_path):
    _,_,out,_=certificate(tmp_path)
    with pytest.raises(ValueError,match='passed|convergence'):
        loader()(out,PhysicalParameters(),0.,PowerHistory(),[0.,.001,.0025],repository_root=ROOT)


def test_reviewed_candidate_restores_p2_frames_without_solver_or_node_copy(tmp_path,monkeypatch):
    source,header,out,summary=certificate(tmp_path)
    calls=[]
    def reviewed(path,stored):
        calls.append(Path(path))
        report=read_json(out/'verification.json');report.update(status='passed',fem_convergence_established=True)
        return dict(verification=report,fem_convergence_established=True)
    # This test isolates consumption after the independent review trust boundary;
    # the preceding test exercises a real (consistent but failed) certificate.
    monkeypatch.setattr(validation,'validate_saved_transient_fem_convergence',reviewed)
    summary['fem_convergence_established']=True;write_summary(out,summary)
    from thermal_opd import fem
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Certificate reuse reran FEM'))
    loaded=loader()(out,PhysicalParameters(),0.,PowerHistory(),[0.,.0025],repository_root=ROOT)
    assert calls==[out]
    assert loaded['header']['file']==source/'data/fem_run0.npz'
    assert loaded['solution'].times_s.tolist()==[0.,.0025]
    assert all(frame.field_degree==2 for frame in loaded['solution'].frames)
    with np.load(source/'data/fem_run0.npz') as raw:
        np.testing.assert_array_equal(loaded['solution'].frames[-1].temperature_K,raw['temperature_K'][-1])
    assert not (out/'data/fem_run0.npz').exists()


@pytest.mark.parametrize('change,match',[
    ('physics','physical'),('history','history'),('angle','angle'),('times','time'),('source','source')])
def test_candidate_scope_and_current_solver_source_are_required(tmp_path,monkeypatch,change,match):
    _,_,out,summary=certificate(tmp_path)
    summary['fem_convergence_established']=True
    report=read_json(out/'verification.json');report.update(status='passed',fem_convergence_established=True)
    monkeypatch.setattr(validation,'validate_saved_transient_fem_convergence',lambda *a:dict(
        verification=report,fem_convergence_established=True))
    p,angle,history,times=PhysicalParameters(),0.,PowerHistory(),[0.,.001,.0025]
    if change=='physics':p=PhysicalParameters(absorbed_power=PhysicalParameters().absorbed_power*2)
    if change=='history':history=PowerHistory('linear',1.)
    if change=='angle':angle=30.
    if change=='times':times=[0.,.001,.0025,.01]
    if change=='source':summary['config']['solver_source_sha256']['src/thermal_opd/fem.py']='0'*64
    write_summary(out,summary)
    with pytest.raises(ValueError,match=match):
        loader()(out,p,angle,history,times,repository_root=ROOT)


def test_formal_saved_case_rechecks_certificate_before_inspecting_fields(tmp_path,monkeypatch):
    p=PhysicalParameters()
    summary=dict(physical_parameters_SI=asdict(p),h_eff_W_per_m2_K=p.h_eff,sanity_tolerance=1e-7,
        sanity_pass=True,transient_truncation_established=False,fem_convergence_established=True,
        fem_convergence_directory='certificate',angle_deg=0.,history=asdict(PowerHistory()),times_s=[0.,.001,.0025])
    def failed_review(*args,**kwargs):
        raise ValueError('Independent certificate review caught corruption')
    monkeypatch.setattr(validation,'load_transient_fem_certificate',failed_review)
    with pytest.raises(ValueError,match='Independent certificate review'):
        validation.validate_transient_fem(tmp_path,summary)


def test_formal_entry_and_saved_review_reuse_original_nodes_and_detect_sample_corruption(tmp_path,monkeypatch):
    source,_,out,summary=certificate(tmp_path)
    original=validation.validate_saved_transient_fem_convergence
    def reviewed(path,stored):
        # Planted trust boundary: actual physical recheck still runs, while the
        # fixture intentionally lacks expensive refinement references.
        value=original(path,{**stored,'fem_convergence_established':False})
        value['fem_convergence_established']=True
        value['verification'].update(status='passed',fem_convergence_established=True)
        return value
    monkeypatch.setattr(validation,'validate_saved_transient_fem_convergence',reviewed)
    summary['fem_convergence_established']=True;write_summary(out,summary)
    entry_path=tmp_path/'scripts/cases/run_transient_case.py';entry_path.parent.mkdir(parents=True)
    entry_path.write_text((ROOT/'scripts/cases/run_transient_case.py').read_text(encoding='utf-8'),encoding='utf-8')
    for name in SOURCES:
        target=tmp_path/'src/thermal_opd'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/'src/thermal_opd'/name).read_bytes())
    spec=importlib.util.spec_from_file_location('_transient_reuse_case_full',entry_path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    entry.modal_settings=entry.ModalResolution(0,2,12,8,8)
    entry.reference_settings=entry.ModalResolution(0,3,16,10,10)
    entry.axial_counts=(2,3);entry.formal_observation_times_s=(0.,.001,.0025)
    entry.observation_times_s=(0.,.001,.0025)
    entry.temperature_r_points,entry.temperature_phi_points,entry.opd_xy_points=5,8,7
    entry.formal_temperature_r_points,entry.formal_temperature_phi_points,entry.formal_opd_xy_points=5,8,7
    output=tmp_path/'outputs/fem_comparison/formal_case'
    monkeypatch.setattr(sys,'argv',['run_transient_case.py','--mode','full','--angle','0',
        '--fem-convergence',str(out),'--output-dir',str(output)])
    from thermal_opd import fem
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Formal entry reran FEM'))
    entry.main()
    assert not (output/'data/fem_run0.npz').exists()
    from thermal_opd.run_io import read_json,resolve_run_path
    stored=read_json(output/'summary.json')
    assert stored['fem_convergence_established'] is True
    assert stored['transient_truncation_established'] is False
    assert stored['fem_runs'][0]['changes_vs_group_reference']==[None]*len(stored['times_s'])
    assert resolve_run_path(output,stored['fem_runs'][0]['data_file'])==source/'data/fem_run0.npz'
    assert validation.validate_saved_transient_run(output)['status']=='PASS'
    with np.load(output/'data/fields.npz') as raw:fields={key:raw[key] for key in raw.files}
    # Change both fields and stored cross-method metrics, so only reconstruction
    # from the original P2 nodes detects the counterfeit sampled field.
    fields['temperature_fem_K'][0,1,0,0]+=1.
    tf=fields['temperature_fem_K'][0];sa=fields['temperature_sa_K'][...,fields['temperature_fem_common_mask']]
    weights=fields['area_weights_m2'][fields['temperature_fem_common_mask']]
    stored['fem_runs'][0]['temperature_absolute_max_difference_K']=np.max(abs(tf-sa),axis=-1).tolist()
    stored['fem_runs'][0]['temperature_area_rms_difference_K']=np.sqrt(np.sum((tf-sa)**2*weights,axis=-1)/weights.sum()).tolist()
    for j in range(1,len(stored['times_s'])):
        stored['fem_runs'][0]['errors_vs_sa'][j]=validation.field_errors(tf[j],sa[j],weights,
            fields['opd_fem_m'][0,j],fields['opd_sa_fem_common_m'][j],temperature_normalization='global')
    np.savez_compressed(output/'data/fields.npz',**fields);write_summary(output,stored)
    with pytest.raises(ValueError,match='sampled temperature'):
        validation.validate_saved_transient_run(output)


def test_formal_report_preserves_reread_convergence_statuses(tmp_path):
    report=dict(status='not_run',checks=[
        dict(name='theta0_xy_convergence',status='passed'),dict(name='theta0_z_convergence',status='passed'),
        dict(name='theta0_joint_convergence',status='failed'),dict(name='theta0_time_convergence',status='not_run')])
    summary=dict(kind='transient_modal_fem_study',angle_deg=0.,fem_convergence_established=True,
        transient_truncation_established=False)
    value=validation.transient_saved_validation_report(tmp_path,summary,fem_validation=report)
    checks={check['name']:check for check in value['checks']}
    assert checks['fem_space_convergence']['status']=='failed'
    assert checks['fem_time_convergence']['status']=='not_run'
    assert checks['cross_method_comparison']['status']=='not_run'


def test_candidate_cannot_change_between_review_and_restoration(tmp_path,monkeypatch):
    source,_,out,summary=certificate(tmp_path)
    summary['fem_convergence_established']=True;write_summary(out,summary)
    report=read_json(out/'verification.json');report.update(status='passed',fem_convergence_established=True)
    def changed_after_review(*args):
        path=source/'data/fem_run0.npz'
        with np.load(path) as f:values={key:f[key] for key in f.files}
        values['temperature_K'][-1,0,0]+=1.
        np.savez_compressed(path,**values)
        return dict(verification=report,fem_convergence_established=True)
    monkeypatch.setattr(validation,'validate_saved_transient_fem_convergence',changed_after_review)
    with pytest.raises(ValueError,match='evidence'):
        loader()(out,PhysicalParameters(),0.,PowerHistory(),[0.,.001,.0025],repository_root=ROOT)


def test_sa_selection_cannot_certify_an_extra_positive_time_even_at_1ms(tmp_path,monkeypatch):
    p=PhysicalParameters();modal=asdict(ModalResolution(0,2,12,8,8))
    selected=dict(transient_truncation_established=True,times_s=[.0025],selected_settings=dict(modal=modal,J=3))
    summary=dict(physical_parameters_SI=asdict(p),h_eff_W_per_m2_K=p.h_eff,sanity_tolerance=1e-7,
        sanity_pass=True,transient_truncation_established=True,fem_convergence_established=False,
        selection_directory='selection',angle_deg=0.,history=asdict(PowerHistory()),times_s=[0.,.001,.0025],
        modal_settings=modal,axial_counts=[3])
    monkeypatch.setattr(validation,'load_transient_selection',lambda *a:selected)
    with pytest.raises(ValueError,match='time|scope'):
        validation.validate_transient_fem(tmp_path,summary)
