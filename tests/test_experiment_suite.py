"""Batch orchestration must preserve failures and use this batch's selections."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def suite():
    spec=importlib.util.spec_from_file_location('_suite_test',ROOT/'run_all.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_plan_has_valid_dependency_order_and_all_scientific_entries(suite):
    stages=suite.build_plan('full',[0.,30.,60.])
    completed=set()
    for stage in stages:
        assert set(stage['needs'])<=completed
        completed.add(stage['name'])
    assert len(completed)==len(stages)
    assert {s['experiment'] for s in stages if s['experiment']}=={
        'convergence','transient-convergence','steady','transient','fem-comparison',
        'semi-validation','fem-validation','transient-fem-convergence','boundaries','source','azimuthal',
        'temperature','opd','sweeps','replot'}
    assert sum(s['name'].startswith('transient_fields_') for s in stages)==3
    assert sum(s['name'].startswith('transient_fem_theta') for s in stages)==3
    gate=next(s for s in stages if s['name']=='transient_fem_convergence')
    for stage in stages:
        if stage['name'].startswith('transient_fem_theta'):
            assert gate['name'] in stage['needs']
            assert stage['args'][stage['args'].index('--fem-convergence')+1]=='@transient_fem_convergence'
    assert not any('baseline' in str(s) for s in stages)


def test_smoke_plan_does_not_launch_selection_full_fem_or_dense_fields(suite):
    stages=suite.build_plan('smoke',[60.])
    assert {s['experiment'] for s in stages if s['experiment']}=={'fem-validation','transient','boundaries'}
    assert not any('full' in s['args'] or 'fields' in s['args'] for s in stages)


@pytest.mark.parametrize('name',('azimuthal','source','temperature','opd','sweeps'))
def test_diagnostic_scripts_accept_the_batch_output_category(suite,name,tmp_path,monkeypatch):
    # Exercise each real entry up to directory allocation, then stop before
    # any physical calculation. Mocked batch executors do not check this.
    from thermal_opd.experiments import EXPERIMENTS
    from thermal_opd.run_io import create_run
    stage=next(s for s in suite.build_plan('full',[60.]) if s['name']==name)
    path=ROOT/'scripts'/EXPERIMENTS[name][1]
    spec=importlib.util.spec_from_file_location('_batch_category_'+name,path)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    output=tmp_path/'outputs'/stage['category']/'category_test'
    class Allocated(Exception):
        pass
    def allocate_only(root,prefix,directory):
        # Substitute only the sandbox root; use the script's actual prefix,
        # CLI output path and production path-validation implementation.
        result=create_run(tmp_path,prefix,directory)
        assert result==output
        raise Allocated
    monkeypatch.setattr(module,'create_run',allocate_only)
    monkeypatch.setattr(sys,'argv',[str(path),'--output-dir',str(output)])
    with pytest.raises(Allocated):
        module.main()
    assert (output/'run.json').is_file()


def test_plan_only_does_not_create_outputs_or_launch_commands(suite,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    monkeypatch.setattr(suite,'_run_command',lambda *a,**kw:pytest.fail('plan started a command'))
    assert suite.main(['--mode','full','--plan'])==0
    assert not (tmp_path/'outputs').exists()
    assert json.loads(capsys.readouterr().out)['mode']=='full'


def fake_project(suite,tmp_path,monkeypatch):
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    (tmp_path/'run_all.py').write_bytes((ROOT/'run_all.py').read_bytes())
    monkeypatch.setattr(suite,'provenance',lambda *a:dict(git_revision='test',working_tree_dirty=False))
    monkeypatch.setattr(suite,'_check_steady_selection',lambda *a:None)


def fake_executor(suite,calls,*,science='not_run',fail_name=None):
    def run(command,log_file,*,append=False):
        calls.append(command)
        log_file.parent.mkdir(parents=True,exist_ok=True)
        with log_file.open('a' if append else 'w',encoding='utf-8') as stream:
            stream.write(json.dumps(command)+'\n')
        if '-m' in command:return 0
        script=Path(command[2]).name
        if fail_name and script==fail_name:return 1
        if script=='validate_run.py':
            path=Path(command[command.index('--results')+1])
            (path/'check_report.json').write_text(json.dumps(dict(status='PASS',verification={'status':science})))
            return 0
        if '--output-dir' in command:
            path=Path(command[command.index('--output-dir')+1]);path.mkdir(parents=True)
            (path/'data').mkdir();(path/'figures').mkdir()
            summary=dict(status='COMPLETE',kind='mock',selected_settings=dict(M=6,N=8,projection_r=400,projection_phi=32,ray_points=16))
            (path/'summary.json').write_text(json.dumps(summary))
            (path/'verification.json').write_text(json.dumps(dict(status=science)))
            (path/'data/fields.npz').write_bytes(b'stored fields')
        return 0
    return run


def test_smoke_reports_execution_success_without_scientific_certification(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch)
    calls=[]
    code,path=suite.run_suite('smoke',[60.],runner=fake_executor(suite,calls))
    report=json.loads((path/'summary.json').read_text())
    assert code==0 and report['status']=='COMPLETE'
    assert report['scientific_status']=='not_run' and report['paper_ready'] is False
    assert all('legacy' not in str(command) for command in calls)


def test_failed_scientific_evidence_does_not_stop_full_data_collection_or_become_passed(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch)
    calls=[]
    code,path=suite.run_suite('full',[60.],runner=fake_executor(suite,calls,science='failed'))
    report=json.loads((path/'summary.json').read_text())
    assert code==2 and report['status']=='COMPLETE'
    assert report['scientific_status']=='failed' and report['paper_ready'] is False
    assert all(s['status']=='COMPLETE' for s in report['stages'])
    selections={s['name']:path/s['run_directory'] for s in report['stages'] if s['name'].endswith('_selection')}
    for command in calls:
        if '--selection' in command:
            selected=Path(command[command.index('--selection')+1])
            assert selected.resolve() in {p.resolve() for p in selections.values()}


def test_resume_skips_completed_stages_and_keeps_artifacts(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch)
    calls=[]
    _,path=suite.run_suite('smoke',[60.],runner=fake_executor(suite,calls))
    calls.clear()
    code,same=suite.run_suite('smoke',[60.],resume=path,runner=fake_executor(suite,calls))
    assert code==0 and same==path and not calls


def test_explicit_fem_reuse_index_is_forwarded_and_cannot_change_on_resume(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch);calls=[]
    index=tmp_path/'reuse.json';index.write_text('{"cases":[]}')
    _,out=suite.run_suite('full',[0.],runner=fake_executor(suite,calls),fem_reuse_index=index)
    command=next(c for c in calls if any('check_transient_fem_convergence.py' in str(v) for v in c))
    assert command[command.index('--reuse-index')+1]==str(index.resolve())
    index.write_text('{"cases":[]}\n')
    with pytest.raises(ValueError,match='reuse index changed'):
        suite.run_suite('full',[0.],resume=out,runner=fake_executor(suite,calls),fem_reuse_index=index)


def test_resume_rejects_changed_code_or_corrupt_completed_data(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch)
    _,path=suite.run_suite('smoke',[60.],runner=fake_executor(suite,[]))
    report=json.loads((path/'summary.json').read_text())
    artifact=next(s for s in report['stages'] if s.get('run_directory'))
    (path/artifact['run_directory']/'data/fields.npz').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='changed|corrupt'):
        suite.run_suite('smoke',[60.],resume=path,runner=fake_executor(suite,[]))
    (tmp_path/'run_all.py').write_text('modified code')
    with pytest.raises(ValueError,match='code'):
        suite.run_suite('smoke',[60.],resume=path,runner=fake_executor(suite,[]))


def test_failed_process_stops_dependents_and_resume_uses_new_output(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch)
    code,path=suite.run_suite('smoke',[60.],runner=fake_executor(suite,[],fail_name='run_transient_case.py'))
    before=json.loads((path/'summary.json').read_text())
    assert code==1 and before['status']=='FAILED'
    failed=next(s for s in before['stages'] if s['status']=='FAILED')
    old=failed['run_directory']
    calls=[]
    code,_=suite.run_suite('smoke',[60.],resume=path,runner=fake_executor(suite,calls))
    after=json.loads((path/'summary.json').read_text())
    retried=next(s for s in after['stages'] if s['name']==failed['name'])
    assert code==0 and retried['run_directory']!=old
    assert len(retried['attempts'])==2
    assert all(Path(c[2]).name!='verify_fem.py' for c in calls)


def test_steady_case_must_match_new_selected_parameters(suite,tmp_path,monkeypatch):
    from thermal_opd.experiments import experiment_configuration
    config=experiment_configuration('steady')['defaults']
    settings=dict(M=config['max_azimuthal_order'],N=config['num_radial_modes'],projection_r=config['num_projection_r'],
                  projection_phi=config['num_projection_phi'],ray_points=config['num_opd_ray_points'])
    selected=tmp_path/'summary.json'
    selected.write_text(json.dumps(dict(selected_settings=settings,selected_figure_xy=121)))
    suite._check_steady_selection(tmp_path)
    settings['N']+=1
    selected.write_text(json.dumps(dict(selected_settings=settings,selected_figure_xy=121)))
    with pytest.raises(ValueError,match='steady|settings'):
        suite._check_steady_selection(tmp_path)


def test_failed_fem_stage_resume_uses_recorded_control_directory(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch);calls=[]
    base=fake_executor(suite,calls)
    def fail_fem(command,log,**kwargs):
        if any('check_transient_fem_convergence.py' in str(v) for v in command):
            calls.append(command)
            control=Path(command[command.index('--control-dir')+1])
            control.mkdir(parents=True);(control/'summary.json').write_text('{}')
            return 1
        return base(command,log,**kwargs)
    code,out=suite.run_suite('full',[0.],runner=fail_fem)
    assert code==1
    first=next(c for c in calls if any('check_transient_fem_convergence.py' in str(v) for v in c))
    previous=first[first.index('--control-dir')+1]
    calls.clear();suite.run_suite('full',[0.],resume=out,runner=base)
    command=next(c for c in calls if any('check_transient_fem_convergence.py' in str(v) for v in c))
    assert command[command.index('--resume-control')+1]==previous
    assert command[command.index('--control-dir')+1]!=previous


def test_fem_resume_keeps_prior_control_when_latest_attempt_did_not_create_one(suite,tmp_path,monkeypatch):
    fake_project(suite,tmp_path,monkeypatch);calls=[];base=fake_executor(suite,calls)
    attempts=[]
    def fail_fem(command,log,**kwargs):
        if any('check_transient_fem_convergence.py' in str(v) for v in command):
            calls.append(command);attempts.append(command)
            if len(attempts)==1:
                control=Path(command[command.index('--control-dir')+1]);control.mkdir(parents=True)
                (control/'summary.json').write_text('{}')
            return 1
        return base(command,log,**kwargs)
    _,out=suite.run_suite('full',[0.],runner=fail_fem)
    suite.run_suite('full',[0.],resume=out,runner=fail_fem)
    suite.run_suite('full',[0.],resume=out,runner=base)
    last=next(c for c in reversed(calls) if any('check_transient_fem_convergence.py' in str(v) for v in c))
    assert last[last.index('--resume-control')+1]==attempts[0][attempts[0].index('--control-dir')+1]


def continuation_fixture(suite,tmp_path,monkeypatch):
    import hashlib
    fake_project(suite,tmp_path,monkeypatch)
    _,parent=suite.run_suite('full',[0.],runner=fake_executor(suite,[],science='failed',
        fail_name='check_transient_fem_convergence.py'))
    old=json.loads((parent/'summary.json').read_text())
    snapshot=tmp_path/'old_source';snapshot.mkdir()
    (snapshot/'run_all.py').write_bytes((tmp_path/'run_all.py').read_bytes())
    with (tmp_path/'run_all.py').open('a') as stream:stream.write('\n# explicit code continuation\n')
    current=suite._code_hashes();before=old['code_sha256']
    test_report=tmp_path/'regression.xml'
    test_report.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    evidence=dict(kind='experiment_suite_code_continuation',version=1,
        purpose='same_formula_residual_accumulation_precision',
        parent_suite_sha256=suite.file_sha256(parent/'summary.json'),
        parent_code_manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        parent_source_snapshot=str(snapshot),
        changes=[dict(path=key,before_sha256=before.get(key,{}).get('sha256'),
            after_sha256=current.get(key,{}).get('sha256'),reason='Explicit tested code version change')
            for key in sorted(before.keys()|current.keys()) if before.get(key)!=current.get(key)],
        independent_test_reports=[dict(data_file=str(test_report),sha256=suite.file_sha256(test_report),status='passed')],
        numerical_policy=dict(residual_formula_unchanged=True,residual_tolerance_unchanged=True,saved_temperature_dtype='float64'))
    path=tmp_path/'continuation.json';path.write_text(json.dumps(evidence))
    return parent,old,path,evidence


@pytest.mark.parametrize('legacy_parent',(False,True))
def test_explicit_continuation_inherits_only_complete_stages_and_keeps_old_suite_immutable(suite,tmp_path,monkeypatch,legacy_parent):
    parent,old,evidence_path,evidence=continuation_fixture(suite,tmp_path,monkeypatch)
    if legacy_parent:
        old.pop('plan');(parent/'summary.json').write_text(json.dumps(old))
        evidence['parent_suite_sha256']=suite.file_sha256(parent/'summary.json')
        evidence_path.write_text(json.dumps(evidence))
    original=(parent/'summary.json').read_bytes();calls=[]
    code,new=suite.run_suite('full',[0.],continue_from=parent,continuation_evidence=evidence_path,
        runner=fake_executor(suite,calls,science='not_run'))
    assert new!=parent and code==2
    assert (parent/'summary.json').read_bytes()==original
    report=json.loads((new/'summary.json').read_text())
    assert report['code_sha256']==suite._code_hashes()
    assert report['continuation']['parent_code_sha256']==old['code_sha256']
    assert report['continuation']['parent_summary_sha256']==evidence['parent_suite_sha256']
    before={row['name']:row for row in old['stages']}
    for row in report['stages']:
        if before[row['name']]['status']=='COMPLETE':
            if row.get('run_directory'):assert row['artifacts']==before[row['name']]['artifacts']
            assert row['scientific_status']==before[row['name']]['scientific_status']
            assert row['inherited_from_stage']==row['name']
            assert row['inherited_code_manifest_sha256']==evidence['parent_code_manifest_sha256']
        else:assert 'inherited_from_stage' not in row
    assert all(Path(c[2]).name not in ('verify_fem.py','select_transient_truncation.py') for c in calls)
    fem=next(c for c in calls if any('check_transient_fem_convergence.py' in str(v) for v in c))
    assert '--resume-control' not in fem
    sa=next(c for c in calls if '--fem-convergence' in c)
    selected=Path(sa[sa.index('--selection')+1])
    assert selected==suite.resolve_run_path(parent,before['transient_selection']['run_directory'])
    assert report['scientific_status']=='failed'
    calls.clear()
    suite.run_suite('full',[0.],resume=new,runner=fake_executor(suite,calls))
    assert calls==[]


@pytest.mark.parametrize('corrupt',('missing_change','wrong_before','wrong_after','parent_manifest','parent_summary',
    'snapshot','report_hash','failed_report','plan','artifact'))
def test_continuation_rejects_missing_or_corrupt_version_evidence_before_creating_batch(suite,tmp_path,monkeypatch,corrupt):
    parent,old,path,evidence=continuation_fixture(suite,tmp_path,monkeypatch)
    if corrupt=='missing_change':evidence['changes']=[]
    if corrupt=='wrong_before':evidence['changes'][0]['before_sha256']='0'*64
    if corrupt=='wrong_after':evidence['changes'][0]['after_sha256']='0'*64
    if corrupt=='parent_manifest':evidence['parent_code_manifest_sha256']='0'*64
    if corrupt=='parent_summary':evidence['parent_suite_sha256']='0'*64
    if corrupt=='snapshot':(Path(evidence['parent_source_snapshot'])/'run_all.py').write_text('changed snapshot')
    if corrupt=='report_hash':evidence['independent_test_reports'][0]['sha256']='0'*64
    if corrupt=='failed_report':
        file=Path(evidence['independent_test_reports'][0]['data_file'])
        file.write_text('<testsuites><testsuite tests="3" failures="1" errors="0"/></testsuites>')
        evidence['independent_test_reports'][0]['sha256']=suite.file_sha256(file)
    if corrupt=='plan':
        original=suite.build_plan;monkeypatch.setattr(suite,'build_plan',lambda *a:original(*a)[:-1])
    if corrupt=='artifact':
        row=next(row for row in old['stages'] if row.get('run_directory'))
        (suite.resolve_run_path(parent,row['run_directory'])/'data/fields.npz').write_bytes(b'corrupted')
    path.write_text(json.dumps(evidence));before=set((tmp_path/'outputs/experiment_suite').iterdir())
    with pytest.raises(ValueError,match='continuation|Continuation|parent|Parent|snapshot|Snapshot|report|Report|plan|Plan|changed|corrupt'):
        suite.run_suite('full',[0.],continue_from=parent,continuation_evidence=path,
            runner=lambda *a,**k:pytest.fail('Invalid continuation launched a command'))
    assert set((tmp_path/'outputs/experiment_suite').iterdir())==before


@pytest.mark.parametrize('corrupt',('artifact','inherited_log','evidence','test_report'))
def test_continuation_keeps_strict_resume_hash_checks(suite,tmp_path,monkeypatch,corrupt):
    parent,old,path,evidence=continuation_fixture(suite,tmp_path,monkeypatch)
    _,out=suite.run_suite('full',[0.],continue_from=parent,continuation_evidence=path,
        runner=fake_executor(suite,[]))
    report=json.loads((out/'summary.json').read_text())
    if corrupt=='artifact':
        row=next(r for r in report['stages'] if r.get('inherited_from_stage') and r.get('run_directory'))
        (suite.resolve_run_path(out,row['run_directory'])/'data/fields.npz').write_bytes(b'changed')
    if corrupt=='inherited_log':
        row=next(r for r in report['stages'] if r.get('inherited_from_stage'))
        suite.resolve_run_path(out,row['log_file']).write_text('changed inherited log')
    if corrupt=='evidence':path.write_text(path.read_text()+'\n')
    if corrupt=='test_report':Path(evidence['independent_test_reports'][0]['data_file']).write_text('changed report')
    with pytest.raises(ValueError,match='changed|corrupt'):
        suite.run_suite('full',[0.],resume=out,runner=lambda *a,**k:pytest.fail('Bad resume launched a command'))


@pytest.mark.parametrize('arguments',({'continue_from':'parent'},{'continuation_evidence':'evidence'},
    {'resume':'existing','continue_from':'parent','continuation_evidence':'evidence'}))
def test_continuation_flags_are_explicit_and_exclusive(suite,tmp_path,monkeypatch,arguments):
    fake_project(suite,tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='[Cc]ontinuation|resume'):
        suite.run_suite('smoke',[60.],**arguments,runner=lambda *a,**k:pytest.fail('Invalid flags ran a command'))
    assert not (tmp_path/'outputs').exists()


def test_continuation_rejects_running_parent_before_creating_batch(suite,tmp_path,monkeypatch):
    parent,old,path,evidence=continuation_fixture(suite,tmp_path,monkeypatch)
    old['status']='RUNNING';(parent/'summary.json').write_text(json.dumps(old))
    evidence['parent_suite_sha256']=suite.file_sha256(parent/'summary.json')
    path.write_text(json.dumps(evidence))
    before=set((tmp_path/'outputs/experiment_suite').iterdir())
    with pytest.raises(ValueError,match='parent.*status|parent.*RUNNING'):
        suite.run_suite('full',[0.],continue_from=parent,continuation_evidence=path,
            runner=lambda *a,**k:pytest.fail('Running parent launched a duplicate stage'))
    assert set((tmp_path/'outputs/experiment_suite').iterdir())==before


def test_explicit_modal_sum_continuation_preserves_the_same_strict_evidence_checks(suite,tmp_path,monkeypatch):
    parent,old,path,evidence=continuation_fixture(suite,tmp_path,monkeypatch)
    evidence['purpose']='same_formula_modal_sum_grouping'
    path.write_text(json.dumps(evidence))
    accepted,_=suite._continuation_parent(parent,path,'full',[0.],suite.build_plan('full',[0.]),suite._code_hashes())
    assert accepted['code_sha256']==old['code_sha256']
    evidence['changes'][0]['after_sha256']='0'*64
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='changes.*hash|before/after'):
        suite._continuation_parent(parent,path,'full',[0.],suite.build_plan('full',[0.]),suite._code_hashes())
