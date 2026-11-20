"""Real FEM workflow controls: isolated roles, saved evidence, failure preservation."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
from thermal_opd.run_io import create_run,read_json,data_directory


@pytest.fixture
def workflow():
    path=Path(__file__).resolve().parents[1]/'scripts/convergence/check_transient_fem_convergence.py'
    spec=importlib.util.spec_from_file_location('_fem_workflow_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def test_full_plan_is_true_quadratic_isolated_and_joint_refinement(workflow):
    rows=workflow.make_plan('full',[0.,30.,60.])
    assert len(rows)==15
    for angle in (0.,30.,60.):
        by={r['role']:r for r in rows if r['angle_deg']==angle}
        base=by['candidate']
        assert base['field_degree']==2 and base['mirror_reduction'] is True
        assert base['times_s']==[0.,.001,.0025,.01,.05,.1,.625,2.5,20.,50.,75.]
        assert sum(base['interval_counts'])==2176
        assert (base['xy'],base['z_resolution'])==(4,64)
        assert (by['xy_reference']['xy'],by['xy_reference']['z_resolution'])==(5,64)
        assert (by['z_reference']['xy'],by['z_reference']['z_resolution'])==(4,128)
        assert by['time_reference']['interval_counts']==[2*n for n in base['interval_counts']]
        assert (by['joint_reference']['xy'],by['joint_reference']['z_resolution'])==(5,128)


def test_real_smoke_worker_retains_one_ms_real_p2_nodes_and_partial_progress(workflow,tmp_path):
    spec=workflow.make_plan('smoke',[0.])[0];out=create_run(tmp_path,'fem_comparison')
    workflow.solve_case(tmp_path,spec,out)
    summary=read_json(out/'summary.json')
    assert summary['status']=='COMPLETE' and summary['fem_convergence_established'] is False
    assert summary['solver_source_sha256']==workflow.solver_hashes()
    with np.load(out/'data/fem_run0.npz') as raw:
        assert raw['field_degree'].item()==2
        assert raw['temperature_K'].shape[2]==raw['xy_dof_coordinates_m'].shape[1]
        assert raw['temperature_K'].shape[2]>raw['disk_nodes_xy_m'].shape[1]
        assert .001 in raw['times_s']
        np.testing.assert_array_equal(raw['temperature_K'][0],0.)
    progress=read_json(out/'data/observation_progress.json')
    assert progress['completed_observation_times_s']==spec['times_s']
    assert progress['can_resume_time_integration'] is False


def test_interrupted_save_keeps_real_partial_but_never_complete(workflow,tmp_path,monkeypatch):
    spec=workflow.make_plan('smoke',[0.])[0];out=create_run(tmp_path,'fem_comparison')
    original=workflow.save_observation
    def interrupt(*args,**kwargs):
        original(*args,**kwargs)
        if args[1].diagnostics['time_s']>0:raise RuntimeError('planted interrupted output')
    monkeypatch.setattr(workflow,'save_observation',interrupt)
    with pytest.raises(RuntimeError,match='planted'):
        workflow.solve_case(tmp_path,spec,out)
    assert read_json(out/'summary.json')['status']=='FAILED'
    progress=read_json(out/'data/observation_progress.json')
    assert .001 in progress['completed_observation_times_s']
    assert not (out/'data/fem_run0.npz').exists()


def test_plan_only_never_starts_solver_or_allocates_results(workflow,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(workflow,'repository_root',tmp_path)
    monkeypatch.setattr(workflow,'solve_case',lambda *a,**k:pytest.fail('plan solved FEM'))
    assert workflow.main(['--mode','full','--plan'])==0
    assert not (tmp_path/'outputs').exists()


def test_reuse_rejects_changed_nodes_configuration_or_solver_source(workflow,tmp_path):
    spec=workflow.make_plan('smoke',[0.])[0];out=create_run(tmp_path,'fem_comparison')
    workflow.solve_case(tmp_path,spec,out)
    workflow.reuse_case(out,spec)
    with pytest.raises(ValueError,match='configuration'):
        workflow.reuse_case(out,{**spec,'angle_deg':30.})
    file=out/'data/fem_run0.npz'
    with file.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        workflow.reuse_case(out,spec)


def test_explicit_probe_bridge_keeps_birth_hashes_and_never_copies_nodes(workflow,tmp_path,monkeypatch):
    import shutil
    from thermal_opd import fem
    from thermal_opd.run_io import write_summary,file_sha256,resolve_run_path
    from thermal_opd.transient_validation import _transient_fem_case_header
    spec=workflow.make_plan('smoke',[0.])[0];original=create_run(tmp_path,'fem_comparison')
    produced=workflow.solve_case(tmp_path,spec,original)
    probe=create_run(tmp_path,'fem_order_probe');file=probe/'data/nodes.npz'
    with np.load(original/'data/fem_run0.npz') as raw:values={k:raw[k] for k in raw.files}
    for old,new in (('disk_nodes_xy_m','xy_vertices_m'),('disk_triangles','triangles'),('axial_nodes_z_m','z_dofs_m')):
        values[new]=values.pop(old)
    np.savez_compressed(file,**values)
    write_summary(probe,dict(kind='fem_order_probe_case',status='COMPLETE',physical_passed=True,
        config=dict(degree=2,xy=spec['xy'],zres=spec['z_resolution'],angle_deg=0.,time_factor=1),
        physical_parameters_SI=produced['physical_parameters_SI'],nodes_sha256=file_sha256(file),
        dofs=produced['cases'][0]['diagnostics']['degrees_of_freedom'],
        gaussian_load_quadrature_orders=[10,14],
        gaussian_load_quadrature_relative_l2=produced['gaussian_load_quadrature_relative_l2']))
    snapshot=tmp_path/'birth';(snapshot/'src/thermal_opd').mkdir(parents=True);(snapshot/'tmp').mkdir()
    for name in workflow.solver_hashes():shutil.copy2(workflow.repository_root/name,snapshot/name)
    (snapshot/'tmp/p2_fem_spike.py').write_text('# synthetic controlled birth')
    signature=(file_sha256(snapshot/'src/thermal_opd/fem.py'),file_sha256(snapshot/'tmp/p2_fem_spike.py'))
    monkeypatch.setattr(workflow,'known_probe_signatures',{signature})
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Bridge solved FEM'))
    reference=workflow.bridge_probe_case(tmp_path,dict(source_run=str(probe),source_snapshot=str(snapshot)),spec,tmp_path)
    canonical=Path(reference['source_run']);header=read_json(canonical/'summary.json')
    assert header['solver_source_sha256']['src/thermal_opd/fem.py']==signature[0]
    assert header['source_spike_sha256']==signature[1]
    assert resolve_run_path(canonical,header['cases'][0]['data_file'])==file
    assert not (canonical/'data/fem_run0.npz').exists()
    _transient_fem_case_header(canonical,dict(source_run='.',case_index=0))
    assert header['birth_evidence_files']
    control=create_run(tmp_path,'fem_convergence')
    write_summary(control,dict(kind='transient_fem_workflow',status='FAILED',plan=workflow.make_plan('smoke',[0.]),
        solver_source_sha256=workflow.solver_hashes(),reuse_index_sha256=None,
        cases=[dict(angle_deg=0.,candidate=reference)]))
    restored=workflow.recovery_references(control,workflow.make_plan('smoke',[0.]),None)
    assert Path(restored[(0.,'candidate')]['source_run'])==canonical
    birth=next(iter(header['birth_evidence_files']))
    (canonical/birth).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='birth|hash'):
        _transient_fem_case_header(canonical,dict(source_run='.',case_index=0))


def test_recovery_retains_only_completed_formal_references(workflow,tmp_path):
    from thermal_opd.run_io import write_summary,resolve_run_path
    plan=workflow.make_plan('smoke',[0.]);spec=plan[0]
    source=create_run(tmp_path,'fem_comparison');workflow.solve_case(tmp_path,spec,source)
    control=create_run(tmp_path,'fem_convergence')
    state=dict(kind='transient_fem_workflow',status='FAILED',mode='smoke',plan=plan,
        solver_source_sha256=workflow.solver_hashes(),reuse_index_sha256=None,
        cases=[dict(angle_deg=0.,candidate=dict(source_run=str(source),case_index=0))],
        runs=[dict(config=plan[1],status='FAILED',source_run='never_reuse_partial')])
    write_summary(control,state)
    recovered=workflow.recovery_references(control,plan,None)
    assert set(recovered)=={(0.,'candidate')}
    assert resolve_run_path(control,recovered[(0.,'candidate')]['source_run'])==source
    state['cases']=[]
    state['runs']=[dict(config=spec,status='COMPLETE',source_run=str(source)),
        dict(config={**plan[1],'times_s':[0.,.0001],'interval_counts':[4]},status='COMPLETE',source_run='startup_only')]
    write_summary(control,state)
    assert set(workflow.recovery_references(control,plan,None))=={(0.,'candidate')}
    with pytest.raises(ValueError,match='reuse index'):
        workflow.recovery_references(control,plan,'changed')
    state['status']='RUNNING';write_summary(control,state)
    with pytest.raises(ValueError,match='RUNNING|stopped'):
        workflow.recovery_references(control,plan,None)
    state['status']='FAILED';write_summary(control,state)
    with pytest.raises(ValueError,match='plan|configuration'):
        workflow.recovery_references(control,[{**spec,'xy':spec['xy']+1}],None)
    state['solver_source_sha256']={};write_summary(control,state)
    with pytest.raises(ValueError,match='source'):
        workflow.recovery_references(control,plan,None)


def test_recovery_rejects_failed_previously_completed_case(workflow,tmp_path):
    from thermal_opd.run_io import write_summary
    plan=workflow.make_plan('smoke',[0.]);source=create_run(tmp_path,'fem_comparison')
    workflow.solve_case(tmp_path,plan[0],source)
    stored=read_json(source/'summary.json');stored['status']='FAILED';write_summary(source,stored)
    control=create_run(tmp_path,'fem_convergence')
    write_summary(control,dict(kind='transient_fem_workflow',status='FAILED',mode='smoke',plan=plan,
        solver_source_sha256=workflow.solver_hashes(),reuse_index_sha256=None,
        cases=[dict(angle_deg=0.,candidate=dict(source_run=str(source),case_index=0))]))
    with pytest.raises(ValueError,match='status|configuration'):
        workflow.recovery_references(control,plan,None)


def test_duplicate_reuse_roles_are_rejected_before_allocating_outputs(workflow,tmp_path,monkeypatch):
    from thermal_opd.run_io import write_json
    index=tmp_path/'reuse.json';entry=dict(angle_deg=0.,role='candidate',source_run='irrelevant')
    write_json(index,dict(cases=[entry,entry]))
    monkeypatch.setattr(workflow,'create_run',lambda *a,**k:pytest.fail('Invalid index allocated a run'))
    with pytest.raises(ValueError,match='Duplicate'):
        workflow.main(['--mode','smoke','--reuse-index',str(index)])


def test_resumed_workflow_never_launches_completed_case_and_keeps_reference(workflow,tmp_path,monkeypatch):
    from thermal_opd.run_io import write_summary,resolve_run_path
    plan=workflow.make_plan('smoke',[0.]);source=create_run(tmp_path,'fem_comparison')
    workflow.solve_case(tmp_path,plan[0],source)
    old=create_run(tmp_path,'fem_convergence')
    write_summary(old,dict(kind='transient_fem_workflow',status='FAILED',mode='smoke',plan=plan,
        solver_source_sha256=workflow.solver_hashes(),reuse_index_sha256=None,
        cases=[dict(angle_deg=0.,candidate=dict(source_run=str(source),case_index=0))]))
    attempted=[]
    def interrupted(root,control,state,spec,budget):
        attempted.append(spec['role']);raise RuntimeError('planted next-case failure')
    monkeypatch.setattr(workflow,'repository_root',tmp_path)
    monkeypatch.setattr(workflow,'launch_case',interrupted)
    monkeypatch.setattr(workflow.fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Recovered case reran FEM'))
    new=tmp_path/'outputs/fem_convergence/recovery_attempt'
    with pytest.raises(RuntimeError,match='planted'):
        workflow.main(['--mode','smoke','--angles','0','--resume-control',str(old),'--control-dir',str(new)])
    assert attempted==['xy_reference']
    stored=read_json(new/'summary.json')
    assert resolve_run_path(new,stored['cases'][0]['candidate']['source_run'])==source
    assert set(workflow.recovery_references(new,plan,None))=={(0.,'candidate')}


@pytest.fixture
def production_birth(workflow,tmp_path,monkeypatch):
    import shutil
    from thermal_opd.run_io import file_sha256,write_summary
    spec=workflow.make_plan('smoke',[0.])[0];source=create_run(tmp_path,'fem_comparison')
    workflow.solve_case(tmp_path,spec,source)
    snapshot=tmp_path/'production_birth'
    for name in workflow.solver_hashes():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(workflow.repository_root/name,target)
    # Plant a distinct prior implementation identity without changing solver output.
    old=snapshot/'src/thermal_opd/fem.py'
    old.write_bytes(old.read_bytes()+b'\n# planted prior production identity\n')
    birth={name:file_sha256(snapshot/name) for name in workflow.solver_hashes()}
    monkeypatch.setattr(workflow,'known_production_fem_hashes',{birth['src/thermal_opd/fem.py']})
    header=read_json(source/'summary.json');header['solver_source_sha256']=birth;write_summary(source,header)
    entry=dict(source_run=str(source),source_snapshot=str(snapshot),data_format='thermal_opd_production_nodes_v1')
    return spec,source,snapshot,entry


def test_complete_production_bridge_keeps_real_birth_and_original_nodes(workflow,tmp_path,monkeypatch,production_birth):
    from thermal_opd.run_io import resolve_run_path,write_summary
    from thermal_opd.transient_validation import _transient_fem_case_header
    spec,source,snapshot,entry=production_birth
    monkeypatch.setattr(workflow.fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Production bridge reran PDE'))
    reference=workflow.bridge_production_case(tmp_path,entry,spec,tmp_path)
    canonical=Path(reference['source_run']);header=read_json(canonical/'summary.json')
    assert header['solver_source_sha256']!=workflow.solver_hashes()
    assert header['applicable_solver_source_sha256']==workflow.solver_hashes()
    assert header['source_data_format']=='thermal_opd_production_nodes_v1'
    assert resolve_run_path(canonical,header['cases'][0]['data_file'])==source/'data/fem_run0.npz'
    assert not (canonical/'data/fem_run0.npz').exists()
    _transient_fem_case_header(canonical,dict(source_run='.',case_index=0))
    control=create_run(tmp_path,'fem_convergence');plan=workflow.make_plan('smoke',[0.])
    write_summary(control,dict(kind='transient_fem_workflow',status='FAILED',plan=plan,
        solver_source_sha256=workflow.solver_hashes(),reuse_index_sha256=None,
        cases=[dict(angle_deg=0.,candidate=reference)]))
    assert Path(workflow.recovery_references(control,plan,None)[(0.,'candidate')]['source_run'])==canonical
    assert header['fem_convergence_established'] is False


@pytest.mark.parametrize('corruption',['failed','source_summary','source_snapshot','physics_source','nodes','mesh','physical'])
def test_production_bridge_rejects_failed_or_polluted_birth(workflow,tmp_path,production_birth,corruption):
    from thermal_opd.run_io import write_summary,file_sha256
    spec,source,snapshot,entry=production_birth;header=read_json(source/'summary.json')
    if corruption=='failed':header['status']='FAILED'
    elif corruption=='source_summary':header['solver_source_sha256']['src/thermal_opd/fem.py']='0'*64
    elif corruption=='source_snapshot':
        with (snapshot/'src/thermal_opd/fem.py').open('ab') as stream:stream.write(b'polluted')
    elif corruption=='physics_source':
        path=snapshot/'src/thermal_opd/heat_source.py';path.write_bytes(b'changed')
        header['solver_source_sha256']['src/thermal_opd/heat_source.py']=file_sha256(path)
    elif corruption in ('nodes','mesh','physical'):
        path=source/'data/fem_run0.npz'
        if corruption=='nodes':
            with path.open('ab') as stream:stream.write(b'changed')
        else:
            with np.load(path) as data:values={k:data[k] for k in data.files}
            if corruption=='mesh':values['axial_nodes_z_m'][2]*=1.01
            else:values['temperature_K'][1]*=2
            np.savez_compressed(path,**values);header['cases'][0]['data_sha256']=file_sha256(path)
    write_summary(source,header)
    with pytest.raises(ValueError):workflow.bridge_production_case(tmp_path,entry,spec,tmp_path)
