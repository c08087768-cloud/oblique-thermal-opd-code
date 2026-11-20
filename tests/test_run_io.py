import json
import numpy as np
import pytest
from thermal_opd.run_io import create_run,write_summary,data_directory,figure_directory


def test_default_runs_are_classified_unique_and_never_overwrite(tmp_path):
    a=create_run(tmp_path,'steady_case')
    b=create_run(tmp_path,'steady_case')
    assert a!=b
    assert a.parent==b.parent==tmp_path/'outputs/steady_case'
    assert data_directory(a)==a/'data'
    assert figure_directory(a)==a/'figures'
    assert data_directory(a).is_dir() and figure_directory(a).is_dir()
    angled=create_run(tmp_path,'steady_refracted_theta30.5')
    assert angled.parent==tmp_path/'outputs/steady_case'
    with pytest.raises(FileExistsError):
        create_run(tmp_path,'steady_case',a)
    with pytest.raises(ValueError):
        create_run(tmp_path,'../escape')


def test_metadata_keeps_configuration_metrics_and_unexecuted_status_separate(tmp_path):
    path=create_run(tmp_path,'transient_case')
    summary=dict(kind='example',status='RUNNING',physical_parameters_SI={'radius':.01},numerical_settings={'M':6},provenance={'git_revision':'abc'})
    write_summary(path,summary)
    config=json.loads((path/'config.json').read_text())
    assert config['physical_parameters_SI']=={'radius':.01}
    assert config['numerical_settings']=={'M':6}
    summary.update(status='COMPLETE',opd_pv_m=1e-7)
    write_summary(path,summary)
    assert json.loads((path/'metrics.json').read_text())['opd_pv_m']==1e-7
    assert json.loads((path/'verification.json').read_text())['status']=='not_run'
    assert json.loads((path/'config.json').read_text())['provenance']['git_revision']=='abc'
    assert json.loads((path/'verification.json').read_text())['config']['physical_parameters_SI']=={'radius':.01}


def test_legacy_flat_artifact_layout_remains_readable(tmp_path):
    assert data_directory(tmp_path)==tmp_path
    assert figure_directory(tmp_path)==tmp_path


def test_new_run_has_all_metadata_even_before_computation_and_on_failure(tmp_path):
    path=create_run(tmp_path,'transient_fields_theta60')
    assert path.parent==tmp_path/'outputs/transient_case'
    for name in ('run','config','summary','metrics','verification'):
        assert json.loads((path/f'{name}.json').read_text()) is not None
    assert json.loads((path/'summary.json').read_text())['status']=='RUNNING'
    assert json.loads((path/'verification.json').read_text())['status']=='not_run'
    write_summary(path,dict(status='FAILED',error='interrupted'))
    assert json.loads((path/'metrics.json').read_text())['status']=='FAILED'
    assert json.loads((path/'verification.json').read_text())['status']=='not_run'


@pytest.mark.parametrize('location', ['results/example','outputs/figures/example',
                                      'outputs/legacy/example','elsewhere/example','outputs/steady_case'])
def test_explicit_output_cannot_create_another_layout(tmp_path,location):
    with pytest.raises(ValueError,match='outputs'):
        create_run(tmp_path,'steady_case',tmp_path/location)
    assert not (tmp_path/location).exists()


def test_explicit_run_id_is_recorded_and_cannot_overwrite(tmp_path):
    path=create_run(tmp_path,'steady_case',tmp_path/'outputs/steady_case/custom-run')
    assert json.loads((path/'run.json').read_text())['run_id']=='custom-run'
    with pytest.raises(FileExistsError):
        create_run(tmp_path,'steady_case',path)


def test_metadata_paths_are_portable_and_resolved_from_the_run(tmp_path,monkeypatch):
    from thermal_opd.run_io import resolve_run_path
    selection=create_run(tmp_path,'convergence')
    path=create_run(tmp_path,'transient_case')
    source=dict(status='COMPLETE',selection_directory=str(selection),
                figure_directory=str(path/'figures'),run_directory=str(path),
                artifacts={'fields':'data/fields.npz','figure':str(path/'figures/case.png')},
                verification_references=[str(selection/'verification.json')])
    write_summary(path,source)
    saved=json.loads((path/'summary.json').read_text())
    assert saved['figure_directory']=='figures'
    assert saved['run_directory']=='.'
    assert saved['artifacts']=={'fields':'data/fields.npz','figure':'figures/case.png'}
    assert saved['selection_directory'].startswith('../../convergence/')
    assert saved['verification_references'][0].startswith('../../convergence/')
    assert source['selection_directory']==str(selection)  # No mutation of live caller settings.
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_run_path(path,saved['selection_directory'])==selection
    assert resolve_run_path(path,str(selection))==selection
    assert json.loads((path/'verification.json').read_text())['run_directory']=='.'


def test_recorded_scientific_failure_is_not_replaced_by_a_summary_update(tmp_path):
    path=create_run(tmp_path,'steady_case')
    report={'status':'failed','checks':[{'name':'pde_residual','status':'failed'}]}
    (path/'verification.json').write_text(json.dumps(report))
    before=(path/'verification.json').read_bytes()
    write_summary(path,dict(status='COMPLETE'))
    assert (path/'verification.json').read_bytes()==before
