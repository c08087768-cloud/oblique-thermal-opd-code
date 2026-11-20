"""Entry-point paths must work independently of the caller's current directory."""
import importlib.util
from pathlib import Path
import sys

import pytest
from thermal_opd.experiments import EXPERIMENTS, ROOT, experiment_configuration


EXPECTED_ENTRIES = {
    'steady':'cases/run_steady_case.py',
    'transient':'cases/run_transient_case.py',
    'convergence':'convergence/select_steady_truncation.py',
    'transient-convergence':'convergence/select_transient_truncation.py',
    'semi-validation':'verification/verify_semi_analytical.py',
    'fem-validation':'verification/verify_fem.py',
    'fem-comparison':'verification/compare_steady_fem.py',
    'transient-fem-validation':'verification/verify_transient_fem_convergence.py',
    'transient-fem-convergence':'convergence/check_transient_fem_convergence.py',
    'validate':'verification/validate_run.py',
    'boundaries':'verification/check_transient_boundaries.py',
    'source':'diagnostics/inspect_heat_source.py',
    'azimuthal':'diagnostics/inspect_azimuthal_modes.py',
    'temperature':'diagnostics/inspect_steady_temperature.py',
    'opd':'diagnostics/inspect_opd.py',
    'sweeps':'convergence/check_steady_convergence.py',
    'numerical-settings':'convergence/check_steady_numerical_settings.py',
    'replot':'tools/replot_case.py',
    'regression':'tools/compare_refactor_baseline.py',
}


def test_exact_script_layout_and_existing_experiment_names():
    assert list(EXPERIMENTS)==list(EXPECTED_ENTRIES)
    assert {name:value[1] for name,value in EXPERIMENTS.items()}==EXPECTED_ENTRIES
    scripts=ROOT/'scripts'
    assert not list(scripts.glob('*.py'))
    assert {p.relative_to(scripts).as_posix() for p in scripts.rglob('*.py')}=={
        *EXPECTED_ENTRIES.values(),'cases/transient_fields.py','cases/recompute_paper_optics.py'}


@pytest.mark.parametrize('name',EXPECTED_ENTRIES)
def test_preview_from_another_directory_keeps_repository_root(name,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    config=experiment_configuration(name)
    assert Path(config['script'])==ROOT/'scripts'/EXPECTED_ENTRIES[name]
    spec=importlib.util.spec_from_file_location('_layout_root_preview',config['script'])
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    if hasattr(module,'repository_root'):
        assert module.repository_root==ROOT


@pytest.mark.parametrize('name',('azimuthal','regression'))
def test_function_local_root_keeps_default_output_in_repository(name,tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('_local_root_test',ROOT/'scripts'/EXPECTED_ENTRIES[name])
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    argv=[str(module.__file__)]
    if name=='regression':
        argv+=['--baseline',str(tmp_path/'baseline')]
    monkeypatch.setattr(sys,'argv',argv)
    class BeforeOutput(Exception):
        pass
    def capture(root,label,output):
        assert root==ROOT
        assert output is None
        raise BeforeOutput
    monkeypatch.setattr(module,'create_run',capture)
    with pytest.raises(BeforeOutput):
        module.main()


@pytest.mark.parametrize('mode',('fields','smoke','full'))
def test_transient_entry_reaches_its_sibling_before_any_solve(mode,tmp_path,monkeypatch):
    cases=ROOT/'scripts/cases'
    spec=importlib.util.spec_from_file_location('transient_fields',cases/'transient_fields.py')
    sibling=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sibling)
    monkeypatch.setitem(sys.modules,'transient_fields',sibling)
    spec=importlib.util.spec_from_file_location('_transient_entry_layout_test',cases/'run_transient_case.py')
    entry=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.chdir(tmp_path)
    selection=tmp_path/'saved-selection'
    arguments=[str(cases/'run_transient_case.py'),'--mode',mode,'--selection',str(selection)]
    if mode=='full':arguments+=['--fem-convergence',str(tmp_path/'saved-fem-certificate')]
    monkeypatch.setattr(sys,'argv',arguments)
    class ReachedSelection(Exception):
        pass
    calls=[]
    def fields(args,p,root):
        calls.append((args.selection,root))
    def load(directory,p,angle,history):
        calls.append((directory,entry.repository_root))
        raise ReachedSelection
    monkeypatch.setattr(sibling,'run_fine_fields',fields)
    monkeypatch.setattr(sibling,'load_selection',load)
    if mode=='fields':
        entry.main()
    else:
        with pytest.raises(ReachedSelection):
            entry.main()
    assert calls==[(selection,ROOT)]
