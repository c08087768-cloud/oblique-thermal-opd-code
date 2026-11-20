"""Model ownership, old report behavior and the single CLI dispatcher."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from thermal_opd.parameters import PhysicalParameters

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=Path(__file__).with_name('fixtures')
REPORTS=json.loads((FIXTURES/'validation_reports_before.json').read_text(encoding='utf-8'))


def stable(value):
    value=deepcopy(value)
    if isinstance(value,dict):
        value.pop('checked_at_utc',None)
        return {key:stable(item) for key,item in value.items()}
    if isinstance(value,list):return [stable(item) for item in value]
    return value


def load_cli():
    spec=importlib.util.spec_from_file_location('_validation_cli_test',ROOT/'scripts/verification/validate_run.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_functions_are_owned_by_their_physical_model():
    from thermal_opd import steady_validation as steady,transient_validation as transient
    for name in ('steady_boundary_residuals','steady_energy_balance','steady_pde_residual',
                 'validate_saved_steady_run','validate_steady_fem'):
        assert getattr(steady,name).__module__==steady.__name__
    for name in ('transient_boundary_fields','transient_residual_fields','transient_pde_residual',
                 'transient_uniform_flux_1d_reference','validate_saved_transient_run','validate_transient_fem'):
        assert getattr(transient,name).__module__==transient.__name__


@pytest.mark.parametrize('case',REPORTS,ids=[row['name'] for row in REPORTS])
def test_scientific_reports_match_the_captured_pre_refactor_behavior(case):
    from thermal_opd.steady_validation import steady_saved_validation_report
    from thermal_opd.transient_validation import transient_saved_validation_report
    if case['summary']['kind'].startswith('steady_') or case['summary']['kind'].startswith('complete_representative_steady_'):
        report=steady_saved_validation_report(Path('validation-run'),case['summary'])
    else:
        report=transient_saved_validation_report(Path('validation-run'),case['summary'],case['boundary'])
    assert stable(report)==case['report']


def test_pde_residuals_match_the_original_shared_implementation_exactly():
    from thermal_opd.steady_validation import steady_pde_residual
    from thermal_opd.transient_validation import transient_pde_residual
    baseline=json.loads((FIXTURES/'pde_residuals_before.json').read_text(encoding='utf-8'))
    p=replace(PhysicalParameters(),radius=1.,thickness=1.,conductivity=1.,density=2.,specific_heat=3.)
    r,phi,z=np.array([.2,.5]),np.array([.1,1.2]),np.array([.3,.6])
    steady=lambda r,phi,z:r*r+z*z+np.sin(phi)
    transient=lambda r,phi,z,t:r*r+z*z+np.sin(phi)+t*t
    np.testing.assert_array_equal(steady_pde_residual(steady,p,r,phi,z,space_step=1e-3),baseline['steady'])
    np.testing.assert_array_equal(transient_pde_residual(transient,p,r,phi,z,1.,space_step=1e-3,time_step=1e-3),baseline['transient'])


@pytest.mark.parametrize('kind,target',[
    ('steady_joint_truncation_selection','validate_saved_steady_run'),
    ('steady_prismatic_fem_comparison','validate_saved_steady_run'),
    ('complete_representative_steady_refracted_ray_case','validate_saved_steady_run'),
    ('transient_joint_truncation_selection','validate_saved_transient_run'),
    ('transient_modal_fem_study','validate_saved_transient_run'),
    ('transient_fine_fields','validate_saved_transient_run'),
    ('transient_fem_convergence_validation','validate_saved_transient_run'),
    ('scientific_verification','validate_saved_evidence_run')])
def test_cli_dispatches_by_saved_experiment_kind_without_running_other_validators(tmp_path,monkeypatch,kind,target):
    cli=load_cli()
    summary={'kind':kind,'status':'COMPLETE'}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    calls=[]
    def accept(directory,summary=None):
        calls.append((directory,summary));return {'sentinel':True}
    def forbidden(*args,**kwargs):
        pytest.fail('Wrong model validator selected')
    for name in ('validate_saved_steady_run','validate_saved_transient_run','validate_saved_evidence_run'):
        monkeypatch.setattr(cli,name,accept if name==target else forbidden)
    assert cli.validate_saved_run(tmp_path)=={'sentinel':True}
    assert calls==[(tmp_path,summary)]


def test_io_does_not_import_scientific_validation_to_create_metadata(tmp_path,monkeypatch):
    import builtins
    from thermal_opd.run_io import create_run
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name in ('thermal_opd.steady_validation','thermal_opd.transient_validation','thermal_opd.verification'):
            pytest.fail('Run I/O tried to load scientific model validation')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    directory=create_run(tmp_path,'example')
    assert json.loads((directory/'verification.json').read_text(encoding='utf-8'))['status']=='not_run'


def test_obsolete_modules_are_removed_without_reexport_wrappers():
    for name in ('validation','verification','run_validation'):
        assert importlib.util.find_spec('thermal_opd.'+name) is None
