"""Known-answer controls for the production FEM reference entry point."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from thermal_opd.run_io import create_run, data_directory


def load_entry_point():
    path = Path(__file__).parents[1] / 'scripts/verification/verify_fem.py'
    spec = spec_from_file_location('fem_reference_under_test', path)
    script = module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def run_entry_point(script, tmp_path, monkeypatch, arguments=()):
    out = tmp_path / 'outputs/fem_verification/reference'
    monkeypatch.setattr(script, 'create_run', lambda root, prefix, output: create_run(tmp_path, prefix, output))
    monkeypatch.setattr(sys, 'argv', ['verify_fem.py', '--output-dir', str(out), *arguments])
    script.main()
    report = json.loads((out / 'verification.json').read_text(encoding='utf-8'))
    return out, report, {row['name']: row for row in report['checks']}


@pytest.mark.parametrize('degree,integrator,xy_dofs', [(2, 'bdf2', 41), (1, 'backward_euler', 13)])
def test_actual_reference_entry_point_preserves_known_answers_and_scope(tmp_path, monkeypatch, degree, integrator, xy_dofs):
    script = load_entry_point()
    arguments = () if degree == 2 else ('--field-degree', '1')
    out, report, checks = run_entry_point(script, tmp_path, monkeypatch, arguments)
    config = report['config']
    assert config['field_degree'] == degree
    assert config['time_integrator'] == integrator
    assert config['solver'] == 'direct' and config['mirror_reduction'] is False
    assert config['quadrature_order'] >= 4
    assert config['times_s'] == [0., .625, 2.5, 20.]
    assert config['reference_intervals'] == [256, 512]
    assert config['reference_tolerance'] == 1e-4
    assert config['local_fem_tolerance'] == 2e-2
    assert config['steady_fem']['degrees_of_freedom'] == xy_dofs * (degree * 32 + 1)
    for row in config['fem_configurations']:
        assert row['field_degree'] == degree
        assert row['time_integrator'] == integrator
        assert row['xy_dofs'] == xy_dofs
        assert row['axial_dofs'] == degree * row['axial_layers'] + 1
        assert row['degrees_of_freedom'] == row['xy_dofs'] * row['axial_dofs']
    for name, check in checks.items():
        expected = 'not_run' if name in {
            'gaussian_case_mesh_convergence', 'gaussian_case_time_convergence',
            'pointwise_boundary_flux', 'cross_method_comparison',
        } else 'passed'
        assert check['status'] == expected, (name, check)
    assert checks['energy_balance']['threshold'] == 1e-9
    assert checks['linear_equation_residual']['threshold'] == 1e-9
    assert checks['space_refinement']['threshold'] == 1e-12
    assert checks['time_refinement']['threshold'] == 1e-12
    assert report['status'] == 'not_run'
    with np.load(data_directory(out) / 'verification_fields.npz') as saved:
        assert saved['configuration_method_balance_errors'].max() < 1e-9
        if degree == 2:
            # The BDF2 equation conserves BDF storage, while ordinary secants
            # retain their separate physical diagnostic meaning.
            assert saved['configuration_secant_balance_errors'].max() > 1e-6
            assert config['energy_balance_diagnostic'] == 'bdf_balance_absolute_error_W'
            assert config['space_refinement_indices'] == [5, 6, 7]
            assert config['time_refinement_indices'] == [2, 3, 4]
            assert config['space_time_control_index'] == 8
            assert saved['configurations'][5:].tolist() == [[8., .005], [16., .005], [32., .005], [32., .0025]]
            legacy = config['legacy_coupled_space_refinement']
            assert legacy['indices'] == [0, 1, 2]
            assert legacy['status'] == 'failed'
            assert legacy['max_error'] > legacy['threshold'] == 1e-12
            assert checks['space_time_step_stability']['threshold'] == 1e-4
            assert checks['spatial_known_1d_reference']['threshold'] == 2e-2
            assert checks['transient_known_1d_reference']['threshold'] == 2e-2
        else:
            assert config['energy_balance_diagnostic'] == 'balance_absolute_error_W'
            assert config['space_refinement_indices'] == [0, 1, 2]
            assert config['time_refinement_indices'] == [2, 3, 4]
            assert saved['configurations'].shape == (5, 2)


@pytest.mark.parametrize('corruption', ['method_energy', 'space_time_step'])
def test_corrupted_diagnostics_cannot_receive_passed_report(tmp_path, monkeypatch, corruption):
    script = load_entry_point()
    solve = script.solve_transient_fem

    def corrupt_result(*args, **kwargs):
        result = solve(*args, **kwargs)
        if corruption == 'method_energy':
            result.steps[-1]['bdf_balance_absolute_error_W'] = result.diagnostics['full_input_W']
        elif kwargs['time_step'] == .0025:
            result.frames[1].temperature_K += 1.
        return result

    monkeypatch.setattr(script, 'solve_transient_fem', corrupt_result)
    _, report, checks = run_entry_point(script, tmp_path, monkeypatch)
    name = 'energy_balance' if corruption == 'method_energy' else 'space_time_step_stability'
    assert checks[name]['status'] == 'failed'
    assert report['status'] == 'failed'


def test_invalid_field_degree_is_rejected_before_creating_output(tmp_path, monkeypatch):
    script = load_entry_point()
    with pytest.raises(SystemExit) as error:
        run_entry_point(script, tmp_path, monkeypatch, ('--field-degree', '3'))
    assert error.value.code == 2
    assert not (tmp_path / 'outputs').exists()
