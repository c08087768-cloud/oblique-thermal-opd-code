import json
import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys

from thermal_opd import steady_validation


def make_selection(tmp_path):
    settings = dict(M=2, N=2, projection_r=40, projection_phi=16, ray_points=8)
    errors = dict(temperature_relative_l2=[0., 0., 0.], temperature_relative_linf=[0., 0., 0.],
                  opd_piston_removed_rms_over_reference_pv=0., opd_piston_removed_linf_over_reference_pv=0., opd_relative_pv=0.)
    summary = dict(kind='steady_joint_truncation_selection', status='COMPLETE', angles_deg=[0.], tolerance=.001,
                   reference_stability_tolerance=.0002, selected_settings=settings, selected_figure_xy=201,
                   cases=[dict(angle_deg=0., reference_pass=True, reference_stability_errors=errors,
                               denser_common_grid_pass=True, denser_common_grid_errors=errors,
                               candidates=[dict(settings=settings, accepted=True, errors=errors)])],
                   sampling_checks=[dict(angle_deg=0., cases=[dict(xy_count=201, relative_pv_sampling_error=0.)])])
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    t, o = np.array([[2., 3.], [2., 3.], [2., 3.]]), np.array([1., 2., 3.])
    fields = dict(temperature_area_weights=np.array([1., 1.]), theta0_reference_temperature_K=t,
                  theta0_selected_temperature_K=t, theta0_reference_opd_m=o, theta0_selected_opd_m=o,
                  theta0_verification_reference_opd_m=o, theta0_verification_selected_opd_m=o)
    np.savez(tmp_path/'fields.npz', **fields)
    return summary, fields


def test_saved_selection_checker_rejects_changed_fields(tmp_path):
    _, fields = make_selection(tmp_path)
    assert steady_validation.validate_saved_steady_run(tmp_path)['status'] == 'PASS'
    fields['theta0_selected_temperature_K'] = np.array([[3., 3.], [2., 3.], [2., 3.]])
    np.savez(tmp_path/'fields.npz', **fields)
    with pytest.raises(ValueError):
        steady_validation.validate_saved_steady_run(tmp_path)


def test_saved_selection_checker_rejects_incomplete_run_and_false_acceptance(tmp_path):
    summary, _ = make_selection(tmp_path)
    summary['status'] = 'RUNNING'
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        steady_validation.validate_saved_steady_run(tmp_path)
    summary['status'] = 'COMPLETE'
    summary['cases'][0]['denser_common_grid_errors']['opd_relative_pv'] = .01
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        steady_validation.validate_saved_steady_run(tmp_path)


def test_cli_replaces_stale_pass_report_when_later_validation_fails(tmp_path):
    summary, _ = make_selection(tmp_path)
    script = Path(__file__).resolve().parents[1]/'scripts/verification/validate_run.py'
    first = subprocess.run([sys.executable, str(script), '--results', str(tmp_path)], capture_output=True)
    assert first.returncode == 0
    summary['status'] = 'RUNNING'
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    second = subprocess.run([sys.executable, str(script), '--results', str(tmp_path)], capture_output=True)
    assert second.returncode != 0
    assert json.loads((tmp_path/'check_report.json').read_text())['status'] == 'FAIL'
