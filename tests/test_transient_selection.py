import json
import numpy as np
import pytest
from thermal_opd.transient_validation import validate_saved_transient_run


def saved_selection(tmp_path):
    # A changed early-time field or a claimed pass above the numeric threshold
    # must not authorize a fine figure run.
    zero = dict(temperature_relative_l2=[0.,0.,0.],temperature_relative_linf=[0.,0.,0.],
        opd_piston_removed_rms_over_reference_pv=0.,opd_piston_removed_linf_over_reference_pv=0.,opd_relative_pv=0.)
    settings = {'modal':dict(M=2,N=2,projection_r=40,projection_phi=16,ray_points=8),'J':12}
    case = dict(angle_deg=0.,reference_pass=True,reference_errors_by_time=[zero,zero],
        candidates=[dict(settings=settings,errors_by_time=[zero,zero],accepted=True)],
        denser_common_grid_pass=True,denser_errors_by_time=[zero,zero],
        sampling_errors_by_time=[0.,0.],sampling_pass=True)
    summary = dict(kind='transient_joint_truncation_selection',status='COMPLETE',angles_deg=[0.],
        times_s=[.625,2.5],tolerance=.001,reference_stability_tolerance=.0002,
        selected_settings=settings,selected_figure_xy=201,cases=[case])
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    T = np.tile(np.array([[1.,2.,3.]]),(2,3,1))
    O = np.tile(np.array([1.,2.,3.]),(2,1))
    fields = dict(temperature_area_weights=np.ones(3))
    for key in ('reference_temperature_K','enlarged_temperature_K','selected_temperature_K'):
        fields['theta0_'+key] = T.copy()
    for key in ('reference_opd_m','enlarged_opd_m','selected_opd_m','fine_reference_opd_m','fine_selected_opd_m','figure_opd_m'):
        fields['theta0_'+key] = O.copy()
    fields['theta0_reference_continuous_pv_m'] = np.array([2.,2.])
    fields['theta0_enlarged_continuous_pv_m'] = np.array([2.,2.])
    fields['theta0_selected_continuous_pv_m'] = np.array([2.,2.])
    np.savez(tmp_path/'fields.npz',**fields)
    return summary,fields


def test_transient_selection_checker_recomputes_time_fields_and_rejects_false_gate(tmp_path):
    summary,fields=saved_selection(tmp_path)
    case=summary['cases'][0]
    T=fields['theta0_selected_temperature_K'].copy()
    assert validate_saved_transient_run(tmp_path)['status']=='PASS'
    fields['theta0_selected_temperature_K'][0,0,0] += .1
    np.savez(tmp_path/'fields.npz',**fields)
    with pytest.raises(ValueError):
        validate_saved_transient_run(tmp_path)
    fields['theta0_selected_temperature_K'] = T
    np.savez(tmp_path/'fields.npz',**fields)
    case['sampling_errors_by_time'] = [.002,0.]
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        validate_saved_transient_run(tmp_path)


def test_selection_loader_derives_missing_flag_only_after_real_saved_checks(tmp_path):
    from dataclasses import asdict
    from thermal_opd.parameters import PhysicalParameters
    from thermal_opd.transient_validation import load_transient_selection
    p=PhysicalParameters()
    summary,fields=saved_selection(tmp_path)
    summary.update(physical_parameters_SI=asdict(p),history={'kind':'step'})
    file=tmp_path/'summary.json'
    file.write_text(json.dumps(summary))
    original=file.read_bytes()
    selected=load_transient_selection(tmp_path,p,0.)
    assert selected['transient_truncation_established'] is True
    assert file.read_bytes()==original
    assert 'transient_truncation_established' not in json.loads(original)
    fields['theta0_selected_temperature_K'][0,0,0]+=.1
    np.savez(tmp_path/'fields.npz',**fields)
    with pytest.raises(ValueError):
        load_transient_selection(tmp_path,p,0.)


def test_selection_loader_rejects_explicit_false_and_uncovered_scope(tmp_path):
    from dataclasses import asdict
    from thermal_opd.parameters import PhysicalParameters
    from thermal_opd.transient_validation import load_transient_selection
    p=PhysicalParameters()
    summary,_=saved_selection(tmp_path)
    summary.update(physical_parameters_SI=asdict(p),history={'kind':'step'},
                   transient_truncation_established=False)
    file=tmp_path/'summary.json';file.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='established'):
        load_transient_selection(tmp_path,p,0.)
    summary.pop('transient_truncation_established');file.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='scope'):
        load_transient_selection(tmp_path,p,30.)


@pytest.mark.parametrize('declared,actual', [([30.],[0.]),([0.,0.],[0.,0.]),([0.,30.],[0.,0.])])
def test_selection_loader_rejects_declared_angle_without_actual_case(tmp_path,declared,actual):
    from dataclasses import asdict
    from thermal_opd.parameters import PhysicalParameters
    from thermal_opd.transient_validation import load_transient_selection
    p=PhysicalParameters()
    summary,_=saved_selection(tmp_path)
    summary['cases']=[{**summary['cases'][0],'angle_deg':a} for a in actual]
    summary.update(physical_parameters_SI=asdict(p),history={'kind':'step'},angles_deg=declared)
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='angle|Angle|scope'):
        load_transient_selection(tmp_path,p,declared[-1])
