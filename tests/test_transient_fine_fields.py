import json
import numpy as np
import pytest
from thermal_opd.transient_validation import validate_saved_transient_run


def test_fine_field_checker_rejects_wrong_pv_and_saved_exterior_values(tmp_path):
    # A stale PV table or values outside the transmitted aperture must fail
    # even if all figures and the declared precision flags are present.
    times = np.array([0.,.625,2.5])
    mask = np.array([[False,True],[True,True]])
    T = np.broadcast_to(times[:,None,None,None],(3,3,2,4)).copy()
    O = np.full((3,2,2),np.nan)
    O[:,mask] = times[:,None]*np.array([1.,2.,3.])*1e-9
    fields = dict(times_s=times,temperature_r_m=np.array([0.,.5]),temperature_phi_rad=np.arange(4)*np.pi/2,
                  temperature_z_m=np.array([0.,.25,.5]),temperature_K=T,opd_x_m=np.array([[.9,0.],[0.,-.1]]),
                  opd_y_m=np.array([[.9,0.],[.1,0.]]),aperture_mask=mask,opd_m=O,
                  sampled_pv_m=np.array([0.,1.25e-9,5.e-9]),orders=np.array([0]),
                  order_temperature_K=T[None,:],order_opd_m=O[None,:])
    summary = dict(kind='transient_fine_fields',status='COMPLETE',times_s=times.tolist(),
        physical_parameters_SI=dict(radius=1.,thickness=.5),aperture_radius_ratio=.9,refraction_angle_deg=0.,
        numerical_settings=dict(temperature_r=2,temperature_phi=4,opd_xy=2),
        snapshots=[dict(time_s=float(t),sampled_pv_m=float(pv),figure=None) for t,pv in zip(times,fields['sampled_pv_m'])],
        figures=['case.png'],figure_directory='figures',transient_truncation_established=False,fem_convergence_established=False,
        selection_directory=None)
    (tmp_path/'figures').mkdir()
    (tmp_path/'figures/case.png').write_bytes(b'figure-presence-fixture')
    np.savez(tmp_path/'fields.npz',**fields)
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    report = validate_saved_transient_run(tmp_path)
    assert report['status']=='PASS'
    assert report['transient_boundary_check']['status']=='NOT_RUN'
    summary['boundary_validation_required'] = True
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Missing.*boundary'):
        validate_saved_transient_run(tmp_path)
    summary['boundary_validation_required'] = False
    summary['snapshots'][1]['sampled_pv_m'] += 1e-9
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='PV'):
        validate_saved_transient_run(tmp_path)
    summary['snapshots'][1]['sampled_pv_m'] -= 1e-9
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    fields['opd_m'][1,0,0] = 0.
    np.savez(tmp_path/'fields.npz',**fields)
    with pytest.raises(ValueError,match='exterior|Exterior'):
        validate_saved_transient_run(tmp_path)


def test_fine_validator_reads_selection_relative_to_its_run(tmp_path,monkeypatch):
    import thermal_opd.transient_validation as validation
    from thermal_opd.run_io import create_run,write_summary
    selection=create_run(tmp_path,'convergence')
    run=create_run(tmp_path,'transient_case')
    selected=dict(kind='transient_joint_truncation_selection',status='COMPLETE',
                  physical_parameters_SI={},angles_deg=[60],times_s=[0,1],
                  selected_settings={'modal':{'M':2},'J':4})
    write_summary(selection,selected)
    summary=dict(physical_parameters_SI={},transient_truncation_established=True,
                 selection_directory=str(selection),angle_deg=60,modal_settings={'M':2},
                 selected_axial_modes=4,precision_scope={'times_s':[0,1]},fem_convergence_established=False)
    write_summary(run,summary)
    stored=json.loads((run/'summary.json').read_text())
    visited=[]
    monkeypatch.setattr(validation,'validate_saved_transient_run',lambda path:visited.append(path))
    class ReachedFields(Exception):
        pass
    def stop_at_fields(*args,**kwargs):
        raise ReachedFields
    monkeypatch.setattr(validation.np,'load',stop_at_fields)
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(ReachedFields):
        validation.validate_transient_fields(run,stored)
    assert visited==[selection]
