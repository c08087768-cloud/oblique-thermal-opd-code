import json
import numpy as np


def test_replot_uses_saved_fields_without_preparing_or_overwriting_model(tmp_path,monkeypatch):
    from thermal_opd.plotting import replot_saved_case
    from thermal_opd.model import ThermalModel
    import thermal_opd.run_io as run_io
    create_run=run_io.create_run
    monkeypatch.setattr(run_io,'create_run',lambda root,label,output: create_run(tmp_path,label,output))
    def forbidden(*args,**kwargs):
        raise AssertionError('Replot reran the physical model')
    monkeypatch.setattr(ThermalModel,'prepare',forbidden)
    r=np.linspace(0,1.,5);phi=np.arange(8)*np.pi/4
    T=np.stack([np.broadcast_to(1-r[:,None]**2,(5,8))*(i+1) for i in range(3)])
    axis=np.linspace(-.5,.5,5);x,y=np.meshgrid(axis,axis);mask=x*x+y*y<=.25
    O=np.where(mask,(x*x+y*y)*1e-8,np.nan)
    np.savez(tmp_path/'fields.npz',temperature_r_m=r,temperature_phi_rad=phi,temperature_rise_K=T,
             opd_x_m=axis,opd_y_m=axis,aperture_mask=mask,opd_m=O)
    (tmp_path/'summary.json').write_text(json.dumps(dict(kind='complete_representative_steady_refracted_ray_case',status='COMPLETE',
                                                       incidence_angle_deg=60.,opd_pv_m=float(np.ptp(O[mask])))))
    before=(tmp_path/'fields.npz').read_bytes()
    rendered=replot_saved_case(tmp_path,output_dir=tmp_path/'outputs/replot/rendered')
    assert (rendered/'figures/steady_case.png').is_file()
    assert (tmp_path/'fields.npz').read_bytes()==before
    assert run_io.resolve_run_path(rendered,json.loads((rendered/'summary.json').read_text())['source_run'])==tmp_path
