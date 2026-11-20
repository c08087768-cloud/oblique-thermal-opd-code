from dataclasses import asdict
import json
import numpy as np
import pytest
from thermal_opd.model import ThermalModel,ModalResolution
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.steady_validation import steady_boundary_residuals,steady_energy_balance
from thermal_opd.steady_validation import validate_saved_steady_run


def test_legacy_steady_fields_are_checked_against_saved_modal_model(tmp_path):
    p=PhysicalParameters();s=ModalResolution(0,1,40,16,16)
    prepared=ThermalModel(p,0.,s).prepare();solution=prepared.solve_steady()
    r,phi,z=np.linspace(0,p.radius,5),np.arange(8)*np.pi/4,np.array([0,p.thickness/2,p.thickness])
    axis=np.linspace(-.5*p.radius,.5*p.radius,5);x,y=np.meshgrid(axis,axis);mask=x*x+y*y<=(.5*p.radius)**2
    T=np.stack([solution.temperature(r[:,None],phi[None,:],zz) for zz in z])
    O=np.full(x.shape,np.nan);O[mask]=solution.opd(x[mask],y[mask])
    fields=dict(temperature_r_m=r,temperature_phi_rad=phi,temperature_z_m=z,temperature_rise_K=T,opd_x_m=axis,opd_y_m=axis,aperture_mask=mask,opd_m=O)
    settings=dict(M=0,N=1,projection_r=40,projection_phi=16,opd_ray_points=16,opd_point_chunk_size=512,boundary_z=9,energy_r=16,energy_z=16,energy_phi=16)
    bc=steady_boundary_residuals(r,phi,np.linspace(0,p.thickness,9),prepared.coefficients,prepared.roots,p.radius,p.conductivity,p.thickness,p.h_eff,p.h_eff,p.h_eff)
    energy=steady_energy_balance(prepared.coefficients,prepared.roots,p.radius,p.conductivity,p.thickness,p.h_eff,p.h_eff,p.h_eff,num_r=16,num_z=16,num_phi=16)
    summary=dict(kind='complete_representative_steady_refracted_ray_case',physical_parameters_SI=asdict(p),incidence_angle_deg=0.,
        numerical_settings=settings,aperture_radius_ratio=.5,opd_pv_m=float(np.ptp(O[mask])),
        eigenvalues_per_m={str(m):a.tolist() for m,a in prepared.roots.items()},source_coefficients_W_per_m2={str(m):a.tolist() for m,a in prepared.coefficients.items()},
        boundary_absolute_residuals_W_per_m2=bc,energy_balance=energy)
    np.savez(tmp_path/'fields.npz',**fields);(tmp_path/'summary.json').write_text(json.dumps(summary))
    assert validate_saved_steady_run(tmp_path)['status']=='PASS'
    fields['temperature_rise_K'][1,2,3]+=.1
    np.savez(tmp_path/'fields.npz',**fields)
    with pytest.raises(ValueError,match='Steady|steady'):
        validate_saved_steady_run(tmp_path)
