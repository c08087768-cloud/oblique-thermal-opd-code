"""Saved quadratic FEM must be rechecked using its actual nodal shape functions."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from skfem import Basis,ElementTriP2
from thermal_opd import fem
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.transient import PowerHistory
from thermal_opd.run_io import create_run,data_directory,write_summary,read_json
from thermal_opd.transient_validation import transient_fem_convergence_report


def saved_case(root):
    p=PhysicalParameters();times=np.array([0.,.001,.0025])
    result=fem.solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
        lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,0.,p.absorbed_power),
        density=p.density,specific_heat=p.specific_heat,times=times,time_step=.001/8,
        history=PowerHistory(),disk_refinements=2,axial_layers=8,quadrature_order=10,
        solver='direct',rtol=1e-12,field_degree=2,time_integrator='bdf2')
    source=create_run(root,'fem_comparison');frame=result.frames[0];keys=list(result.steps[0])
    np.savez_compressed(data_directory(source)/'fem_run0.npz',disk_nodes_xy_m=frame.disk_mesh.p,
        disk_triangles=frame.disk_mesh.t,axial_nodes_z_m=frame.z_m,times_s=times,
        xy_dof_coordinates_m=Basis(frame.disk_mesh,ElementTriP2()).doflocs,field_degree=2,
        temperature_K=np.array([f.temperature_K for f in result.frames]),step_keys=np.array(keys),
        steps=np.array([[s[k] for k in keys] for s in result.steps]))
    header=dict(kind='transient_fem_joint_case',status='COMPLETE',angle_deg=0.,
        physical_parameters_SI=asdict(p),history=asdict(PowerHistory()),times_s=times.tolist(),
        cases=[dict(index=0,diagnostics=result.diagnostics)])
    write_summary(source,header)
    out=create_run(root,'fem_convergence_validation')
    manifest=dict(angles_deg=[0.],times_s=times.tolist(),temperature_xy_points=15,opd_xy_points=13,
        diagnostic_only_times_s=[.001],cases=[dict(angle_deg=0.,candidate=dict(source_run=str(source),case_index=0))])
    return source,header,out,manifest


def test_saved_p2_nodal_energy_robin_and_opd_are_recomputed_without_solver(tmp_path,monkeypatch):
    source,header,out,manifest=saved_case(tmp_path)
    monkeypatch.setattr(fem,'solve_transient_fem',lambda *a,**k:pytest.fail('Saved review reran FEM'))
    report,fields=transient_fem_convergence_report(out,manifest)
    checks={c['name']:c for c in report['checks']}
    for name in ('source_completion','initial_condition','energy_balance','weak_equation_residual','robin_loss_diagnostics','prescribed_load'):
        assert checks['theta0_'+name]['status']=='passed',checks['theta0_'+name]
    assert 'theta0_candidate_temperature_K' in fields
    assert report['fem_convergence_established'] is False


def test_saved_p2_dof_coordinate_corruption_is_rejected(tmp_path):
    source,header,out,manifest=saved_case(tmp_path);file=source/'data/fem_run0.npz'
    with np.load(file) as f:values={k:f[k] for k in f.files}
    values['xy_dof_coordinates_m'][0,0]+=.001
    np.savez_compressed(file,**values)
    with pytest.raises(ValueError,match='DOF|coordinate'):
        transient_fem_convergence_report(out,manifest)


def test_explicit_referenced_probe_nodes_can_be_read_without_copying_data(tmp_path):
    source,header,out,manifest=saved_case(tmp_path);file=source/'data/fem_run0.npz'
    with np.load(file) as f:values={k:f[k] for k in f.files}
    probe=create_run(tmp_path,'fem_order_probe');path=probe/'data/nodes.npz'
    for old,new in (('disk_nodes_xy_m','xy_vertices_m'),('disk_triangles','triangles'),('axial_nodes_z_m','z_dofs_m')):
        values[new]=values.pop(old)
    np.savez_compressed(path,**values)
    header['cases'][0].update(data_file=str(path),data_format='thermal_opd_probe_nodes_v1')
    write_summary(source,header)
    assert not Path(read_json(source/'summary.json')['cases'][0]['data_file']).is_absolute()
    file.unlink()  # No canonical duplicate is needed for the explicit reference.
    report,_=transient_fem_convergence_report(out,manifest)
    checks={c['name']:c for c in report['checks']}
    assert checks['theta0_source_completion']['status']=='passed'
    assert checks['theta0_robin_loss_diagnostics']['status']=='passed'


def test_changing_polynomial_order_cannot_be_called_isolated_time_refinement(tmp_path):
    source,header,out,manifest=saved_case(tmp_path);p=PhysicalParameters();times=manifest['times_s']
    with np.load(source/'data/fem_run0.npz') as raw:z=raw['axial_nodes_z_m']
    result=fem.solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
        lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,0.,p.absorbed_power),
        density=p.density,specific_heat=p.specific_heat,times=times,time_step=.001/16,
        history=PowerHistory(),disk_refinements=2,axial_layers=len(z)-1,axial_nodes_m=z,
        quadrature_order=10,solver='direct',rtol=1e-12,time_integrator='bdf2')
    reference=create_run(tmp_path,'fem_comparison');frame=result.frames[0];keys=list(result.steps[0])
    np.savez_compressed(data_directory(reference)/'fem_run0.npz',disk_nodes_xy_m=frame.disk_mesh.p,
        disk_triangles=frame.disk_mesh.t,axial_nodes_z_m=frame.z_m,times_s=times,
        temperature_K=np.array([f.temperature_K for f in result.frames]),step_keys=np.array(keys),
        steps=np.array([[s[k] for k in keys] for s in result.steps]))
    write_summary(reference,{**header,'cases':[dict(index=0,diagnostics=result.diagnostics)]})
    manifest['cases'][0]['time_reference']=dict(source_run=str(reference),case_index=0)
    report,_=transient_fem_convergence_report(out,manifest)
    check=next(c for c in report['checks'] if c['name']=='theta0_time_convergence')
    assert check['metrics']['isolated_parameter_change'] is False
    assert check['status']=='failed'
