"""Independent P2 saved-field integrals and directional steady evidence."""
from copy import deepcopy
from dataclasses import asdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import csv
import json
import sys

import numpy as np
import pytest
from skfem import Basis, ElementTriP2, MeshTri

from thermal_opd import validation_common
from thermal_opd.convergence import field_errors, within_tolerance
from thermal_opd.fem import solve_steady_fem
from thermal_opd.run_io import create_run, data_directory, figure_directory
from thermal_opd.steady_validation import validate_steady_fem


def test_quadratic_shape_integrals_recover_known_polynomial_and_reject_pollution():
    # Unit right triangle: integral x²+y²=1/6; boundary integral is
    # (2+2*sqrt(2))/3.  Nonuniform axial elements integrate z² to 8/3.
    mesh = MeshTri(np.array([[0., 1., 0.], [0., 0., 1.]]), np.array([[0], [1], [2]]))
    xy = Basis(mesh, ElementTriP2()).doflocs
    z = np.array([0., .2, .4, 1.2, 2.])
    assert hasattr(validation_common, 'quadratic_fem_integral_weights')
    area, side, axial = validation_common.quadratic_fem_integral_weights(mesh.p, mesh.t, z, xy)
    q = xy[0]**2 + xy[1]**2
    np.testing.assert_allclose(area @ q, 1/6, rtol=1e-14)
    np.testing.assert_allclose(side @ q, (2+2*np.sqrt(2))/3, rtol=1e-14)
    np.testing.assert_allclose(axial @ z**2, 8/3, rtol=1e-14)
    np.testing.assert_allclose(axial @ (z[:, None]**2*q), (8/3)*q, rtol=1e-14, atol=1e-15)
    bad = xy.copy(); bad[0, -1] += .001
    with pytest.raises(ValueError, match='DOF|coordinate'):
        validation_common.quadratic_fem_integral_weights(mesh.p, mesh.t, z, bad)
    bad_z = z.copy(); bad_z[1] += .001
    with pytest.raises(ValueError, match='midpoint|axial'):
        validation_common.quadratic_fem_integral_weights(mesh.p, mesh.t, bad_z, xy)


def test_formal_steady_defaults_require_p2_separate_xy_z_and_joint():
    path = Path(__file__).parents[1]/'scripts/verification/compare_steady_fem.py'
    spec = spec_from_file_location('steady_fem_comparison_under_test', path)
    script = module_from_spec(spec); spec.loader.exec_module(script)
    assert script.full_meshes == ((4, 32), (5, 32), (4, 64), (5, 64))
    assert script.full_temperature_xy == 201 and script.full_opd_xy == 101
    assert script.quadrature_order == 10 and script.check_quadrature_order == 14
    assert script.full_field_degree == 2 and script.full_mirror_reduction is True


def saved_steady_case(root, degree=2):
    directory = create_run(root, 'fem_comparison')
    settings = [(1, 2), (2, 2), (1, 4), (2, 4)] if degree == 2 else [(1, 2), (2, 4)]
    solutions = [solve_steady_fem(1., .5, 1.4, 2., 2., 2., lambda x,y: 1.+0*x,
        disk_refinements=l, axial_layers=n, quadrature_order=10, solver='direct',
        rtol=1e-12, field_degree=degree) for l,n in settings]
    x, y = np.meshgrid([-.05, 0., .05], [-.05, 0., .05])
    planes = [0., .25, .5]; mask = np.ones_like(x, dtype=bool)
    temperatures = np.array([np.array([s.temperature_at(x,y,z) for z in planes]) for s in solutions])
    opds = np.array([s.refracted_opd(x,y,0.,1.,point_chunk_size=9) for s in solutions])
    rows = []
    for j, s in enumerate(solutions):
        row = dict(mesh_index=j, **s.diagnostics,
            fem_opd_pv_common_aperture_m=float(np.ptp(opds[j])),
            errors_vs_sa=field_errors(temperatures[j][:,mask], temperatures[-1][:,mask], np.ones(mask.sum()), opds[j,mask], opds[-1,mask]),
            changes_vs_finest_fem=field_errors(temperatures[j][:,mask], temperatures[-1][:,mask], np.ones(mask.sum()), opds[j,mask], opds[-1,mask]))
        if degree == 2:
            row['mesh_role'] = ('candidate', 'xy_reference', 'z_reference', 'joint_reference')[j]
        rows.append(row)
        payload = dict(disk_nodes_xy_m=s.disk_mesh.p, disk_triangles=s.disk_mesh.t,
                       axial_nodes_z_m=s.z_m, temperature_K=s.temperature_K)
        if degree == 2:
            payload.update(xy_dof_coordinates_m=Basis(s.disk_mesh,ElementTriP2()).doflocs,field_degree=degree)
        np.savez_compressed(data_directory(directory)/f'theta0_mesh{j}.npz', **payload)
    q_errors = field_errors(temperatures[-1][:,mask], temperatures[-1][:,mask], np.ones(mask.sum()), opds[-1,mask],opds[-1,mask])
    case = dict(angle_deg=0.,refraction_angle_deg=0.,temperature_common_sample_count=int(mask.sum()),
        opd_common_sample_count=int(mask.sum()),sa_opd_pv_common_aperture_m=float(np.ptp(opds[-1])),
        meshes=rows,load_quadrature_check=dict(order=14,errors=q_errors,passed=True),
        finest_mesh_change_within_target=within_tolerance(rows[-2]['changes_vs_finest_fem'], 1e-3))
    if degree == 2:
        from thermal_opd.steady_validation import steady_fem_directional_checks
        case['directional_mesh_checks'] = steady_fem_directional_checks(rows,temperatures[...,mask],opds[...,mask],1e-3)
    summary = dict(physical_parameters_SI={'thickness':.5,'dn_dT':1.,'radius':1.},h_eff_W_per_m2_K=2.,sanity_pass=True,
        angles_deg=[0.],cases=[case],sanity_tolerance=1e-7,mesh_change_target=1e-3,mode='smoke',
        mesh_convergence_established=False)
    if degree == 2:
        summary['mesh_refinement_protocol']='p2_xy_z_joint_v1'
    np.savez_compressed(data_directory(directory)/'theta0_fields.npz',temperature_mask=mask,opd_common_mask=mask,
        opd_physical_mask=mask,temperature_sa_K=temperatures[-1],opd_sa_m=opds[-1],temperature_fem_K=temperatures,
        opd_fem_m=opds,temperature_higher_quadrature_K=temperatures[-1],opd_higher_quadrature_m=opds[-1],
        temperature_x_m=x,temperature_y_m=y,temperature_z_m=planes,opd_x_m=x,opd_y_m=y)
    (figure_directory(directory)/'theta0_comparison.png').touch()
    (data_directory(directory)/'comparison.csv').touch()
    return directory,summary


def test_p2_saved_steady_reloads_real_dofs_and_integrates_independently(tmp_path,monkeypatch):
    assert hasattr(validation_common,'quadratic_fem_integral_weights')
    directory,summary = saved_steady_case(tmp_path)
    import skfem
    from thermal_opd import fem
    def forbidden(*args,**kwargs):
        pytest.fail('Saved review reran a solver or assembled a form')
    monkeypatch.setattr(skfem,'asm',forbidden)
    monkeypatch.setattr(fem,'asm',forbidden)
    monkeypatch.setattr(fem,'solve_steady_fem',forbidden)
    validate_steady_fem(directory,summary)
    file = data_directory(directory)/'theta0_mesh0.npz'
    with np.load(file) as f: fields = {k:f[k] for k in f.files}
    fields['xy_dof_coordinates_m'][0,-1] += .001
    np.savez_compressed(file,**fields)
    with pytest.raises(ValueError,match='DOF|coordinate'):
        validate_steady_fem(directory,summary)


def test_legacy_p1_saved_steady_remains_readable(tmp_path):
    directory,summary = saved_steady_case(tmp_path,degree=1)
    validate_steady_fem(directory,summary)


def test_directional_evidence_rejects_mixed_degree_and_missing_direction():
    from thermal_opd import steady_validation
    assert hasattr(steady_validation,'steady_fem_directional_checks')
    rows = [dict(mesh_role=role,disk_refinements=l,axial_layers=n,field_degree_xy=2,field_degree_z=2)
            for role,l,n in [('candidate',4,32),('xy_reference',5,32),('z_reference',4,64),('joint_reference',5,64)]]
    # Nonconstant fields make PV meaningful; all grids initially identical.
    temperature = np.tile(np.array([[1.,2.,3.]]),(4,3,1)); opd = np.tile([1.,2.,3.],(4,1))
    checks=steady_validation.steady_fem_directional_checks(rows,temperature,opd,1e-3)
    assert all(c['passed'] for c in checks.values())
    broken=deepcopy(rows); broken[1]['axial_layers']=64
    checks=steady_validation.steady_fem_directional_checks(broken,temperature,opd,1e-3)
    assert checks['xy']['passed'] is False
    broken=deepcopy(rows); broken[2]['field_degree_xy']=1;broken[2]['field_degree_z']=1
    checks=steady_validation.steady_fem_directional_checks(broken,temperature,opd,1e-3)
    assert checks['z']['passed'] is False
    temperature[1] *= 1.01
    checks=steady_validation.steady_fem_directional_checks(rows,temperature,opd,1e-3)
    assert checks['xy']['passed'] is False and checks['z']['passed'] is True


def test_directional_check_uses_actual_axial_nodes_not_only_layer_count():
    from thermal_opd.steady_validation import steady_fem_directional_checks
    rows = [dict(mesh_role=role,disk_refinements=l,axial_layers=n,field_degree_xy=2,field_degree_z=2)
            for role,l,n in [('candidate',1,2),('xy_reference',2,2),('z_reference',1,4),('joint_reference',2,4)]]
    temperature = np.tile(np.array([[1.,2.,3.]]),(4,3,1));opd=np.tile([1.,2.,3.],(4,1))
    meshes=[]
    for row in rows:
        mesh=MeshTri.init_circle(nrefs=row['disk_refinements'])
        meshes.append(dict(xy=mesh.p,tri=mesh.t,z=np.linspace(0,1,2*row['axial_layers']+1)))
    meshes[1]['z']=np.array([0.,.1,.2,.6,1.])  # valid P2 nodes, different axial vertices
    checks=steady_fem_directional_checks(rows,temperature,opd,1e-3,mesh_coordinates=meshes)
    assert checks['xy']['configuration_valid'] is False
    assert checks['z']['configuration_valid'] is True


def test_p2_saved_directional_flags_cannot_be_forged(tmp_path):
    directory,summary=saved_steady_case(tmp_path)
    summary['cases'][0]['directional_mesh_checks']['xy']['passed'] = not summary['cases'][0]['directional_mesh_checks']['xy']['passed']
    with pytest.raises(ValueError,match='directional FEM'):
        validate_steady_fem(directory,summary)


def test_p2_saved_interior_nodal_pollution_cannot_hide_behind_zero_integral(tmp_path):
    directory,summary=saved_steady_case(tmp_path)
    file=data_directory(directory)/'theta0_mesh0.npz'
    with np.load(file) as raw:fields={key:raw[key] for key in raw.files}
    # A P2 triangle vertex has zero area integral; the center vertex is not
    # on the side, so this pollution leaves all saved heat-loss totals intact.
    center=np.flatnonzero(np.all(fields['xy_dof_coordinates_m']==0,axis=0))[0]
    fields['temperature_K'][0,center]+=.1
    np.savez_compressed(file,**fields)
    with pytest.raises(ValueError,match='temperature|Temperature'):
        validate_steady_fem(directory,summary)


def test_p2_script_small_end_to_end_saves_real_dofs_and_nested_csv(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/'scripts/verification/compare_steady_fem.py'
    spec=spec_from_file_location('steady_fem_cli_under_test',path)
    script=module_from_spec(spec);spec.loader.exec_module(script)
    selection=tmp_path/'selection';selection.mkdir()
    (selection/'summary.json').write_text(json.dumps(dict(kind='steady_joint_truncation_selection',status='COMPLETE',
        physical_parameters_SI=asdict(script.parameters),aperture_radius_ratio=script.aperture_radius_ratio,
        selected_settings=asdict(script.ModalResolution(0,1,24,12,12)),angles_deg=[0.])),encoding='utf-8')
    # Deliberately small execution fixture; the formal defaults are checked
    # separately above and this coarse result must not assert convergence.
    monkeypatch.setattr(script,'full_meshes',((1,2),(2,2),(1,4),(2,4)))
    monkeypatch.setattr(script,'full_angles_deg',(0.,))
    monkeypatch.setattr(script,'full_temperature_xy',15)
    monkeypatch.setattr(script,'full_opd_xy',13)
    monkeypatch.setattr(script,'create_run',lambda repository_root,prefix,output_dir: create_run(tmp_path,prefix,output_dir))
    out=tmp_path/'outputs/fem_comparison/p2_comparison'
    monkeypatch.setattr(sys,'argv',['compare_steady_fem.py','--mode','full','--selection',str(selection),'--output-dir',str(out)])
    script.main()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['mesh_convergence_established'] is False
    assert set(summary['cases'][0]['directional_mesh_checks'])=={'xy','z','joint'}
    assert summary['field_degree']==2 and summary['mirror_reduction'] is True
    with np.load(data_directory(out)/'theta0_mesh0.npz') as saved:
        assert saved['field_degree'].item()==2
        assert saved['temperature_K'].shape==(5,saved['xy_dof_coordinates_m'].shape[1])
        assert saved['xy_dof_coordinates_m'].shape[1]>saved['disk_nodes_xy_m'].shape[1]
    with (data_directory(out)/'comparison.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==4
    assert isinstance(json.loads(rows[0]['mirror_reduction']),dict)
    validate_steady_fem(out,summary)
