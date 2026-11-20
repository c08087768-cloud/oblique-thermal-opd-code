"""Compare independent steady FEM and selected modal solution on common points.

--mode smoke is a small local batch, not a final mesh-convergence certificate.
All settings are defined below; output directories are unique and never reused.
"""

from thermal_opd.run_io import script_defaults
from thermal_opd.model import ThermalModel
from thermal_opd.run_io import data_directory, figure_directory
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skfem import Basis, ElementTriP2

from thermal_opd.convergence import field_errors, within_tolerance
from thermal_opd.fem import solve_steady_fem
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.metrics import circular_aperture_mask
from thermal_opd.model import ModalResolution
from thermal_opd.opd import refracted_angle, transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import create_run, provenance, write_summary
from thermal_opd.steady_validation import steady_fem_directional_checks


parameters = PhysicalParameters()
aperture_radius_ratio = 0.9
smoke_angles_deg = (0., 60.)
full_angles_deg = (0., 30., 60.)
smoke_meshes = ((2, 4), (3, 8), (4, 12))  # disk refinements, axial layers
# P2 candidate and independent xy-only, z-only, joint references.  These
# configurations are candidates until each recorded directional gate passes.
full_meshes = ((4, 32), (5, 32), (4, 64), (5, 64))
full_mesh_roles = ('candidate', 'xy_reference', 'z_reference', 'joint_reference')
full_field_degree = 2
full_mirror_reduction = True
smoke_temperature_xy = 51
full_temperature_xy = 201
smoke_opd_xy = 41
full_opd_xy = 101
quadrature_order = 10
check_quadrature_order = 14
solver_rtol = 1e-10
sanity_tolerance = 1e-7
mesh_change_target = 1e-3
point_chunk_size = 128
repository_root = Path(__file__).resolve().parents[2]


def save_figure(directory, label, tx, ty, tm, t_sa, t_fem, ox, oy, om, o_sa, o_fem):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    fields = (t_sa[0], t_fem[0], t_fem[0]-t_sa[0], o_sa*1e9, o_fem*1e9,
              (o_fem-o_sa-np.nanmean(o_fem-o_sa))*1e9)
    titles = ('SA front temperature [K]', 'FEM front temperature [K]', 'FEM - SA front [K]',
              'SA thermo-optic OPD [nm]', 'FEM thermo-optic OPD [nm]', 'FEM - SA, piston removed [nm]')
    for i, (ax, values, title) in enumerate(zip(axes.flat, fields, titles)):
        xx, yy, mask = (tx, ty, tm) if i < 3 else (ox, oy, om)
        field = np.where(mask, values, np.nan)
        if i % 3 < 2:
            offset = i//3*3
            finite = np.concatenate([fields[offset].ravel(), fields[offset+1].ravel()])
            limits = dict(vmin=np.nanmin(finite), vmax=np.nanmax(finite))
            cmap = 'viridis'
        else:
            bound = np.nanmax(np.abs(field))
            limits, cmap = dict(vmin=-bound, vmax=bound), 'coolwarm'
        mappable = ax.pcolormesh(xx*1e3, yy*1e3, field, shading='auto', cmap=cmap, **limits)
        ax.set_aspect('equal')
        ax.set_xlabel('Front coordinate x [mm]')
        ax.set_ylabel('Front coordinate y [mm]')
        ax.set_title(title)
        fig.colorbar(mappable, ax=ax)
    fig.suptitle(f'{label}: finest mesh on common polygon/optical aperture')
    fig.savefig(figure_directory(directory)/f'{label}_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--selection', type=Path, required=True, help='Completed truncation run directory')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    selection = json.loads((args.selection/'summary.json').read_text(encoding='utf-8'))
    if selection.get('kind') != 'steady_joint_truncation_selection' or selection.get('status') != 'COMPLETE':
        raise ValueError('A completed joint truncation selection is required')
    # Heat capacity was added for transients and does not change a steady run.
    selected_physics = {k:v for k,v in selection['physical_parameters_SI'].items() if k not in ('density','specific_heat')}
    current_physics = {k:v for k,v in asdict(parameters).items() if k not in ('density','specific_heat')}
    if selected_physics != current_physics or selection['aperture_radius_ratio'] != aperture_radius_ratio:
        raise ValueError('Selection and FEM physical parameters/aperture differ; rerun selection')
    modal = ModalResolution(**selection['selected_settings'])
    angles = smoke_angles_deg if args.mode == 'smoke' else full_angles_deg
    meshes = smoke_meshes if args.mode == 'smoke' else full_meshes
    count_t = smoke_temperature_xy if args.mode == 'smoke' else full_temperature_xy
    count_o = smoke_opd_xy if args.mode == 'smoke' else full_opd_xy
    degree = 1 if args.mode == 'smoke' else full_field_degree
    mirror = False if args.mode == 'smoke' else full_mirror_reduction
    if not set(angles).issubset(selection['angles_deg']):
        raise ValueError('The chosen angles have not passed modal selection')
    directory = create_run(repository_root, f'fem_{args.mode}', args.output_dir)
    started = perf_counter()
    p = parameters
    summary = {'kind': 'steady_prismatic_fem_comparison', 'status': 'RUNNING', 'mode': args.mode,
               'physical_parameters_SI': asdict(p), 'h_eff_W_per_m2_K': p.h_eff,
               'selection_run': str(args.selection.resolve()), 'modal_settings': asdict(modal),
               'angles_deg': angles, 'mesh_settings': meshes, 'aperture_radius_ratio': aperture_radius_ratio,
               'mesh_change_target': mesh_change_target, 'sanity_tolerance': sanity_tolerance,
               'evaluation_settings': {'temperature_xy': count_t, 'opd_xy': count_o},
               'field_degree': degree, 'mirror_reduction': mirror,
               'load_quadrature_order': quadrature_order, 'check_load_quadrature_order': check_quadrature_order,
               'opd_definition': 'fixed refracted ray; dn_dT * integral DeltaT ds; front entrance coordinates',
               'fem_ray_quadrature': ('split at all crossed triangle edges and axial elements; '
                                      + ('three Gauss points per quartic segment' if degree==2 else 'two Gauss points per quadratic segment')),
               'mesh_geometry': 'inscribed polygon xy disk x line layers; common domain for all meshes, no extrapolation',
               'provenance': provenance(repository_root, __file__), 'cases': []}
    if args.mode=='full':
        summary['mesh_refinement_protocol']='p2_xy_z_joint_v1'
    write_summary(directory, summary, defaults=script_defaults(globals()))
    flat_rows = []
    planes = np.array([0., p.thickness/2, p.thickness])
    axis_t = np.linspace(-p.radius, p.radius, count_t)
    tx, ty = np.meshgrid(axis_t, axis_t, indexing='xy')
    axis_o = np.linspace(-aperture_radius_ratio*p.radius, aperture_radius_ratio*p.radius, count_o)
    ox, oy = np.meshgrid(axis_o, axis_o, indexing='xy')
    for theta in angles:
        label = f'theta{theta:g}'
        internal = refracted_angle(np.deg2rad(theta), p.incident_refractive_index, p.substrate_refractive_index)
        print(f'{label}: solving {len(meshes)} independent FEM meshes', flush=True)
        solutions = []
        def flux(x, y):
            return oblique_gaussian_heat_flux(x, y, p.beam_radius, np.deg2rad(theta), p.absorbed_power)
        for level, layers in meshes:
            solution = solve_steady_fem(p.radius, p.thickness, p.conductivity, p.h_eff, p.h_eff, p.h_eff, flux,
                                        disk_refinements=level, axial_layers=layers, quadrature_order=quadrature_order,
                                        solver='direct' if args.mode == 'smoke' else 'cg', rtol=solver_rtol,
                                        field_degree=degree,mirror_reduction=mirror)
            solutions.append(solution)
            print(f"  disk={level}, z={layers}, DOF={solution.diagnostics['degrees_of_freedom']}, solve={solution.diagnostics['solve_seconds']:.3f}s", flush=True)
            saved_nodes={'disk_nodes_xy_m':solution.disk_mesh.p,'disk_triangles':solution.disk_mesh.t,
                         'axial_nodes_z_m':solution.z_m,'temperature_K':solution.temperature_K,
                         'field_degree':solution.field_degree,
                         'xy_dof_coordinates_m':(Basis(solution.disk_mesh,ElementTriP2()).doflocs
                                                 if degree==2 else solution.disk_mesh.p)}
            np.savez_compressed(data_directory(directory)/f'{label}_mesh{len(solutions)-1}.npz',**saved_nodes)
        tm = tx**2+ty**2 <= p.radius**2
        physical_mask = circular_aperture_mask(ox, oy, aperture_radius_ratio*p.radius)
        physical_mask &= transmitted_ray_mask(ox, oy, p.radius, p.thickness, internal)
        om = physical_mask.copy()
        for solution in solutions:
            tm &= solution.transmitted_mask(tx, ty, 0.)
            om &= solution.transmitted_mask(ox, oy, internal)
        if not np.any(tm) or not np.any(om):
            raise RuntimeError('Common FEM evaluation domain is empty')
        prepared = ThermalModel(p,theta,modal).prepare()
        coefficients,roots = prepared.coefficients,prepared.roots
        steady = prepared.solve_steady()
        thermal_args = (coefficients, roots, p.conductivity, p.thickness, p.h_eff, p.h_eff)
        tick = perf_counter()
        sa_t = np.stack([steady.temperature(np.hypot(tx[tm],ty[tm]),np.arctan2(ty[tm],tx[tm]),z) for z in planes])
        sa_o = steady.opd(ox[om],oy[om],point_chunk_size=point_chunk_size)
        sa_evaluation_seconds = perf_counter()-tick
        def sample(solution):
            t = np.stack([solution.temperature_at(tx[tm], ty[tm], z) for z in planes])
            o = solution.refracted_opd(ox[om],oy[om],internal,p.dn_dT,point_chunk_size=point_chunk_size)
            return t, o
        fields, sampling_seconds = [], []
        for solution in solutions:
            tick = perf_counter()
            fields.append(sample(solution))
            sampling_seconds.append(perf_counter()-tick)
        fine_t, fine_o = fields[-1]
        rows = []
        for j, (solution, (t, o)) in enumerate(zip(solutions, fields)):
            errors = field_errors(t, sa_t, np.ones(sa_t.shape[1]), o, sa_o)
            changes = field_errors(t, fine_t, np.ones(sa_t.shape[1]), o, fine_o)
            row = {'mesh_index': j, **solution.diagnostics, 'fem_evaluation_seconds': sampling_seconds[j],
                   'fem_opd_pv_common_aperture_m': float(np.ptp(o)), 'errors_vs_sa': errors,
                   'changes_vs_finest_fem': changes}
            if args.mode=='full':
                row['mesh_role']=full_mesh_roles[j]
            rows.append(row)
            # Preserve scalar columns and serialize nested mirror diagnostics as
            # JSON; a dict must never expand into mismatched CSV columns.
            flat = {'angle_deg': theta, 'mesh_index': j,
                    **{key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list,tuple)) else value
                       for key,value in solution.diagnostics.items()},
                    'fem_opd_pv_common_aperture_m': float(np.ptp(o))}
            for metric, value in errors.items():
                if isinstance(value, list):
                    for plane_name, number in zip(('front', 'mid', 'rear'), value):
                        flat[metric+'_'+plane_name] = number
                else:
                    flat[metric] = value
            flat_rows.append(flat)
            print(f"  mesh {j}: front L2 vs SA={errors['temperature_relative_l2'][0]:.4%}, OPD PV error={errors['opd_relative_pv']:.4%}", flush=True)
        # Source-load quadrature is checked independently on the finest mesh.
        level, layers = meshes[-1]
        higher_q = solve_steady_fem(p.radius, p.thickness, p.conductivity, p.h_eff, p.h_eff, p.h_eff, flux,
                                    disk_refinements=level, axial_layers=layers, quadrature_order=check_quadrature_order,
                                    solver='direct' if args.mode == 'smoke' else 'cg', rtol=solver_rtol,
                                    field_degree=degree,mirror_reduction=mirror)
        higher_t, higher_o = sample(higher_q)
        q_errors = field_errors(higher_t, fine_t, np.ones(sa_t.shape[1]), higher_o, fine_o)
        case = {'angle_deg': theta, 'refraction_angle_deg': float(np.rad2deg(internal)),
                'temperature_common_sample_count': int(tm.sum()), 'opd_physical_sample_count': int(physical_mask.sum()),
                'opd_common_sample_count': int(om.sum()), 'opd_polygon_excluded_count': int((physical_mask & ~om).sum()),
                'sa_evaluation_seconds': sa_evaluation_seconds, 'sa_opd_pv_common_aperture_m': float(np.ptp(sa_o)),
                'meshes': rows, 'load_quadrature_check': {'order': check_quadrature_order, 'errors': q_errors,
                                                          'passed': within_tolerance(q_errors, mesh_change_target/10)},
                'finest_mesh_change_within_target': within_tolerance(rows[-2]['changes_vs_finest_fem'], mesh_change_target)}
        if args.mode=='full':
            case['directional_mesh_checks']=steady_fem_directional_checks(rows,
                np.asarray([t for t,_ in fields]),np.asarray([o for _,o in fields]),mesh_change_target,
                mesh_coordinates=[dict(xy=s.disk_mesh.p,tri=s.disk_mesh.t,z=s.z_m) for s in solutions])
        def scatter(values, mask):
            shaped = np.full((*values.shape[:-1], *mask.shape), np.nan)
            shaped[..., mask] = values
            return shaped
        t_sa_grid, t_fem_grid = scatter(sa_t, tm), scatter(fine_t, tm)
        o_sa_grid, o_fem_grid = scatter(sa_o, om), scatter(fine_o, om)
        np.savez_compressed(data_directory(directory)/f'{label}_fields.npz', temperature_x_m=tx, temperature_y_m=ty,
                            temperature_z_m=planes, temperature_mask=tm, temperature_sa_K=t_sa_grid,
                            temperature_fem_K=np.stack([scatter(t, tm) for t, _ in fields]),
                            temperature_higher_quadrature_K=scatter(higher_t, tm),
                            opd_x_m=ox, opd_y_m=oy, opd_physical_mask=physical_mask, opd_common_mask=om,
                            opd_sa_m=o_sa_grid, opd_fem_m=np.stack([scatter(o, om) for _, o in fields]),
                            opd_higher_quadrature_m=scatter(higher_o, om))
        save_figure(directory, label, tx, ty, tm, t_sa_grid, t_fem_grid, ox, oy, om, o_sa_grid, o_fem_grid)
        summary['cases'].append(case)
        write_summary(directory, summary, defaults=script_defaults(globals()))
    sanity_pass = all(c['load_quadrature_check']['passed'] and all(
        r['relative_energy_imbalance'] <= sanity_tolerance and r['relative_linear_residual'] <= sanity_tolerance
        for r in c['meshes']) for c in summary['cases'])
    summary['status'] = 'COMPLETE' if sanity_pass else 'FAILED'
    summary['sanity_pass'] = sanity_pass
    summary['mesh_convergence_established'] = args.mode == 'full' and all(
        c['finest_mesh_change_within_target'] and all(check['passed'] for check in c['directional_mesh_checks'].values())
        for c in summary['cases'])
    summary['elapsed_seconds'] = perf_counter()-started
    summary['limits'] = 'Smoke tests the P1 implementation and coarse-mesh trend only. Full uses P2 fields and separate xy/z/joint refinement gates; configurations alone do not establish convergence. FEM polygon geometry and field discretization both change with xy refinement. Field errors/PV are on a recorded common subset, not the full physical aperture. Timings are component timings, not an end-to-end speedup claim.'
    write_summary(directory, summary, defaults=script_defaults(globals()))
    with (data_directory(directory)/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in flat_rows for key in row)))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"{summary['status']}: {directory}; final mesh convergence={summary['mesh_convergence_established']}", flush=True)
    if not sanity_pass:
        raise RuntimeError('FEM sanity or load quadrature checks failed')


if __name__ == '__main__':
    main()
