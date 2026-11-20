"""Select a jointly verified steady modal resolution at 0/30/60 degrees.

Run from any directory after pip install -e '.[dev]'. No full FEM is launched.
The 0.1% criterion concerns finite numerical references, not physical accuracy.
"""

from thermal_opd.run_io import script_defaults
from thermal_opd.model import ThermalModel
from thermal_opd.run_io import data_directory
import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss

from thermal_opd.convergence import field_errors, within_tolerance, continuous_aperture_pv
from thermal_opd.metrics import circular_aperture_mask
from thermal_opd.model import ModalResolution
from thermal_opd.opd import refracted_angle, transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import create_run, provenance, write_summary


parameters = PhysicalParameters()
angles_deg = (0., 30., 60.)
tolerance = 1e-3
reference_tolerance = tolerance/5
aperture_radius_ratio = 0.9
reference = ModalResolution(14, 32, 2400, 256, 64)
enlarged_reference = ModalResolution(16, 40, 3200, 512, 80)
candidates = (ModalResolution(6, 8, 400, 32, 16),
              ModalResolution(8, 12, 800, 64, 16),
              ModalResolution(10, 20, 1200, 128, 32))
num_temperature_r = 96
num_temperature_phi = 128
num_comparison_xy = 61
num_verification_xy = 121
figure_xy_counts = (121, 201, 301)
num_pv_seed_radius = 9
num_pv_seed_phi = 48
point_chunk_size = 512
repository_root = Path(__file__).resolve().parents[2]


def optical_coordinates(p, angle, count):
    axis = np.linspace(-aperture_radius_ratio*p.radius, aperture_radius_ratio*p.radius, count)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    mask = circular_aperture_mask(x, y, aperture_radius_ratio*p.radius)
    mask &= transmitted_ray_mask(x, y, p.radius, p.thickness, angle)
    return x[mask], y[mask]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    directory = create_run(repository_root, 'truncation', args.output_dir)
    started = perf_counter()
    p = parameters
    summary = {'kind': 'steady_joint_truncation_selection', 'status': 'RUNNING',
               'run_directory': str(directory), 'physical_parameters_SI': asdict(p),
               'h_eff_W_per_m2_K': p.h_eff, 'angles_deg': angles_deg,
               'tolerance': tolerance, 'reference_stability_tolerance': reference_tolerance,
               'reference': asdict(reference), 'enlarged_reference': asdict(enlarged_reference),
               'aperture_radius_ratio': aperture_radius_ratio,
               'provenance': provenance(repository_root, __file__), 'cases': []}
    write_summary(directory, summary, defaults=script_defaults(globals()))
    nodes, weights = leggauss(num_temperature_r)
    r, wr = (nodes+1)*p.radius/2, weights*p.radius/2
    phi = np.linspace(0, 2*np.pi, num_temperature_phi, endpoint=False)
    rt, phit = np.broadcast_arrays(r[:, None], phi[None, :])
    area_weights = np.broadcast_to((wr*r)[:, None], rt.shape).ravel()
    planes = (0., p.thickness/2, p.thickness)
    saved = {'temperature_r_m': rt.ravel(), 'temperature_phi_rad': phit.ravel(),
             'temperature_area_weights': area_weights, 'temperature_z_m': planes}
    retained = []
    for theta in angles_deg:
        print(f'Angle {theta:g}: preparing high-order references', flush=True)
        internal = refracted_angle(np.deg2rad(theta), p.incident_refractive_index, p.substrate_refractive_index)
        xy = optical_coordinates(p, internal, num_comparison_xy)
        def evaluate(resolution):
            tick = perf_counter()
            prepared = ThermalModel(p,theta,resolution).prepare()
            coefficients,roots = prepared.coefficients,prepared.roots
            steady = prepared.solve_steady()
            print(f'  source projections ready ({perf_counter()-tick:.2f}s)', flush=True)
            thermal = (coefficients, roots, p.conductivity, p.thickness, p.h_eff, p.h_eff)
            temperatures = np.stack([steady.temperature(rt.ravel(),phit.ravel(),z) for z in planes])
            opd = steady.prepare_opd(point_chunk_size=point_chunk_size)
            optical = opd(*xy)
            print(f'  sampled fields ready ({perf_counter()-tick:.2f}s); searching PV', flush=True)
            extrema = continuous_aperture_pv(opd, radius=p.radius, thickness=p.thickness,
                                             refraction_angle=internal, aperture_radius=aperture_radius_ratio*p.radius,
                                             num_seed_radius=num_pv_seed_radius, num_seed_phi=num_pv_seed_phi)
            return temperatures, optical, extrema, opd, perf_counter()-tick
        ref_t, ref_o, ref_pv, ref_evaluator, _ = evaluate(reference)
        bigger_t, bigger_o, bigger_pv, _, _ = evaluate(enlarged_reference)
        stable = field_errors(ref_t, bigger_t, area_weights, ref_o, bigger_o)
        stable['opd_relative_pv'] = abs(ref_pv['pv']-bigger_pv['pv'])/bigger_pv['pv']
        reference_pass = within_tolerance(stable, reference_tolerance)
        case = {'angle_deg': theta, 'refraction_angle_deg': float(np.rad2deg(internal)),
                'reference_stability_errors': stable, 'reference_pass': reference_pass,
                'reference_extrema_m': ref_pv, 'enlarged_reference_extrema_m': bigger_pv, 'candidates': []}
        selected_here = []
        for candidate in candidates:
            print(f'Angle {theta:g}: candidate {asdict(candidate)}', flush=True)
            t, o, pv, evaluator, seconds = evaluate(candidate)
            errors = field_errors(t, ref_t, area_weights, o, ref_o)
            errors['opd_relative_pv'] = abs(pv['pv']-ref_pv['pv'])/ref_pv['pv']
            passed = within_tolerance(errors, tolerance)
            case['candidates'].append({'settings': asdict(candidate), 'errors': errors,
                                        'accepted': passed, 'extrema_m': pv, 'seconds': seconds})
            selected_here.append((passed, t, o, pv, evaluator))
            print(f"  accepted={passed}; max error={max(np.max(v) for v in errors.values()):.4e}; PV={pv['pv']*1e9:.6f} nm", flush=True)
        retained.append((internal, ref_t, ref_evaluator, ref_pv, selected_here))
        saved[f'theta{theta:g}_reference_temperature_K'] = ref_t
        saved[f'theta{theta:g}_reference_opd_m'] = ref_o
        saved[f'theta{theta:g}_opd_x_m'], saved[f'theta{theta:g}_opd_y_m'] = xy
        summary['cases'].append(case)
        write_summary(directory, summary, defaults=script_defaults(globals()))
    feasible = [i for i in range(len(candidates)) if all(c['reference_pass'] and c['candidates'][i]['accepted'] for c in summary['cases'])]
    if not feasible:
        summary['status'] = 'FAILED'
        write_summary(directory, summary, defaults=script_defaults(globals()))
        raise RuntimeError(f'No candidate passed all criteria: {directory}')
    index = feasible[0]
    selected = candidates[index]
    summary['selected_settings'] = asdict(selected)
    sampling = []
    for case, (internal, ref_t, ref_eval, ref_pv, evaluated) in zip(summary['cases'], retained):
        _, selected_t, selected_o, selected_pv, selected_eval = evaluated[index]
        fine_xy = optical_coordinates(p, internal, num_verification_xy)
        fine_ref, fine_selected = ref_eval(*fine_xy), selected_eval(*fine_xy)
        errors = field_errors(selected_t, ref_t, area_weights, fine_selected, fine_ref)
        errors['opd_relative_pv'] = abs(selected_pv['pv']-ref_pv['pv'])/ref_pv['pv']
        case['denser_common_grid_errors'] = errors
        case['denser_common_grid_pass'] = within_tolerance(errors, tolerance)
        rows = []
        for count in figure_xy_counts:
            sampled = selected_eval(*optical_coordinates(p, internal, count))
            rows.append({'xy_count': count, 'sampled_pv_m': float(np.ptp(sampled)),
                         'relative_pv_sampling_error': float(abs(np.ptp(sampled)-selected_pv['pv'])/selected_pv['pv'])})
        sampling.append({'angle_deg': case['angle_deg'], 'cases': rows})
        label = f"theta{case['angle_deg']:g}"
        saved[label+'_selected_temperature_K'] = selected_t
        saved[label+'_selected_opd_m'] = selected_o
        saved[label+'_verification_opd_x_m'], saved[label+'_verification_opd_y_m'] = fine_xy
        saved[label+'_verification_reference_opd_m'] = fine_ref
        saved[label+'_verification_selected_opd_m'] = fine_selected
    acceptable_grids = [count for count in figure_xy_counts if all(next(r for r in c['cases'] if r['xy_count']==count)['relative_pv_sampling_error'] <= tolerance for c in sampling)]
    summary['sampling_checks'] = sampling
    summary['selected_figure_xy'] = min(acceptable_grids) if acceptable_grids else None
    summary['evaluation_settings'] = {'temperature_r_gauss': num_temperature_r, 'temperature_phi': num_temperature_phi,
                                      'opd_xy': num_comparison_xy, 'verification_xy': num_verification_xy,
                                      'extrema_seed_radius': num_pv_seed_radius, 'extrema_seed_phi': num_pv_seed_phi}
    summary['limits'] = 'Finite numerical reference; sampled field norms; multistart extrema not a certified global bound. Scope: stated material, w0/R=0.30, 0/30/60 deg, steady only. No FEM accuracy claim.'
    summary['status'] = 'COMPLETE' if acceptable_grids and all(c['denser_common_grid_pass'] for c in summary['cases']) else 'FAILED'
    summary['elapsed_seconds'] = perf_counter()-started
    np.savez_compressed(data_directory(directory)/'fields.npz', **saved)
    write_summary(directory, summary, defaults=script_defaults(globals()))
    print(f"{summary['status']}: {directory}; selected={asdict(selected)}; figure xy={summary['selected_figure_xy']}", flush=True)
    if summary['status'] != 'COMPLETE':
        raise RuntimeError('Verification or figure sampling failed')


if __name__ == '__main__':
    main()
