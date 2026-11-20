"""Vary one numerical setting at a time for temperature and refracted-ray OPD.

The finest case in each sweep is a numerical reference, not an exact solution.
This script does not compare against FEM or impose an accuracy threshold.
"""

from thermal_opd.run_io import script_defaults
from thermal_opd.run_io import data_directory
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from thermal_opd.model import PreparedModel, ModalResolution, project_modal_source
from thermal_opd.run_io import create_run,figure_directory,provenance,write_summary,parse_output_directory
import csv

import matplotlib.pyplot as plt
import numpy as np

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.metrics import circular_aperture_mask
from thermal_opd.opd import refracted_angle, transmitted_ray_mask, steady_refracted_opd
from thermal_opd.parameters import PhysicalParameters


parameters = PhysicalParameters()
incidence_angle_deg = 60.0
aperture_radius_ratio = 0.90

# Numerical settings belong to this experiment; all internal units are SI.
azimuthal_orders = (0, 2, 4, 6, 8, 10)
radial_mode_counts = (6, 8, 12, 16, 20)
projection_r_counts = (100, 200, 400, 800, 1200)
projection_phi_counts = (24, 32, 64, 128, 256)
opd_xy_counts = (80, 120, 160, 201, 301)
opd_ray_point_counts = (8, 16, 24, 32)
opd_point_chunk_size = 1024
num_temperature_r = 161
num_temperature_phi = 180
common_num_opd_xy = 201

repository_root = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    M: int
    N: int
    projection_r: int
    projection_phi: int
    opd_xy: int
    ray_points: int


def aperture_coordinates(count):
    radius = aperture_radius_ratio * parameters.radius
    axis = np.linspace(-radius, radius, count)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    internal = refracted_angle(np.deg2rad(incidence_angle_deg), parameters.incident_refractive_index, parameters.substrate_refractive_index)
    mask = circular_aperture_mask(x, y, radius) & transmitted_ray_mask(x, y, parameters.radius, parameters.thickness, internal)
    return x[mask], y[mask]


def relative_disk_l2(fields, reference, r):
    """Per-plane relative L2 change with disk area weight r dr dphi."""
    numerator = np.trapezoid(np.mean((fields - reference)**2, axis=2) * r, x=r, axis=1)
    denominator = np.trapezoid(np.mean(reference**2, axis=2) * r, x=r, axis=1)
    return np.sqrt(numerator / denominator)


def main():
    result_directory=create_run(repository_root,'convergence_diagnostic',parse_output_directory(__doc__))
    figure_path=figure_directory(result_directory)/'convergence.png'
    result_path=result_directory/'summary.json'
    p = parameters
    baseline = Settings(
        max(azimuthal_orders), max(radial_mode_counts),
        max(projection_r_counts), max(projection_phi_counts), common_num_opd_xy,
        max(opd_ray_point_counts),
    )
    sweeps = {
        "azimuthal_order": ("M", azimuthal_orders),
        "radial_modes": ("N", radial_mode_counts),
        "projection_radial": ("projection_r", projection_r_counts),
        "projection_angular": ("projection_phi", projection_phi_counts),
        "opd_sampling": ("opd_xy", opd_xy_counts),
        "ray_quadrature": ("ray_points", opd_ray_point_counts),
    }
    if min(projection_phi_counts) <= 2 * baseline.M:
        raise ValueError("Every angular projection grid must contain more than 2*M samples")
    print(f"Temperature / refracted-ray OPD convergence at {incidence_angle_deg:g} deg", flush=True)
    print(f"Fixed settings before each single-parameter sweep: {asdict(baseline)}", flush=True)
    roots = {
        m: radial_eigenvalues(m, p.radius, p.conductivity, p.h_eff, baseline.N)
        for m in range(0, baseline.M + 1, 2)
    }
    r = np.linspace(0.0, p.radius, num_temperature_r)
    phi = np.linspace(0.0, 2 * np.pi, num_temperature_phi, endpoint=False)
    z_planes = (0.0, p.thickness / 2, p.thickness)
    thermal_args = (p.conductivity, p.thickness, p.h_eff, p.h_eff)
    opd_args = (*thermal_args, p.dn_dT)
    internal = refracted_angle(np.deg2rad(incidence_angle_deg), p.incident_refractive_index, p.substrate_refractive_index)
    ray_geometry = dict(radius=p.radius, refraction_angle=internal, point_chunk_size=opd_point_chunk_size)
    common_x, common_y = aperture_coordinates(common_num_opd_xy)

    # OPD is linear in source coefficients. Cache unit responses on the common
    # evaluation grid, reusing the production solver rather than a new formula.
    opd_basis_cache = {}

    def opd_basis(ray_points):
        if ray_points not in opd_basis_cache:
            basis = {}
            for m, values in roots.items():
                print(f"Preparing OPD responses: {ray_points} ray points, m={m}", flush=True)
                basis[m] = np.stack([
                    steady_refracted_opd(common_x, common_y, {m: np.array([1.0])}, {m: np.array([lam])},
                                         *opd_args, **ray_geometry, num_ray_points=ray_points)
                    for lam in values
                ])
            opd_basis_cache[ray_points] = basis
        return opd_basis_cache[ray_points]
    projection_cache = {}
    solution_cache = {}
    evaluation_cache = {}

    def solve(settings):
        key = (settings.M, settings.N, settings.projection_r, settings.projection_phi)
        if key in solution_cache:
            return solution_cache[key]
        grid_key = (settings.projection_r, settings.projection_phi)
        if grid_key not in projection_cache:
            projection_cache[grid_key]=project_modal_source(p,incidence_angle_deg,roots,num_r=settings.projection_r,num_phi=settings.projection_phi)
        selected_roots = {m: values[:settings.N] for m, values in roots.items() if m <= settings.M}
        coefficients = {m: projection_cache[grid_key][m][:settings.N] for m in selected_roots}
        solution=PreparedModel(p,incidence_angle_deg,ModalResolution(settings.M,settings.N,settings.projection_r,settings.projection_phi,settings.ray_points),coefficients,selected_roots).solve_steady()
        fields=np.stack([solution.temperature(r[:,None],phi[None,:],z) for z in z_planes])
        solution_cache[key] = coefficients, selected_roots, fields
        return solution_cache[key]

    def evaluate(settings):
        if settings in evaluation_cache:
            return evaluation_cache[settings]
        coefficients, selected_roots, fields = solve(settings)
        if settings.opd_xy == common_num_opd_xy:
            opd = np.zeros_like(common_x)
            basis = opd_basis(settings.ray_points)
            for m, values in coefficients.items():
                opd += np.sum(values[:, None] * basis[m][:settings.N], axis=0)
        else:
            evaluation_x, evaluation_y = aperture_coordinates(settings.opd_xy)
            opd = steady_refracted_opd(evaluation_x, evaluation_y, coefficients, selected_roots,
                                       *opd_args, **ray_geometry, num_ray_points=settings.ray_points)
        record = {
            **asdict(settings),
            "opd_pv_m": float(np.ptp(opd)),
            "aperture_sample_count": int(opd.size),
            "center_temperature_rises_K": fields[:, 0, 0].tolist(),
            "sampled_peak_temperature_rises_K": fields.max(axis=(1, 2)).tolist(),
        }
        evaluation_cache[settings] = record, fields
        return record, fields

    # Check cache summation against an ordinary production call at several points.
    check_coefficients, check_roots, _ = solve(baseline)
    indices = np.linspace(0, common_x.size - 1, 9, dtype=int)
    expected = steady_refracted_opd(common_x[indices], common_y[indices], check_coefficients, check_roots,
                                    *opd_args, **ray_geometry, num_ray_points=baseline.ray_points)
    basis = opd_basis(baseline.ray_points)
    cached = sum(np.sum(values[:, None] * basis[m][:baseline.N, indices], axis=0)
                 for m, values in check_coefficients.items())
    np.testing.assert_allclose(cached, expected, rtol=1e-12, atol=1e-20)
    cache_max_difference = float(np.max(np.abs(cached - expected)))
    groups = {}
    csv_records = []
    for name, (setting_name, counts) in sweeps.items():
        reference_settings = replace(baseline, **{setting_name: counts[-1]})
        reference, reference_fields = evaluate(reference_settings)
        rows = []
        previous = None
        print(f"\n{name}: value, OPD-PV [nm], change vs finest, front-temperature L2 change", flush=True)
        for count in counts:
            record, fields = evaluate(replace(baseline, **{setting_name: count}))
            row = dict(record)
            row["varied_value"] = count
            row["relative_pv_change_vs_reference"] = abs(record["opd_pv_m"] - reference["opd_pv_m"]) / abs(reference["opd_pv_m"])
            row["temperature_relative_disk_l2_vs_reference"] = relative_disk_l2(fields, reference_fields, r).tolist()
            row["relative_pv_change_vs_previous"] = (
                None if previous is None else abs(record["opd_pv_m"] - previous) / abs(record["opd_pv_m"])
            )
            previous = record["opd_pv_m"]
            rows.append(row)
            print(f"{count:4d}  {row['opd_pv_m']*1e9:13.6f}  {row['relative_pv_change_vs_reference']:.6e}  {row['temperature_relative_disk_l2_vs_reference'][0]:.6e}", flush=True)
            flat_row = {k: v for k, v in row.items() if not isinstance(v, list)}
            flat_row["sweep"] = name
            for label, i in zip(("front", "mid", "rear"), range(3)):
                flat_row[f"{label}_center_temperature_K"] = row["center_temperature_rises_K"][i]
                flat_row[f"{label}_peak_temperature_K"] = row["sampled_peak_temperature_rises_K"][i]
                flat_row[f"{label}_relative_disk_l2_vs_reference"] = row["temperature_relative_disk_l2_vs_reference"][i]
            csv_records.append(flat_row)
        groups[name] = {"varied_setting": setting_name, "reference_settings": asdict(reference_settings), "cases": rows}

    summary = {
        "kind": "single_parameter_temperature_and_refracted_ray_opd_sweeps",
        "physical_parameters_SI": asdict(p),
        "h_eff_W_per_m2_K": p.h_eff,
        "incidence_angle_deg": incidence_angle_deg,
        "aperture_radius_ratio": aperture_radius_ratio,
        "opd_definition": "dn_dT / cos(theta_t) * integral_0^L DeltaT(x0,y0+z*tan(theta_t),z) dz; front-face coordinates; unit m",
        "refraction_angle_deg": float(np.rad2deg(internal)),
        "rear_y_shift_m": float(p.thickness*np.tan(internal)),
        "opd_point_chunk_size": opd_point_chunk_size,
        "aperture_convention": "0.90R entrance aperture intersected with rays reaching rear face inside R; excludes side-exiting rays",
        "temperature_evaluation": {"r": num_temperature_r, "phi": num_temperature_phi, "z_m": z_planes},
        "reference_convention": "last (finest) case in each sweep; not an exact or FEM reference",
        "limits": "one angle; fixed temperature evaluation grid; finite reference rays and quadrature; sampling of PV extrema can mask field changes; no acceptance threshold",
        "opd_cache_max_absolute_difference_m": cache_max_difference,
        "sweeps": groups,
    }
    result_directory.mkdir(parents=True, exist_ok=True)
    summary.update(status="COMPLETE",physical_parameters_SI=asdict(parameters),provenance=provenance(repository_root,__file__))
    write_summary(result_directory, summary, defaults=script_defaults(globals()))
    with (data_directory(result_directory) / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")
    for ax, (name, group) in zip(axes.flat, groups.items()):
        rows = group["cases"]
        is_quadrature = name == "ray_quadrature"
        values = ([row["relative_pv_change_vs_reference"] for row in rows] if is_quadrature
                  else [row["opd_pv_m"]*1e9 for row in rows])
        ax.plot([row["varied_value"] for row in rows], values, "o-")
        ax.set_xlabel(group["varied_setting"])
        ax.set_ylabel("Relative PV change vs finest" if is_quadrature else "OPD-PV [nm]")
        ax.set_title(name.replace("_", " "))
        ax.ticklabel_format(axis="y", style="sci" if is_quadrature else "plain", scilimits=(0, 0), useOffset=False)
        ax.grid(alpha=0.3)
    ax = axes.flat[-1]
    for name in ("azimuthal_order", "radial_modes"):
        rows = groups[name]["cases"][:-1]  # Reference change is zero, undefined on log axis.
        ax.semilogy([row["varied_value"] for row in rows],
                    [row["temperature_relative_disk_l2_vs_reference"][0] for row in rows],
                    "o-", label=groups[name]["varied_setting"])
    ax.set_xlabel("M or N")
    ax.set_ylabel("Front-temperature relative disk L2 change")
    ax.set_title("Change against each finest case")
    ax.legend()
    ax.grid(alpha=0.3)
    axes.flat[6].axis("off")
    axes.flat[6].text(0.0, 0.8, f"Front-face ray coordinates\nInternal angle: {np.rad2deg(internal):.4f} deg\nRear shift: {p.thickness*np.tan(internal)*1e3:.4f} mm\n{common_x.size} rear-transmitted rays\nNumerical references, no threshold", va="top")
    fig.suptitle(f"Single-parameter sweeps at {incidence_angle_deg:g} deg; numerical references only")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"\nFigure: {figure_path}\nResults: {result_directory}", flush=True)


if __name__ == "__main__":
    main()
