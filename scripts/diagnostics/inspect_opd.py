"""Inspect refracted-ray thermo-optic OPD on front-face entrance coordinates."""

from thermal_opd.run_io import script_defaults
from thermal_opd.run_io import data_directory
from dataclasses import asdict
from pathlib import Path
from thermal_opd.model import PreparedModel, ModalResolution, project_modal_source
from thermal_opd.run_io import create_run,figure_directory,provenance,write_summary,parse_output_directory

import matplotlib.pyplot as plt
import numpy as np

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.metrics import circular_aperture_mask, opd_peak_to_valley
from thermal_opd.opd import refracted_angle, transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters


parameters = PhysicalParameters()
incidence_angles_deg = (0.0, 30.0, 60.0)
aperture_radius_ratio = 0.90

# Numerical settings for this inspection, defined in the calling code.
max_azimuthal_order = 8
num_radial_modes = 12
num_projection_r = 400
num_projection_phi = 256
num_opd_xy = 201  # Odd count includes the center and axis/aperture intersections.
num_opd_ray_points = 32
opd_point_chunk_size = 1024
num_profile_points = 401

# A small sensitivity comparison, not the full convergence study.
finer_num_projection_r = 800
higher_max_azimuthal_order = 10
higher_num_radial_modes = 16

repository_root = Path(__file__).resolve().parents[2]


def source_coefficients(angle_deg, eigenvalues, num_r, num_phi):
    return project_modal_source(parameters,angle_deg,eigenvalues,num_r=num_r,num_phi=num_phi)


def opd_at(x0, y0, coefficients, eigenvalues, angle_deg):
    p = parameters
    internal = refracted_angle(np.deg2rad(angle_deg), p.incident_refractive_index, p.substrate_refractive_index)
    x0, y0 = np.broadcast_arrays(x0, y0)
    valid = transmitted_ray_mask(x0, y0, p.radius, p.thickness, internal)
    output = np.full(x0.shape, np.nan)
    solution=PreparedModel(p,angle_deg,ModalResolution(max(eigenvalues),max(len(v) for v in eigenvalues.values()),num_projection_r,num_projection_phi,num_opd_ray_points),coefficients,eigenvalues).solve_steady()
    output[valid]=solution.opd(x0[valid],y0[valid],point_chunk_size=opd_point_chunk_size)
    return output


def main():
    result_directory=create_run(repository_root,'opd_inspection',parse_output_directory(__doc__))
    figure_path=figure_directory(result_directory)/'refracted_opd.png'
    result_path=result_directory/'summary.json'
    p = parameters
    aperture_radius = aperture_radius_ratio * p.radius
    axis = np.linspace(-aperture_radius, aperture_radius, num_opd_xy)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    entrance_mask = circular_aperture_mask(X, Y, aperture_radius)
    eigenvalues = {
        m: radial_eigenvalues(m, p.radius, p.conductivity, p.h_eff, num_radial_modes)
        for m in range(0, max_azimuthal_order + 1, 2)
    }
    records = []
    opd_maps = []
    transmitted_masks = []
    case_coefficients = []
    print("Refracted-ray substrate thermo-optic OPD; front-face coordinates; units m", flush=True)
    print(f"Aperture: {aperture_radius_ratio:.2f} R = {aperture_radius * 1e3:.3f} mm")
    print(f"M={max_azimuthal_order}, N={num_radial_modes}, projection={num_projection_r}x{num_projection_phi}")
    print(f"Cartesian sampling: {num_opd_xy}x{num_opd_xy}, {entrance_mask.sum()} entrance aperture points")
    print("angle [deg]    center OPD [nm]    aperture OPD-PV [nm]")
    for angle_deg in incidence_angles_deg:
        internal = refracted_angle(np.deg2rad(angle_deg), p.incident_refractive_index, p.substrate_refractive_index)
        mask = entrance_mask & transmitted_ray_mask(X, Y, p.radius, p.thickness, internal)
        print(f"Evaluating {angle_deg:g} deg, internal {np.rad2deg(internal):.6f} deg, {mask.sum()} transmitted rays", flush=True)
        coefficients = source_coefficients(angle_deg, eigenvalues, num_projection_r, num_projection_phi)
        field = np.full(X.shape, np.nan)
        field[mask] = opd_at(X[mask], Y[mask], coefficients, eigenvalues, angle_deg)
        pv = opd_peak_to_valley(field, mask)
        center = float(opd_at(0.0, 0.0, coefficients, eigenvalues, angle_deg))
        records.append({
            "incidence_angle_deg": angle_deg,
            "refraction_angle_deg": float(np.rad2deg(internal)),
            "transmitted_sample_count": int(mask.sum()),
            "excluded_side_exit_sample_count": int(np.count_nonzero(entrance_mask & ~mask)),
            "center_opd_m": center,
            "opd_pv_m": pv,
            "source_coefficients_W_per_m2": {str(m): c.tolist() for m, c in coefficients.items()},
        })
        opd_maps.append(field)
        transmitted_masks.append(mask)
        case_coefficients.append(coefficients)
        print(f"{angle_deg:11.1f}    {center*1e9:15.6f}    {pv*1e9:20.6f}")

    # Compare the last case on the same Cartesian aperture coordinates.
    selected_angle = incidence_angles_deg[-1]
    selected_field = opd_maps[-1]
    selected_coefficients = case_coefficients[-1]
    mask = transmitted_masks[-1]
    base_pv = records[-1]["opd_pv_m"]
    fine_coefficients = source_coefficients(
        selected_angle, eigenvalues, finer_num_projection_r, num_projection_phi
    )
    fine_values = opd_at(X[mask], Y[mask], fine_coefficients, eigenvalues, selected_angle)
    fine_pv = float(np.ptp(fine_values))
    higher_roots = {
        m: radial_eigenvalues(m, p.radius, p.conductivity, p.h_eff, higher_num_radial_modes)
        for m in range(0, higher_max_azimuthal_order + 1, 2)
    }
    higher_coefficients = source_coefficients(
        selected_angle, higher_roots, finer_num_projection_r, num_projection_phi
    )
    higher_values = opd_at(X[mask], Y[mask], higher_coefficients, higher_roots, selected_angle)
    higher_pv = float(np.ptp(higher_values))
    sensitivity = {
        "incidence_angle_deg": selected_angle,
        "evaluation_xy": num_opd_xy,
        "base_pv_m": base_pv,
        "fine_projection": {
            "M": max_azimuthal_order, "N": num_radial_modes,
            "projection_r": finer_num_projection_r, "projection_phi": num_projection_phi,
            "pv_m": fine_pv, "relative_pv_difference": abs(fine_pv - base_pv) / abs(fine_pv),
        },
        "higher_modes_and_fine_projection": {
            "M": higher_max_azimuthal_order, "N": higher_num_radial_modes,
            "projection_r": finer_num_projection_r, "projection_phi": num_projection_phi,
            "pv_m": higher_pv, "relative_pv_difference": abs(higher_pv - base_pv) / abs(higher_pv),
        },
    }
    print(f"Fine-projection OPD-PV: {fine_pv*1e9:.6f} nm; relative change={sensitivity['fine_projection']['relative_pv_difference']:.6e}")
    print(f"Higher-mode OPD-PV: {higher_pv*1e9:.6f} nm; relative change={sensitivity['higher_modes_and_fine_projection']['relative_pv_difference']:.6e}")

    # Remove one constant only for display; PV and stored fields use raw OPD.
    display_reference = float(np.min(selected_field[mask]))
    profile_axis = np.linspace(-aperture_radius, aperture_radius, num_profile_points)
    x_profile = opd_at(profile_axis, 0.0, selected_coefficients, eigenvalues, selected_angle)
    y_profile = opd_at(0.0, profile_axis, selected_coefficients, eigenvalues, selected_angle)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    contour = axes[0].contourf(
        X*1e3, Y*1e3, (selected_field-display_reference)*1e9,
        levels=np.linspace(0, base_pv*1e9, 61), cmap="viridis",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Entry x [mm]")
    axes[0].set_ylabel("Entry y [mm]")
    axes[0].set_title("OPD over rear-transmitted entrance rays")
    fig.colorbar(contour, ax=axes[0], label="OPD minus aperture minimum [nm]", shrink=0.85)
    axes[1].plot(profile_axis*1e3, (x_profile-display_reference)*1e9, label="x axis (minor heating axis)")
    axes[1].plot(profile_axis*1e3, (y_profile-display_reference)*1e9, label="y axis (major heating axis)")
    axes[1].set_xlabel("Position along axis [mm]")
    axes[1].set_ylabel("OPD minus aperture minimum [nm]")
    axes[1].set_title("Refracted-ray OPD profiles")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"Thermo-optic OPD: {selected_angle:g} deg, PV={base_pv*1e9:.3f} nm")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "kind": "refracted_ray_thermo_optic_opd_inspection",
        "physical_parameters_SI": asdict(p),
        "h_eff_W_per_m2_K": p.h_eff,
        "opd_definition": "dn_dT / cos(theta_t) * integral_0^L DeltaT(x0,y0+z*tan(theta_t),z) dz; unit m; front-face entrance coordinates",
        "ray_reference": "straight refracted ray in unheated uniform substrate",
        "aperture_convention": "0.90R entrance aperture intersected with rays reaching rear face inside R; side-exiting rays excluded",
        "aperture_radius_ratio": aperture_radius_ratio,
        "aperture_radius_m": aperture_radius,
        "numerical_settings": {
            "M": max_azimuthal_order, "N": num_radial_modes,
            "projection_r": num_projection_r, "projection_phi": num_projection_phi,
            "opd_xy": num_opd_xy, "profile_points": num_profile_points,
            "opd_ray_points": num_opd_ray_points, "opd_point_chunk_size": opd_point_chunk_size,
        },
        "eigenvalues_per_m": {str(m): values.tolist() for m, values in eigenvalues.items()},
        "cases": records,
        "sensitivity": sensitivity,
        "display_reference_m": display_reference,
    }
    result_directory.mkdir(parents=True, exist_ok=True)
    summary.update(status="COMPLETE",physical_parameters_SI=asdict(parameters),provenance=provenance(repository_root,__file__))
    write_summary(result_directory, summary, defaults=script_defaults(globals()))
    np.savez_compressed(
        data_directory(result_directory) / "opd_fields.npz", x_m=axis, y_m=axis,
        entrance_aperture_mask=entrance_mask, aperture_mask=np.stack(transmitted_masks), incidence_angles_deg=incidence_angles_deg,
        opd_m=np.stack(opd_maps), profile_axis_m=profile_axis,
        selected_angle_deg=selected_angle, x_axis_opd_m=x_profile, y_axis_opd_m=y_profile,
    )
    print(f"Figure: {figure_path}")
    print(f"Results: {result_directory}")


if __name__ == "__main__":
    main()
