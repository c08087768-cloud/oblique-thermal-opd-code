from thermal_opd.run_io import script_defaults
from thermal_opd.run_io import data_directory
from dataclasses import asdict
from pathlib import Path
from thermal_opd.model import ThermalModel, ModalResolution
from thermal_opd.run_io import create_run,figure_directory,provenance,write_summary,parse_output_directory

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.steady import steady_temperature
from thermal_opd.steady_validation import steady_boundary_residuals, steady_energy_balance


parameters = PhysicalParameters()
incidence_angle_deg = 60.0

# Numerical settings for this case. z=0 is front; z=L is rear.
max_azimuthal_order = 8
num_radial_modes = 12
num_projection_r = 400
num_projection_phi = 256
num_field_r = 161
num_field_phi = 180
num_energy_r = 80
num_energy_z = 64

repository_root = Path(__file__).resolve().parents[2]


def main():
    result_directory=create_run(repository_root,'temperature_inspection',parse_output_directory(__doc__))
    figure_path=figure_directory(result_directory)/'steady_temperature.png'
    result_path=result_directory/'summary.json'
    p = parameters
    theta = np.deg2rad(incidence_angle_deg)
    prepared=ThermalModel(p,incidence_angle_deg,ModalResolution(max_azimuthal_order,num_radial_modes,num_projection_r,num_projection_phi,16)).prepare()
    solution=prepared.solve_steady()
    coefficients,roots=prepared.coefficients,prepared.roots
    args = (coefficients, roots, p.conductivity, p.thickness, p.h_eff, p.h_eff)

    r = np.linspace(0.0, p.radius, num_field_r)
    phi = np.linspace(0.0, 2 * np.pi, num_field_phi, endpoint=False)
    z_planes = np.array([0.0, p.thickness / 2, p.thickness])
    # Broadcast 1D radial/azimuthal factors rather than repeat Bessel evaluations.
    fields = np.stack([
        solution.temperature(r[:,None],phi[None,:],z) for z in z_planes
    ])
    _, front_dz = solution.temperature_gradient(r[:,None],phi[None,:],0.)
    specified_source = oblique_gaussian_heat_flux(
        r[:, None] * np.cos(phi), r[:, None] * np.sin(phi),
        p.beam_radius, theta, p.absorbed_power,
    )
    peak_flux = np.max(specified_source)
    front_flux = p.h_eff * fields[0] - p.conductivity * front_dz
    front_specified_residual = np.max(np.abs(front_flux - specified_source)) / peak_flux

    # Independent Gauss quadrature for all heat-loss surfaces and input power.
    nodes_r, weights_r = leggauss(num_energy_r)
    energy_r = (nodes_r + 1) * p.radius / 2
    weights_r = weights_r * p.radius / 2
    nodes_z, _ = leggauss(num_energy_z)
    energy_z = (nodes_z + 1) * p.thickness / 2
    residuals = steady_boundary_residuals(
        r, phi, energy_z, coefficients, roots,
        p.radius, p.conductivity, p.thickness, p.h_eff, p.h_eff, p.h_eff,
    )
    front_modal_residual = residuals["front_modal_W_per_m2"] / peak_flux
    rear_residual = residuals["rear_W_per_m2"] / peak_flux
    side_residual = residuals["side_W_per_m2"] / peak_flux
    balance = steady_energy_balance(
        coefficients, roots, p.radius, p.conductivity, p.thickness,
        p.h_eff, p.h_eff, p.h_eff,
        num_r=num_energy_r, num_z=num_energy_z, num_phi=num_field_phi,
    )
    front_loss, rear_loss, side_loss = (balance[key] for key in ("front_loss_W", "rear_loss_W", "side_loss_W"))
    total_loss = balance["total_loss_W"]
    modal_input = balance["modal_input_W"]
    gaussian_on_energy_grid = oblique_gaussian_heat_flux(
        energy_r[:, None] * np.cos(phi), energy_r[:, None] * np.sin(phi),
        p.beam_radius, theta, p.absorbed_power,
    )
    specified_input = 2 * np.pi * np.sum(
        weights_r * energy_r * np.mean(gaussian_on_energy_grid, axis=1)
    )
    energy_error = balance["balance_absolute_error_W"] / abs(modal_input)
    input_difference = abs(modal_input - specified_input) / specified_input
    center_temperatures = [float(steady_temperature(0.0, 0.0, z, *args)) for z in z_planes]
    print(f"theta = {incidence_angle_deg:g} deg, M={max_azimuthal_order}, N={num_radial_modes}")
    print("Axial coordinates: front=0, mid-plane=L/2, rear=L")
    print(f"Projection grid: {num_projection_r} x {num_projection_phi}")
    for label, field, center in zip(("Front", "Mid-plane", "Rear"), fields, center_temperatures):
        print(f"{label}: center rise={center:.9f} K, sampled range=[{field.min():.9f}, {field.max():.9f}] K")
    print(f"Front residual vs retained modes / peak flux: {front_modal_residual:.6e}")
    print(f"Front residual vs prescribed Gaussian / peak flux: {front_specified_residual:.6e}")
    print(f"Rear Robin residual / peak flux: {rear_residual:.6e}")
    print(f"Side Robin residual / peak flux: {side_residual:.6e}")
    print(f"Heat loss front/rear/side: {front_loss*1e3:.6f}, {rear_loss*1e3:.6f}, {side_loss*1e3:.6f} mW")
    print(f"Total loss: {total_loss*1e3:.6f} mW; retained-mode input: {modal_input*1e3:.6f} mW")
    print(f"Retained-mode energy-balance relative error: {energy_error:.6e}")
    print(f"Prescribed disk input: {specified_input*1e3:.6f} mW; modal input relative difference: {input_difference:.6e}")

    plot_phi = np.append(phi, 2 * np.pi)
    X = r[:, None] * np.cos(plot_phi) * 1e3
    Y = r[:, None] * np.sin(plot_phi) * 1e3
    levels = np.linspace(fields.min(), fields.max(), 61)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for ax, field, label in zip(axes, fields, ("Front: z=0", "Mid-plane: z=L/2", "Rear: z=L")):
        mesh = ax.contourf(X, Y, np.column_stack((field, field[:, 0])), levels=levels, cmap="inferno")
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(mesh, ax=ax, label="Temperature rise [K]", shrink=0.85)
    fig.suptitle(f"Steady temperature rise: {incidence_angle_deg:g} deg, M={max_azimuthal_order}, N={num_radial_modes}")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "kind": "steady_temperature_inspection",
        "physical_parameters_SI": asdict(p),
        "h_eff_W_per_m2_K": p.h_eff,
        "incidence_angle_deg": incidence_angle_deg,
        "axial_coordinate": "z_code=z_paper+L/2; front=0, rear=L",
        "numerical_settings": {
            "M": max_azimuthal_order, "N": num_radial_modes,
            "projection_r": num_projection_r, "projection_phi": num_projection_phi,
            "field_r": num_field_r, "field_phi": num_field_phi,
            "energy_r": num_energy_r, "energy_z": num_energy_z,
        },
        "center_temperature_rises_K": center_temperatures,
        "front_modal_residual_over_peak_flux": float(front_modal_residual),
        "front_gaussian_residual_over_peak_flux": float(front_specified_residual),
        "rear_residual_over_peak_flux": float(rear_residual),
        "side_residual_over_peak_flux": float(side_residual),
        "heat_losses_W": {"front": float(front_loss), "rear": float(rear_loss), "side": float(side_loss)},
        "modal_input_W": float(modal_input),
        "specified_disk_input_W": float(specified_input),
        "modal_energy_balance_relative_error": float(energy_error),
        "modal_input_relative_difference": float(input_difference),
        "eigenvalues_per_m": {str(m): v.tolist() for m, v in roots.items()},
        "source_coefficients_W_per_m2": {str(m): v.tolist() for m, v in coefficients.items()},
    }
    result_directory.mkdir(parents=True, exist_ok=True)
    summary.update(status="COMPLETE",physical_parameters_SI=asdict(parameters),provenance=provenance(repository_root,__file__))
    write_summary(result_directory, summary, defaults=script_defaults(globals()))
    np.savez_compressed(data_directory(result_directory) / "surface_fields.npz", r_m=r, phi_rad=phi, z_m=z_planes, temperature_rise_K=fields)
    print(f"Figure: {figure_path}")
    print(f"Results: {result_directory}")


if __name__ == "__main__":
    main()
