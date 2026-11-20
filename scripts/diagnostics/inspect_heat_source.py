from thermal_opd.run_io import script_defaults
from pathlib import Path
from dataclasses import asdict
from thermal_opd.model import ThermalModel, ModalResolution
from thermal_opd.run_io import create_run,figure_directory,provenance,write_summary,parse_output_directory

import matplotlib.pyplot as plt
import numpy as np

from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.eigenmodes import radial_eigenvalue_equation
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.projection import reconstruct_cosine_field


parameters = PhysicalParameters()
incidence_angle_deg = 60.0

# These are inspection settings, not a claim of converged temperature or OPD.
max_azimuthal_order = 8
num_radial_modes = 12
num_projection_r = 400
num_projection_phi = 256

repository_root = Path(__file__).resolve().parents[2]


def main():
    result_directory=create_run(repository_root,'heat_source_inspection',parse_output_directory(__doc__))
    figure_path=figure_directory(result_directory)/'heat_source_reconstruction.png'
    result_path=result_directory/'summary.json'
    theta = np.deg2rad(incidence_angle_deg)
    r = np.linspace(0.0, parameters.radius, num_projection_r)
    phi = np.linspace(0.0, 2 * np.pi, num_projection_phi, endpoint=False)
    R, Phi = np.meshgrid(r, phi, indexing="ij")
    q_true = oblique_gaussian_heat_flux(
        R * np.cos(Phi), R * np.sin(Phi),
        parameters.beam_radius, theta, parameters.absorbed_power,
    )

    prepared=ThermalModel(parameters,incidence_angle_deg,ModalResolution(max_azimuthal_order,num_radial_modes,num_projection_r,num_projection_phi,16)).prepare()
    eigenvalues,coefficients=prepared.roots,prepared.coefficients
    robin_residuals=[]
    for m,roots in eigenvalues.items():
        robin_residuals.extend(np.abs(radial_eigenvalue_equation(roots,m,parameters.radius,parameters.conductivity,parameters.h_eff)))

    q_reconstructed = reconstruct_cosine_field(coefficients, eigenvalues, r, phi)
    difference = q_reconstructed - q_true
    dphi = 2 * np.pi / len(phi)
    # The disk area measure is r dr dphi, unlike an unweighted array norm.
    squared_error = np.trapezoid(dphi * np.sum(difference**2, axis=1) * r, x=r)
    squared_reference = np.trapezoid(dphi * np.sum(q_true**2, axis=1) * r, x=r)
    relative_l2_error = np.sqrt(squared_error / squared_reference)
    disk_power = np.trapezoid(dphi * np.sum(q_true, axis=1) * r, x=r)
    reconstructed_power = np.trapezoid(
        dphi * np.sum(q_reconstructed, axis=1) * r, x=r
    )
    residual = float(np.max(robin_residuals))

    print(f"theta = {incidence_angle_deg:g} deg, w0 = {parameters.beam_radius * 1e3:.3f} mm")
    print(f"Sidewall h_eff = {parameters.h_eff:.6f} W/(m^2 K)")
    print(f"M = {max_azimuthal_order}, N = {num_radial_modes}")
    print(f"Projection grid: {len(r)} x {len(phi)}")
    print(" m    first lambda [1/m]    first lambda*R    first coefficient [W/m^2]")
    for m, roots in eigenvalues.items():
        print(f"{m:2d}    {roots[0]:14.6f}    {roots[0] * parameters.radius:14.6f}    {coefficients[m][0]: .6e}")
    print(f"Maximum radial Robin equation residual: {residual:.6e} W/(m^2 K)")
    print(f"Area-weighted relative L2 source error: {relative_l2_error:.6e}")
    print(f"Gaussian power on full plane: {parameters.absorbed_power * 1e3:.6f} mW")
    print(f"Power intercepted by disk: {disk_power * 1e3:.6f} mW")
    print(f"Power in reconstructed source: {reconstructed_power * 1e3:.6f} mW")

    # Close the seam for plotting only, without duplicating quadrature samples.
    plot_phi = np.append(phi, 2 * np.pi)
    plot_r, plot_phi = np.meshgrid(r, plot_phi, indexing="ij")
    X = plot_r * np.cos(plot_phi) * 1e3
    Y = plot_r * np.sin(plot_phi) * 1e3
    levels = np.linspace(min(0.0, q_reconstructed.min()), max(q_true.max(), q_reconstructed.max()), 61)
    error_limit = np.max(np.abs(difference))
    error_levels = np.linspace(-error_limit, error_limit, 61)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for ax, field, title, color_levels, cmap in zip(
        axes,
        (q_true, q_reconstructed, difference),
        ("Original Gaussian", "Fourier-Bessel reconstruction", "Reconstruction minus original"),
        (levels, levels, error_levels),
        ("viridis", "viridis", "RdBu_r"),
    ):
        closed_field = np.column_stack((field, field[:, 0]))
        mesh = ax.contourf(X, Y, closed_field, levels=color_levels, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(mesh, ax=ax, label="Heat flux [W/m^2]", shrink=0.85)
    fig.suptitle(
        f"Oblique heat source: {incidence_angle_deg:g} deg, M={max_azimuthal_order}, "
        f"N={num_radial_modes}; disk L2 error={relative_l2_error:.2e}"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "kind": "heat_source_inspection",
        "incidence_angle_deg": incidence_angle_deg,
        "radius_m": parameters.radius,
        "beam_radius_m": parameters.beam_radius,
        "conductivity_W_per_m_K": parameters.conductivity,
        "sidewall_h_W_per_m2_K": parameters.h_eff,
        "gaussian_power_W": parameters.absorbed_power,
        "max_azimuthal_order": max_azimuthal_order,
        "num_radial_modes": num_radial_modes,
        "num_projection_r": len(r),
        "num_projection_phi": len(phi),
        "relative_disk_L2_error": float(relative_l2_error),
        "maximum_radial_robin_residual_W_per_m2_K": residual,
        "intercepted_power_W": float(disk_power),
        "reconstructed_power_W": float(reconstructed_power),
        "eigenvalues_per_m": {str(m): roots.tolist() for m, roots in eigenvalues.items()},
        "source_coefficients_W_per_m2": {str(m): c.tolist() for m, c in coefficients.items()},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    summary.update(status="COMPLETE",physical_parameters_SI=asdict(parameters),provenance=provenance(repository_root,__file__))
    write_summary(result_directory, summary, defaults=script_defaults(globals()))
    print(f"Figure: {figure_path}")
    print(f"Summary: {result_path}")


if __name__ == "__main__":
    main()
