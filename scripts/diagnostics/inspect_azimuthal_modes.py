from thermal_opd.run_io import script_defaults
import numpy as np
from pathlib import Path
from dataclasses import asdict
from thermal_opd.run_io import create_run,data_directory,write_summary,provenance,parse_output_directory

from thermal_opd.azimuthal import azimuthal_coefficients
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.parameters import PhysicalParameters


# Physical inputs and case choices.
parameters = PhysicalParameters()
incidence_angles_deg = (0.0, 30.0, 60.0)
sample_radius_ratio = 1.0  # Sample one circle at r / w0 = 1.

# Numerical settings for this inspection, defined here rather than in the solver.
max_azimuthal_order = 10
num_phi_samples = 256  # Must exceed 2 * max_azimuthal_order.


def main():
    """Inspect the Fourier expansion of heat flux on one fixed-radius circle."""
    root=Path(__file__).resolve().parents[2]
    directory=create_run(root,'azimuthal_inspection',parse_output_directory(__doc__))
    records=[];saved={}
    r = sample_radius_ratio * parameters.beam_radius
    # Do not duplicate the periodic endpoint at 2*pi.
    phi = np.linspace(0.0, 2 * np.pi, num_phi_samples, endpoint=False)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    print(f"Beam radius w0: {parameters.beam_radius * 1e3:.3f} mm")
    print(f"Sample radius r: {r * 1e3:.3f} mm")
    print(f"Gaussian power: {parameters.absorbed_power * 1e3:.3f} mW")
    print(f"M = {max_azimuthal_order}, n_phi = {num_phi_samples}")
    print("Expansion: q(phi) = a_0 + sum[a_m cos(m phi) + b_m sin(m phi)]")

    for angle_deg in incidence_angles_deg:
        theta = np.deg2rad(angle_deg)
        q_phi = oblique_gaussian_heat_flux(
            x, y, parameters.beam_radius, theta, parameters.absorbed_power
        )
        a, b = azimuthal_coefficients(q_phi, phi, max_azimuthal_order)

        # Reconstruct on the same circle. a_0 is the mean, not twice the mean.
        q_reconstructed = np.full_like(q_phi, a[0])
        for m in range(1, max_azimuthal_order + 1):
            q_reconstructed += (
                a[m] * np.cos(m * phi) + b[m] * np.sin(m * phi)
            )

        relative_l2_error = (
            np.linalg.norm(q_reconstructed - q_phi) / np.linalg.norm(q_phi)
        )
        # On a uniform circle, the common arc-length weight cancels in the ratio.
        odd_amplitude = np.max(np.abs(a[1::2])) / a[0]
        sine_amplitude = np.max(np.abs(b)) / a[0]

        print(f"\nIncidence angle: {angle_deg:.1f} deg")
        print(" m      a_m [W/m^2]      b_m [W/m^2]       a_m / a_0")
        for m in range(len(a)):
            print(f"{m:2d}  {a[m]: .6e}  {b[m]: .6e}  {a[m] / a[0]: .6e}")
        print(f"Relative L2 reconstruction error on circle: {relative_l2_error:.6e}")
        print(f"Largest odd cosine amplitude / a_0: {odd_amplitude:.6e}")
        print(f"Largest sine amplitude / a_0: {sine_amplitude:.6e}")
        records.append(dict(angle_deg=angle_deg,relative_L2_error=float(relative_l2_error),
                            odd_over_mean=float(odd_amplitude),sine_over_mean=float(sine_amplitude)))
        saved[f'q_theta{angle_deg:g}_W_per_m2']=q_phi
        saved[f'a_theta{angle_deg:g}_W_per_m2']=a
        saved[f'b_theta{angle_deg:g}_W_per_m2']=b
    np.savez_compressed(data_directory(directory)/'azimuthal_fields.npz',phi_rad=phi,**saved)
    write_summary(directory, dict(kind='azimuthal_inspection', status='COMPLETE', physical_parameters_SI=asdict(parameters), numerical_settings=dict(max_azimuthal_order=max_azimuthal_order, num_phi_samples=num_phi_samples), sample_radius_m=r, sample_radius_ratio=sample_radius_ratio, cases=records, provenance=provenance(root, __file__)), defaults=script_defaults(globals()))
    print(f'Results: {directory}')


if __name__ == "__main__":
    main()
