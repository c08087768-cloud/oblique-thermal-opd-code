"""Steady temperature rise for 0 <= z <= L, front face at z=0.

The manuscript uses a centered axial coordinate: z_code = z_paper + L/2.
Temperatures returned here are rises above ambient, in kelvin.
"""

import numpy as np
from scipy.special import jv
import thermal_opd.modal_evaluation as modal_evaluation


def axial_coefficients(lam, source_coefficient, conductivity, thickness, h_front, h_back):
    """Return reference cosh/sinh coefficients A=Z(0), B=Z'(0)/lam.

    For field evaluation use axial_response, avoiding cancellation between
    these two coefficients at large lam*z.
    """
    front, front_gradient = axial_response(
        0.0, lam, source_coefficient, conductivity, thickness, h_front, h_back
    )
    return front, front_gradient / lam


def axial_temperature(z, lam, A, B):
    """Evaluate the reference cosh/sinh expression for modest lam*z.

    Use axial_response for production evaluation of high radial modes.
    """

    return A * np.cosh(lam * z) + B * np.sinh(lam * z)


def axial_response(z, lam, source_coefficient, conductivity, thickness, h_front, h_back):
    """Return axial temperature amplitude (K) and z derivative (K/m).

    For lam>0 and 0<=z<=L this solves Z''-lam^2*Z=0 with
    h_front*Z(0)-k*Z'(0)=source_coefficient and h_back*Z(L)+k*Z'(L)=0.
    Source coefficients have units W/m^2. Decaying exponentials avoid
    subtraction of exponentially large cosh/sinh terms.
    """
    z = np.asarray(z)
    amplitude, rear_ratio = _axial_decay_parameters(
        lam, source_coefficient, conductivity, thickness, h_front, h_back
    )
    forward = np.exp(-lam * z)
    backward = rear_ratio * np.exp(-lam * (2 * thickness - z))
    temperature = amplitude * (forward + backward)
    derivative = lam * amplitude * (-forward + backward)
    return temperature, derivative


def _axial_decay_parameters(lam, source_coefficient, conductivity, thickness, h_front, h_back):
    """Return shared constants for stable axial response and integration."""
    k_lam = conductivity * lam
    rear_ratio = (k_lam - h_back) / (k_lam + h_back)
    round_trip = np.exp(-2 * lam * thickness)
    amplitude = source_coefficient / (
        k_lam + h_front + (h_front - k_lam) * rear_ratio * round_trip
    )
    return amplitude, rear_ratio


def steady_mode(r, phi, z, m, lam, A, B):
    """Evaluate one steady Fourier-Bessel temperature mode."""

    radial = jv(m, lam * r)
    angular = np.cos(m * phi)
    axial = axial_temperature(z, lam, A, B)

    return radial * angular * axial

def steady_temperature(r, phi, z, coefficients, eigenvalues, conductivity, thickness, h_front, h_back):
    """Return temperature rise (K) by summing cosine Fourier-Bessel modes.

    r, phi, z are broadcastable coordinates in m, radians, m. z=0 is front,
    z=L is rear. coefficients contains heat-flux projections (W/m^2),
    eigenvalues contains radial wavenumbers (1/m). No source reconstruction
    is used in this temperature evaluation.
    """

    def response(m, n, depths):
        return axial_response(depths, eigenvalues[m][n], coefficients[m][n],
                              conductivity, thickness, h_front, h_back)[0]
    return modal_evaluation.modal_temperature(r, phi, z, coefficients, eigenvalues, response)


def steady_temperature_gradient(r, phi, z, coefficients, eigenvalues, conductivity, thickness, h_front, h_back):
    """Return analytical radial and axial gradients (K/m) for boundary checks.

    Coordinates and modal inputs follow steady_temperature. This returns
    dT/dr and dT/dz, not the azimuthal component of the full gradient.
    """
    def response(m,n,depths):
        return axial_response(depths,eigenvalues[m][n],coefficients[m][n],
                              conductivity,thickness,h_front,h_back)
    return modal_evaluation.modal_temperature_gradient(r,phi,z,coefficients,eigenvalues,response)
