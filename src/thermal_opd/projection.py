import numpy as np
from scipy.special import jv

from thermal_opd.eigenmodes import radial_mode_norm


def fourier_bessel_coefficient(q, r, phi, m, lam):
    """Return a cosine Fourier-Bessel source coefficient, in the units of q.

    q has shape (len(r), len(phi)). r spans the disk from 0 to its radius;
    phi is a uniform full-period grid without the duplicate endpoint, with
    more than 2*m samples. The area element is r dr dphi.

    The numerator uses numerical source quadrature. The denominator uses
    the analytic radial norm and angular norm (2*pi for m=0, pi otherwise).
    The resulting coefficient multiplies J_m(lam*r)*cos(m*phi), and is
    distinct from the steady axial response coefficients.
    """
    radial_basis = jv(m, lam * r)
    angular_basis = np.cos(m * phi)
    dphi = 2 * np.pi / len(phi)
    angular_projection = dphi * np.sum(q * angular_basis, axis=1)
    numerator = np.trapezoid(angular_projection * radial_basis * r, x=r)
    angular_norm = 2 * np.pi if m == 0 else np.pi
    denominator = radial_mode_norm(lam, m, r[-1]) * angular_norm

    return numerator / denominator


def reconstruct_cosine_field(coefficients, eigenvalues, r, phi):
    """Reconstruct a field from cosine Fourier-Bessel modes."""

    R, Phi = np.meshgrid(r, phi, indexing="ij")
    field = np.zeros_like(R)

    for m, coeffs_m in coefficients.items():
        for n, coefficient in enumerate(coeffs_m):
            lam = eigenvalues[m][n]
            field += coefficient * jv(m, lam * R) * np.cos(m * Phi)

    return field


def fourier_bessel_coefficients(q, r, phi, m, eigenvalues):
    """Project one angular order onto many roots, sharing its angular integral."""
    eigenvalues = np.asarray(eigenvalues)
    angular_projection = (2*np.pi/len(phi))*np.sum(q*np.cos(m*phi), axis=1)
    radial = jv(m, eigenvalues[:, None]*r)
    numerator = np.trapezoid(radial*angular_projection*r, x=r, axis=1)
    angular_norm = 2*np.pi if m == 0 else np.pi
    return numerator/(radial_mode_norm(eigenvalues, m, r[-1])*angular_norm)
