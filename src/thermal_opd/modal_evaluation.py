"""Shared Fourier-Bessel spatial sums for steady and transient responses."""
import numpy as np
from scipy.special import jv, jvp


def modal_temperature(r, phi, z, coefficients, eigenvalues, response, *, group_by_order=False):
    """Spatial sum with optional grouping for azimuthal-contribution studies.

    The default retains the historical flat accumulation bit for bit, including
    finite-difference evidence replays. Grouping changes only addition order.
    """
    r, phi, z = (np.asarray(a) for a in (r, phi, z))
    shape = np.broadcast_shapes(r.shape,phi.shape,z.shape)
    temperature = np.zeros(shape)
    orders = []
    for m, values in coefficients.items():
        order = np.zeros(shape) if group_by_order else temperature
        for n in range(len(values)):
            order += jv(m, eigenvalues[m][n]*r)*np.cos(m*phi)*response(m, n, z)
        if group_by_order:orders.append(order)
    # The same azimuthal grouping defines independently evaluated modal fields.
    # Keep radial order and formula, avoiding cross-order cancellation while
    # accumulating each radial subtotal; no saved field is used in this sum.
    return (np.sum(orders,axis=0) if orders else np.zeros(shape)) if group_by_order else temperature


def modal_temperature_gradient(r, phi, z, coefficients, eigenvalues, response):
    """Shared analytic spatial derivatives; response returns (amplitude, dz)."""
    r,phi,z = (np.asarray(a) for a in (r,phi,z))
    shape = np.broadcast_shapes(r.shape,phi.shape,z.shape)
    radial_gradient = np.zeros(shape)
    axial_gradient = np.zeros(shape)
    for m, coeffs_m in coefficients.items():
        angular = np.cos(m * phi)
        for n in range(len(coeffs_m)):
            lam = eigenvalues[m][n]
            axial, derivative = response(m,n,z)
            radial_gradient += lam * jvp(m, lam * r) * angular * axial
            axial_gradient += jv(m, lam * r) * angular * derivative
    return radial_gradient, axial_gradient
