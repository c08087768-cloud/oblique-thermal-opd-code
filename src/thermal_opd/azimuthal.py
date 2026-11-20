import numpy as np


def azimuthal_coefficients(q_phi, phi, max_mode):
    """Return Fourier coefficients for heat flux on one fixed-radius circle.

    ``q_phi`` and ``phi`` are one-dimensional samples on a uniform full-period
    grid, without a duplicate endpoint. ``phi`` is in radians. Use more than
    ``2 * max_mode`` samples to resolve the retained sine/cosine pairs; finer
    sampling can still be needed for heat flux with higher angular orders.

    The convention is q = a[0] + sum(a[m]*cos(m*phi) + b[m]*sin(m*phi)),
    so a[0] is the angular mean. Coefficients have the same units as q_phi.
    Sampling and truncation settings are provided by the caller.
    """

    a = np.zeros(max_mode + 1)
    b = np.zeros(max_mode + 1)

    a[0] = np.mean(q_phi)

    for m in range(1, max_mode + 1):
        a[m] = 2 * np.mean(q_phi * np.cos(m * phi))
        b[m] = 2 * np.mean(q_phi * np.sin(m * phi))

    return a, b
