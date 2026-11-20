import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, j0, j1


def bessel_j(order,argument):
    """Integer J_m via fast J0/J1 recurrence away from the small-x region.

    For x<max(8,2m) retain SciPy's general jv to avoid an unstable upward
    recurrence. Noninteger/negative orders also use jv. This changes the
    numerical evaluation only; no modal coefficients are omitted.
    """
    x = np.asarray(argument,dtype=float)
    if order==0:
        return j0(x)
    if order==1:
        return j1(x)
    if order<0 or int(order)!=order:
        return jv(order,x)
    # Small PV-search queries are faster in SciPy than a Python recurrence.
    if x.size<256:
        return jv(order,x)
    large = x>=max(8.,2.*order)
    if not np.any(large):
        return jv(order,x)
    result = np.empty_like(x)
    result[~large] = jv(order,x[~large])
    values = x[large]
    previous,current = j0(values),j1(values)
    for m in range(1,int(order)):
        previous,current = current,2*m/values*current-previous
    result[large] = current
    return result


def radial_eigenvalue_equation(lam, m, radius, conductivity, h):
    """Return k*lam*J_m'(lam*R) + h*J_m(lam*R) for sidewall heat loss.

    ``lam`` is a radial wavenumber (1/m), ``conductivity`` is k (W/(m K)),
    and ``h`` is the sidewall heat-transfer coefficient (W/(m^2 K)).
    """

    x = lam * radius

    return (
        conductivity * lam * jvp(m, x)
        + h * jv(m, x)
    )


def radial_eigenvalues(m, radius, conductivity, h, num_modes):
    """Return the first positive radial wavenumbers (1/m) in ascending order.

    The current search is used for nonnegative integer m, positive radius,
    conductivity and h, and a positive number of modes. For h=0 the m=0
    zero mode requires separate treatment; this unsupported case is rejected.
    """

    if m == 0 and h == 0:
        raise ValueError(
            'The insulated axisymmetric radial zero mode is not implemented; '
            'use positive sidewall heat transfer for modal source preparation'
        )

    def equation(lam):
        return radial_eigenvalue_equation(
            lam, m, radius, conductivity, h
        )

    roots = []

    # Work with the dimensionless variable x = lambda * R.
    x_max = (num_modes + m / 2 + 5) * np.pi
    x_grid = np.linspace(1e-8, x_max, 10000)

    values = equation(x_grid / radius)

    for i in range(len(x_grid) - 1):
        if values[i] * values[i + 1] < 0:
            lam_left = x_grid[i] / radius
            lam_right = x_grid[i + 1] / radius

            root = brentq(equation, lam_left, lam_right)
            roots.append(root)

            if len(roots) == num_modes:
                break

    if len(roots) < num_modes:
        raise RuntimeError(
            f"Found only {len(roots)} roots, expected {num_modes}."
        )

    return np.array(roots)


def radial_mode_norm(lam, m, radius):
    """Return integral_0^R J_m(lam*r)^2 r dr analytically (m^2).

    The Bessel integral identity holds for positive lam, including the
    Robin eigenvalues. At those roots it is equivalent to the norm in
    supplemental Eq. (S12), without requiring h as another input.
    """
    x = lam * radius
    return 0.5 * radius**2 * (
        jv(m, x)**2 - jv(m - 1, x) * jv(m + 1, x)
    )
