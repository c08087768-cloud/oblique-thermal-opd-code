"""Field comparisons and numerical extrema on a rear-transmitted aperture."""

import numpy as np
from scipy.optimize import minimize


def field_errors(temperature, reference_temperature, area_weights, opd, reference_opd, *,
                 temperature_normalization='per_plane'):
    """Compare temperature planes and raw OPD arrays on identical coordinates.

    Temperature shape is (plane, point), area_weights is (point,). OPD points
    have equal area weight (a uniform Cartesian grid). OPD shape errors remove
    the mean difference, and normalize by reference PV, never by its piston.
    The default retains historical per-plane norms. 'global' uses the largest
    reference plane norm for all planes, for early-time unheated depths whose
    individual norms vanish. It does not change the acceptance threshold.
    """
    temperature, reference_temperature = np.asarray(temperature), np.asarray(reference_temperature)
    opd, reference_opd = np.asarray(opd), np.asarray(reference_opd)
    weights = np.asarray(area_weights)
    if temperature.shape != reference_temperature.shape or temperature.ndim != 2:
        raise ValueError('Temperature arrays must share (plane, point) shape')
    if weights.shape != temperature.shape[1:] or np.any(weights <= 0):
        raise ValueError('Positive area weights must match temperature points')
    if opd.shape != reference_opd.shape or opd.ndim != 1 or opd.size < 2:
        raise ValueError('OPD arrays must share a nonempty point dimension')
    if not all(np.all(np.isfinite(a)) for a in (temperature, reference_temperature, weights, opd, reference_opd)):
        raise ValueError('Comparison data must be finite')
    difference = temperature - reference_temperature
    norm2 = np.sum(weights * reference_temperature**2, axis=1)
    norminf = np.max(np.abs(reference_temperature), axis=1)
    if temperature_normalization == 'global':
        norm2 = np.full_like(norm2,np.max(norm2))
        norminf = np.full_like(norminf,np.max(norminf))
    elif temperature_normalization != 'per_plane':
        raise ValueError('Unknown temperature normalization')
    pv = np.ptp(reference_opd)
    if np.any(norm2 == 0) or np.any(norminf == 0) or pv == 0:
        raise ValueError('Reference norms and OPD PV must be nonzero')
    optical_difference = opd - reference_opd
    optical_difference -= optical_difference.mean()
    return {
        'temperature_relative_l2': np.sqrt(np.sum(weights*difference**2, axis=1)/norm2).tolist(),
        'temperature_relative_linf': (np.max(np.abs(difference), axis=1)/norminf).tolist(),
        'opd_piston_removed_rms_over_reference_pv': float(np.sqrt(np.mean(optical_difference**2))/pv),
        'opd_piston_removed_linf_over_reference_pv': float(np.max(np.abs(optical_difference))/pv),
        'opd_relative_pv': float(abs(np.ptp(opd)-pv)/pv),
    }


def within_tolerance(errors, tolerance):
    values = np.concatenate([np.atleast_1d(v) for v in errors.values()])
    return bool(np.all(np.isfinite(values)) and np.all(values <= tolerance))


def continuous_aperture_pv(evaluate, *, radius, thickness, refraction_angle,
                           aperture_radius, num_seed_radius, num_seed_phi):
    """Numerically search both extrema, independent of a figure's XY grid.

    Coordinates are normalized radial fraction s and periodic angle phi. The
    radius bound enforces both the entrance aperture and shifted rear disk.
    Assumes the central ray reaches the rear (|L tan(theta_t)| < R). Multiple
    seeds reduce local-minimum risk; this is not a certified global bound.
    """
    shift = thickness*np.tan(refraction_angle)
    if not (radius > 0 and 0 < aperture_radius <= radius and thickness > 0
            and abs(refraction_angle) < np.pi/2 and abs(shift) < radius):
        raise ValueError('Require forward rays, an interior central ray and a valid aperture')
    def coordinates(s, phi):
        limit = np.minimum(aperture_radius,
                           -shift*np.sin(phi) + np.sqrt(radius**2 - (shift*np.cos(phi))**2))
        return s*limit*np.cos(phi), s*limit*np.sin(phi)
    s, phi = np.meshgrid(np.linspace(0, 1, num_seed_radius),
                         np.linspace(0, 2*np.pi, num_seed_phi, endpoint=False), indexing='ij')
    x, y = coordinates(s.ravel(), phi.ravel())
    values = np.asarray(evaluate(x, y))
    scale = np.ptp(values)
    if scale <= 0 or not np.all(np.isfinite(values)):
        raise ValueError('Extrema search requires finite nonconstant samples')
    candidates = [(float(v), float(a), float(b)) for v, a, b in zip(values, x, y)]
    successes = 0
    for sign in (1, -1):
        # Well separated angular seeds prevent all starts falling in one basin.
        starts = [int(np.argmin(sign*values))]
        for angular_index in range(0, num_seed_phi, max(1, num_seed_phi//4)):
            ids = np.arange(angular_index, values.size, num_seed_phi)
            starts.append(int(ids[np.argmin(sign*values[ids])]))
        for index in dict.fromkeys(starts):
            def objective(point):
                a, b = coordinates(*point)
                return float(sign*np.asarray(evaluate(a, b))/scale)
            fit = minimize(objective, [s.ravel()[index], phi.ravel()[index]],
                           method='SLSQP', bounds=((0., 1.), (0., 2*np.pi)),
                           options={'ftol': 1e-12, 'maxiter': 100})
            a, b = coordinates(*fit.x)
            value = float(np.asarray(evaluate(a, b)))
            candidates.append((value, float(a), float(b)))
            successes += int(fit.success)
    low, high = min(candidates), max(candidates)
    return {'minimum': low[0], 'maximum': high[0], 'pv': high[0]-low[0],
            'minimum_xy_m': list(low[1:]), 'maximum_xy_m': list(high[1:]),
            'successful_optimizations': successes,
            'method': 'multistart bounded SLSQP on normalized polar aperture; numerical extrema'}
