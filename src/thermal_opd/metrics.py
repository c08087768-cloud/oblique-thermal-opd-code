"""Aperture selection and optical-path-difference metrics."""

import numpy as np


def circular_aperture_mask(x, y, aperture_radius):
    """Return a boundary-inclusive circular mask for broadcastable x,y (m)."""
    if not np.isfinite(aperture_radius) or aperture_radius <= 0:
        raise ValueError("aperture_radius must be finite and positive")
    x, y = np.broadcast_arrays(x, y)
    return x**2 + y**2 <= aperture_radius**2


def opd_peak_to_valley(opd, aperture_mask):
    """Return max(OPD)-min(OPD) on the supplied aperture, in the input units.

    The mask must have the same shape as opd. All selected values must be
    finite; values outside the aperture are excluded, including NaNs.
    """
    opd = np.asarray(opd)
    aperture_mask = np.asarray(aperture_mask, dtype=bool)
    if opd.shape != aperture_mask.shape:
        raise ValueError("opd and aperture_mask must have the same shape")
    selected = opd[aperture_mask]
    if selected.size == 0:
        raise ValueError("aperture is empty")
    if not np.all(np.isfinite(selected)):
        raise ValueError("OPD values inside aperture must be finite")
    return float(np.max(selected) - np.min(selected))


def opd_metrics(opd,aperture_mask):
    """Equal-area sampled PV, mean and piston-removed RMS, all in meters.

    Intended for uniform Cartesian aperture grids. This is not a continuous
    aperture extremum or an area-weighted metric for irregular sample points.
    """
    pv=opd_peak_to_valley(opd,aperture_mask)
    values=np.asarray(opd)[np.asarray(aperture_mask,dtype=bool)]
    mean=float(np.mean(values))
    return dict(pv_m=pv,mean_opd_m=mean,piston_removed_rms_m=float(np.sqrt(np.mean((values-mean)**2))))


def azimuthal_omission_metrics(full_opd, axisymmetric_thermal_opd, aperture_mask):
    """Quantify omission of m>0 THERMAL orders on one uniform Cartesian pupil.

    Both fields use the SAME refracted paths and pupil. The m=0 thermal
    approximation need not be an axisymmetric optical field. RMS and Linf of
    the piston-removed difference are normalized by full-field PV, not energy
    fractions. All outputs are dimensionless. At zero full PV no ratio exists.
    Added 2026-11-03; known-answer/null/corruption tests accompany this function.
    """
    mask = np.asarray(aperture_mask, dtype=bool)
    full, zero = np.asarray(full_opd), np.asarray(axisymmetric_thermal_opd)
    if full.shape != zero.shape or full.shape != mask.shape:
        raise ValueError('Both fields and the pupil must have the same shape')
    pv = opd_peak_to_valley(full, mask)
    opd_peak_to_valley(zero, mask)  # validate finite values on the same pupil
    if pv <= 0:
        raise ValueError('A positive full-field PV is required for normalization')
    delta = full[mask] - zero[mask]
    delta = delta - np.mean(delta)
    return dict(rms_over_full_pv=float(np.sqrt(np.mean(delta**2))/pv),
                linf_over_full_pv=float(np.max(np.abs(delta))/pv),
                axisymmetric_thermal_pv_over_full_pv=float(np.ptp(zero[mask])/pv))
