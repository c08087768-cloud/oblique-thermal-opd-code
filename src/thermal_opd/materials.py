"""Matched absolute optical inputs from Leviton--Frey's Corning 7980 fit.

Source: NASA NTRS 20070018851, Table 3 (2006 author manuscript, PDF page 6).
The three-term Sellmeier coefficients have fourth-order temperature dependence.
Use this for a reference index and its temperature derivative at the SAME
wavelength and temperature. The fit is not a material calibration or a model
of thermoelastic OPD. Domain: 0.4--2.6 micrometers and 30--300 K (table limits).
The manuscript freezes both outputs at 1064 nm and 300 K; it does not evaluate
this fit at locally heated temperatures above its tabulated domain.
Validation: source Table 4 indices and a separately evaluated derivative;
see tests/test_optical_material_and_omission.py. Added 2026-11-03.
"""
import math


# Columns: S1, S2, S3, lambda1, lambda2, lambda3; rows: T**0 through T**4.
_COEFFICIENTS = (
    (1.10127, 1.78752e-5, .793552, -.0890600, .297562, 9.34454),
    (-4.94251e-5, 4.76391e-5, -1.27815e-3, 9.08730e-6, -8.59578e-4, -7.09788e-3),
    (5.27414e-7, -4.49019e-7, 1.84595e-5, -6.53638e-8, 6.59069e-6, 1.01968e-4),
    (-1.59700e-9, 1.44546e-9, -9.20275e-8, 7.77072e-11, -1.09482e-8, -5.07660e-7),
    (1.75949e-12, -1.57223e-12, 1.48829e-10, 6.84605e-14, 7.85145e-13, 8.21348e-10),
)


def corning_7980_reference(wavelength_m, temperature_K):
    """Return (absolute n, dn_absolute/dT [1/K]) from the published fit."""
    lam, t = wavelength_m * 1e6, temperature_K
    if not (math.isfinite(lam) and .4 <= lam <= 2.6 and
            math.isfinite(t) and 30 <= t <= 300):
        raise ValueError('Corning7980 fit requires 0.4--2.6 um and 30--300 K')
    value = [math.fsum(_COEFFICIENTS[j][i]*t**j for j in range(5)) for i in range(6)]
    slope = [math.fsum(j*_COEFFICIENTS[j][i]*t**(j-1) for j in range(1, 5)) for i in range(6)]
    n = math.sqrt(1 + math.fsum(value[i]*lam**2/(lam**2-value[i+3]**2) for i in range(3)))
    derivative = math.fsum(
        slope[i]*lam**2/(lam**2-value[i+3]**2) +
        value[i]*lam**2*2*value[i+3]*slope[i+3]/(lam**2-value[i+3]**2)**2
        for i in range(3)) / (2*n)
    return n, derivative
