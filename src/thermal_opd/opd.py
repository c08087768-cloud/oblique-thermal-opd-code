"""Thermo-optic OPD on straight refracted rays through the rear plane face."""

import operator
import numpy as np
from numpy.polynomial.legendre import leggauss
from thermal_opd.eigenmodes import bessel_j

from thermal_opd.steady import axial_response


def _ray_settings(prepared, num_ray_points, point_chunk_size):
    """Resolve solution ray geometry and quadrature from its prepared problem."""
    return dict(radius=prepared.parameters.radius, refraction_angle=prepared.refraction_angle,
                num_ray_points=prepared.resolution.ray_points if num_ray_points is None else num_ray_points,
                point_chunk_size=point_chunk_size)


def _forward_angle(angle):
    if not np.isfinite(angle) or abs(angle) >= np.pi/2:
        raise ValueError("Angle must be finite and strictly between -pi/2 and pi/2")


def refracted_angle(incidence_angle, n_incident, n_substrate):
    """Return the internal signed angle (rad) from Snell's law in the y-z plane."""
    _forward_angle(incidence_angle)
    if not np.all(np.isfinite([n_incident, n_substrate])) or min(n_incident, n_substrate) <= 0:
        raise ValueError("Refractive indices must be finite and positive")
    sine = n_incident / n_substrate * np.sin(incidence_angle)
    if abs(sine) >= 1:
        raise ValueError("No forward transmitted ray at this incidence angle")
    return float(np.arcsin(sine))


def transmitted_ray_mask(x0, y0, radius, thickness, refraction_angle):
    """Select rays with entry z=0 and exit z=L inside the physical cylinder.

    x0,y0 are front-face coordinates (m), broadcastable. A ray travels along
    x=x0, y=y0+z*tan(angle); convexity makes endpoint inclusion sufficient.
    Side-exiting rays are excluded from rear-face transmitted OPD.
    """
    _forward_angle(refraction_angle)
    if not np.all(np.isfinite([radius, thickness])) or min(radius, thickness) <= 0:
        raise ValueError("Cylinder radius and thickness must be finite and positive")
    x0, y0 = np.broadcast_arrays(np.asarray(x0, dtype=float), np.asarray(y0, dtype=float))
    y_exit = y0 + thickness * np.tan(refraction_angle)
    limit = radius**2 * (1 + 8*np.finfo(float).eps)
    return np.isfinite(x0) & np.isfinite(y0) & (x0**2 + y0**2 <= limit) & (x0**2 + y_exit**2 <= limit)


def steady_refracted_opd(
    x0, y0, coefficients, eigenvalues, conductivity, thickness, h_front, h_back,
    dn_dT, *, radius, refraction_angle, num_ray_points, point_chunk_size,
):
    """Return dn_dT * integral_ray DeltaT ds (m), labelled by front-face x0,y0.

    Straight rays use the unheated substrate's Snell angle. At depth z,
    y=y0+z*tan(angle), x=x0, and ds=dz/cos(angle). Modal temperature is
    evaluated directly along the ray, using Gauss-Legendre integration in z.
    This is the first-order thermo-optic term on a fixed reference ray; ray
    bending by the thermal index gradient and mechanical effects are omitted.

    All rays must reach z=L inside the cylinder; otherwise raise ValueError.
    Call transmitted_ray_mask before evaluating a sampled entrance aperture.
    Quadrature count and memory chunk size are explicit caller settings.
    Output preserves broadcast shape and raw piston relative to unheated glass.
    """
    def response(m, n, z):
        return axial_response(z, eigenvalues[m][n], coefficients[m][n],
                              conductivity, thickness, h_front, h_back)[0]
    return modal_refracted_opd(x0, y0, coefficients, eigenvalues, response, thickness, dn_dT,
                              radius=radius, refraction_angle=refraction_angle,
                              num_ray_points=num_ray_points, point_chunk_size=point_chunk_size)


def modal_refracted_opd(x0, y0, coefficients, eigenvalues, response, thickness, dn_dT,
                        *, radius, refraction_angle, num_ray_points, point_chunk_size):
    """Shared ray geometry, quadrature and spatial modes for any axial response."""
    return modal_refracted_opd_series(x0,y0,coefficients,eigenvalues,
        lambda m,n,z:np.asarray(response(m,n,z))[None,:],1,thickness,dn_dT,
        radius=radius,refraction_angle=refraction_angle,num_ray_points=num_ray_points,
        point_chunk_size=point_chunk_size)[0]


def modal_refracted_opd_series(x0,y0,coefficients,eigenvalues,response,num_frames,thickness,dn_dT,
                               *,radius,refraction_angle,num_ray_points,point_chunk_size):
    """Reuse every ray's Bessel samples across frames; response returns (frame,z)."""
    return prepare_modal_ray_series(coefficients,eigenvalues,response,num_frames,thickness,dn_dT,
        radius=radius,refraction_angle=refraction_angle,num_ray_points=num_ray_points,
        point_chunk_size=point_chunk_size)(x0,y0)


def prepare_modal_ray_series(coefficients,eigenvalues,response,num_frames,thickness,dn_dT,
                              *,radius,refraction_angle,num_ray_points,point_chunk_size):
    """Prepare time/axial weights once for dense grids or repeated PV queries."""
    num_frames = operator.index(num_frames)
    if num_frames<1:
        raise ValueError('Require at least one frame')
    num_ray_points = operator.index(num_ray_points)
    point_chunk_size = operator.index(point_chunk_size)
    if num_ray_points <= 0 or point_chunk_size <= 0:
        raise ValueError("Ray quadrature count and point chunk size must be positive")
    transmitted_ray_mask(0.,0.,radius,thickness,refraction_angle)  # validate geometry
    nodes, weights = leggauss(num_ray_points)
    z = (nodes + 1) * thickness/2
    path_weights = weights * thickness/2 / np.cos(refraction_angle) * dn_dT
    mode_weights = {
        m: [response(m, n, z)*path_weights for n in range(len(values))]
        for m, values in coefficients.items()
    }
    def evaluate(x0,y0):
        return _integrate_ray_modes(x0,y0,eigenvalues,mode_weights,z,num_frames,thickness,
                                    radius,refraction_angle,point_chunk_size)
    return evaluate


def _integrate_ray_modes(x0,y0,eigenvalues,mode_weights,z,num_frames,thickness,radius,refraction_angle,point_chunk_size):
    x0,y0 = np.broadcast_arrays(np.asarray(x0,dtype=float),np.asarray(y0,dtype=float))
    if not np.all(transmitted_ray_mask(x0,y0,radius,thickness,refraction_angle)):
        raise ValueError('All evaluated rays must reach the rear face inside the cylinder')
    x_flat, y_flat = x0.ravel(), y0.ravel()
    output = np.zeros((num_frames,x_flat.size))
    y_shift = z * np.tan(refraction_angle)
    for start in range(0, x_flat.size, point_chunk_size):
        stop = min(start + point_chunk_size, x_flat.size)
        # At zero angle the lateral coordinates are constant; broadcasting
        # one Bessel value over z retains the same Gauss ray quadrature.
        ray_y = y_flat[start:stop, None] + (y_shift if refraction_angle != 0 else 0.)
        ray_r = np.hypot(x_flat[start:stop, None], ray_y)
        ray_phi = np.arctan2(ray_y, x_flat[start:stop, None])
        for m, weighted_responses in mode_weights.items():
            angular = np.cos(m * ray_phi)
            for n, weighted in enumerate(weighted_responses):
                spatial = bessel_j(m,eigenvalues[m][n]*ray_r)*angular
                # Zero angle needs only one lateral sample; integrate the
                # axial response independently with the same Gauss rule.
                if spatial.shape[1]==1:
                    output[:,start:stop] += weighted.sum(axis=1)[:,None]*spatial[:,0]
                else:
                    output[:,start:stop] += np.einsum('pz,tz->tp',spatial,weighted)
    return output.reshape((num_frames,)+x0.shape)
