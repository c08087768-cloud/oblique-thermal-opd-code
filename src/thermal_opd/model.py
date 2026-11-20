"""Compose existing modal functions, with numerical inputs supplied by scripts."""

from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

from thermal_opd.eigenmodes import radial_eigenvalues
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.projection import fourier_bessel_coefficients


@dataclass(frozen=True)
class ModalResolution:
    M: int
    N: int
    projection_r: int
    projection_phi: int
    ray_points: int


@dataclass(frozen=True)
class ThermalModel:
    """Fixed problem description; construction performs no modal preparation."""
    parameters: object
    angle_deg: float
    resolution: ModalResolution

    def __post_init__(self):
        from thermal_opd.opd import refracted_angle
        p = self.parameters
        refracted_angle(np.deg2rad(self.angle_deg),p.incident_refractive_index,p.substrate_refractive_index)
        if not all(np.isfinite(v) and v>0 for v in (p.radius,p.thickness,p.conductivity,p.beam_radius)):
            raise ValueError('Require positive finite geometry, conductivity and beam radius')
        if not np.isfinite(p.absorbed_power) or p.absorbed_power<0 or not np.isfinite(p.h_eff) or p.h_eff<0:
            raise ValueError('Require finite nonnegative heating and heat transfer')

    def prepare(self):
        """Compute shared roots/source once for this PreparedModel; no axial modes."""
        c,roots = solve_modal_source(self.parameters,self.angle_deg,self.resolution)
        return PreparedModel(self.parameters,self.angle_deg,self.resolution,c,roots)


@dataclass(frozen=True)
class PreparedModel:
    """Common immutable modal data, composed with either solution algorithm."""
    parameters: object
    angle_deg: float
    resolution: ModalResolution
    coefficients: object
    roots: object

    def __post_init__(self):
        ThermalModel(self.parameters,self.angle_deg,self.resolution)
        if not self.coefficients or set(self.coefficients)!=set(self.roots):
            raise ValueError('Require matching nonempty coefficient/root dictionaries')
        for name in ('coefficients','roots'):
            copied = {}
            for m,values in getattr(self,name).items():
                a = np.array(values,dtype=float,copy=True)
                if a.ndim!=1 or not a.size or not np.all(np.isfinite(a)):
                    raise ValueError('Require finite nonempty modal vectors')
                if name=='roots' and np.any(a<=0):
                    raise ValueError('Radial roots must be positive')
                a.flags.writeable = False
                copied[m] = a
            object.__setattr__(self,name,MappingProxyType(copied))
        if any(self.coefficients[m].shape!=self.roots[m].shape for m in self.roots):
            raise ValueError('Coefficient and root lengths differ')

    @property
    def refraction_angle(self):
        from thermal_opd.opd import refracted_angle
        p = self.parameters
        return refracted_angle(np.deg2rad(self.angle_deg),p.incident_refractive_index,p.substrate_refractive_index)

    def solve_steady(self):
        from thermal_opd.solutions import SteadySolution
        return SteadySolution(self)

    def solve_transient(self, *, num_axial_modes, history=None):
        from thermal_opd.transient import TransientSolution
        return TransientSolution(self,num_axial_modes=num_axial_modes,history=history)


def solve_modal_source(parameters, angle_deg, resolution):
    """Return source coefficients and radial roots; no temperature interpolation."""
    p, s = parameters, resolution
    if s.M < 0 or s.M % 2 or min(s.N, s.ray_points) < 1 or s.projection_r < 2 or s.projection_phi <= 2*s.M:
        raise ValueError('Require even nonnegative M and adequately sampled projection')
    # Exactly normal incidence is axisymmetric; numerical cosine sums should
    # not create artificial angular modes at roundoff amplitude.
    orders = (0,) if angle_deg == 0 else range(0, s.M + 1, 2)
    roots = {m: radial_eigenvalues(m, p.radius, p.conductivity, p.h_eff, s.N) for m in orders}
    coefficients = project_modal_source(p,angle_deg,roots,num_r=s.projection_r,num_phi=s.projection_phi)
    return coefficients, roots


def project_modal_source(parameters,angle_deg,roots,*,num_r,num_phi):
    """Shared projection on supplied roots, also useful to cached validation sweeps."""
    p=parameters
    r=np.linspace(0,p.radius,num_r)
    phi=np.linspace(0,2*np.pi,num_phi,endpoint=False)
    q=oblique_gaussian_heat_flux(r[:,None]*np.cos(phi),r[:,None]*np.sin(phi),
                               p.beam_radius,np.deg2rad(angle_deg),p.absorbed_power)
    return {m:fourier_bessel_coefficients(q,r,phi,m,values) for m,values in roots.items()}
