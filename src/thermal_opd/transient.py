"""Surface-driven diffusion using existing steady modes and ray evaluation.

Axial homogeneous Robin eigenfunctions supply only the decaying correction.
The exact steady axial response completes the high-j tail.  This is a
spectral acceleration of S4's convolution, not one exponential per radial
mode.  For a step at t=0 the initial value is explicitly zero.  A finite J
does not resolve the limit t->0+: verify J at the earliest positive time.
"""
from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.optimize import brentq

from thermal_opd.steady import axial_response
from thermal_opd.modal_evaluation import modal_temperature, modal_temperature_gradient
from thermal_opd.opd import modal_refracted_opd, modal_refracted_opd_series, prepare_modal_ray_series, _ray_settings


class AxialModes:
    """X_j(0)=1, X'_j(0)=hf/k, X'_j(L)=-hb/k X_j(L)."""
    def __init__(self, thickness, conductivity, h_front, h_back, count):
        count = index(count)
        if count < 1 or not all(np.isfinite(x) and x > 0 for x in (thickness, conductivity)):
            raise ValueError('Require positive thickness, conductivity and axial count')
        if not all(np.isfinite(x) and x >= 0 for x in (h_front, h_back)):
            raise ValueError('Robin coefficients must be finite and nonnegative')
        self.thickness = thickness
        self.front_ratio = h_front/conductivity
        a, b = h_front*thickness/conductivity, h_back*thickness/conductivity
        if a == b == 0:
            x = np.arange(count)*np.pi
        else:
            # Phase equation brackets exactly one root per j, including the
            # small fundamental root which a sign scan can otherwise miss.
            x = np.array([brentq(lambda v: v-np.arctan2(a,v)-np.arctan2(b,v)-j*np.pi,
                                 j*np.pi, (j+1)*np.pi, xtol=1e-14)
                          for j in range(count)])
        self.beta = x/thickness
        ratio = np.divide(self.front_ratio, self.beta, out=np.zeros(count), where=self.beta!=0)
        self.norm = thickness/2*(1+np.sinc(2*x/np.pi)) + thickness/2*ratio**2*(1-np.sinc(2*x/np.pi))
        self.norm += np.divide(ratio*np.sin(x)**2, self.beta, out=np.zeros(count), where=self.beta!=0)

    def values(self, z):
        z = np.asarray(z, dtype=float)
        b = self.beta.reshape((-1,)+(1,)*z.ndim)
        return np.cos(b*z) + self.front_ratio*z*np.sinc(b*z/np.pi)

    def derivatives(self, z):
        """Analytic X'_j(z), including the insulated beta=0 mode."""
        z = np.asarray(z,dtype=float)
        b = self.beta.reshape((-1,)+(1,)*z.ndim)
        return -b*np.sin(b*z)+self.front_ratio*np.cos(b*z)


@dataclass(frozen=True)
class PowerHistory:
    """Normalized front power: step, min(t/tau,1), or 1-exp(-t/tau)."""
    kind: str = 'step'
    tau: float = 1.0

    def __post_init__(self):
        if self.kind not in ('step', 'linear', 'exponential') or not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('Require a supported history and finite positive ramp time')

    def value(self, time):
        time = _time(time)
        if self.kind == 'step':
            return float(time > 0)
        if self.kind == 'linear':
            return min(time/self.tau, 1.)
        return float(-np.expm1(-time/self.tau))

    def deficit(self, rates, time):
        """g(t)-d*int exp[-d(t-s)]g(s)ds; stable also for d=1/tau."""
        time = _time(time)
        rates = np.asarray(rates, dtype=float)
        if not np.all(np.isfinite(rates) & (rates > 0)):
            raise ValueError('Decay rates must be finite and positive')
        if time == 0:
            return np.zeros_like(rates)
        if self.kind == 'step':
            return np.exp(-rates*time)
        if self.kind == 'linear':
            return -np.expm1(-rates*min(time,self.tau))/(rates*self.tau)*np.exp(-rates*max(time-self.tau,0.))
        a = 1/self.tau
        difference = np.abs(rates-a)
        # Using the smaller decay outside prevents overflow in either branch.
        ratio = np.divide(-np.expm1(-difference*time), difference,
                          out=np.full_like(rates,time), where=difference!=0)
        return a*np.exp(-np.minimum(rates,a)*time)*ratio


def _time(time):
    if np.ndim(time) != 0 or not np.isfinite(time) or time < 0:
        raise ValueError('Time must be a finite nonnegative scalar')
    return float(time)


@dataclass(frozen=True,init=False)
class TransientSolution:
    """Transient axial state and field evaluation on one prepared problem.

    A_inf_mnj = C_mn/[k*(lambda_mn^2+beta_j^2)*norm_j], since X_j(0)=1.
    C_mn is the existing front-flux projection, in W/m^2.  The response is
    g(t)*Z_steady - sum_j A_inf_mnj X_j [g(t)-response_j(t)].
    Unretained fast modes are completed by g(t)*Z_steady; J convergence is
    required, especially close to step onset.  No time stepping is used.
    """
    prepared: object
    history: PowerHistory
    axial: AxialModes
    rates: dict
    amplitudes: dict
    slowest_decay_rate: float

    def __init__(self, prepared, *, num_axial_modes, history=None):
        p,coefficients,eigenvalues = prepared.parameters,prepared.coefficients,prepared.roots
        # Geometry, conductivity and modal arrays are validated by PreparedModel.
        if not all(np.isfinite(v) and v > 0 for v in (p.density,p.specific_heat)):
            raise ValueError('Thermal properties must be finite and positive')
        axial = AxialModes(p.thickness,p.conductivity,p.h_eff,p.h_eff,num_axial_modes)
        rates, amplitudes = {}, {}
        for m, values in coefficients.items():
            lam = eigenvalues[m]
            squared = lam[:,None]**2 + axial.beta[None,:]**2
            rates[m] = p.thermal_diffusivity*squared
            amplitudes[m] = values[:,None]/(p.conductivity*squared*axial.norm[None,:])
        object.__setattr__(self,'prepared',prepared)
        object.__setattr__(self,'history',PowerHistory() if history is None else history)
        object.__setattr__(self,'axial',axial)
        object.__setattr__(self,'rates',rates)
        object.__setattr__(self,'amplitudes',amplitudes)
        object.__setattr__(self,'slowest_decay_rate',float(min(np.min(v) for v in rates.values())))

    def _response(self, time, *, derivative=False):
        time = _time(time)
        prepared = self.prepared
        p, g = prepared.parameters, self.history.value(time)
        deficits = {m:self.history.deficit(rates,time) for m,rates in self.rates.items()}
        cached_z, cached_basis = None, None
        def response(m,n,z):
            nonlocal cached_z, cached_basis
            if time == 0:
                return np.zeros_like(z,dtype=float)
            if cached_z is not z:
                cached_z = z
                cached_basis = self.axial.derivatives(z) if derivative else self.axial.values(z)
            steady = axial_response(z,prepared.roots[m][n],prepared.coefficients[m][n],
                                    p.conductivity,p.thickness,p.h_eff,p.h_eff)[int(derivative)]
            return g*steady - np.tensordot(self.amplitudes[m][n]*deficits[m][n],cached_basis,axes=(0,0))
        return response

    def temperature(self, r, phi, z, time):
        return modal_temperature(r,phi,z,self.prepared.coefficients,self.prepared.roots,self._response(time))

    def temperature_gradient(self, r, phi, z, time):
        """Return analytic dT/dr and dT/dz (K/m), reusing the steady spatial sum."""
        value,dz = self._response(time),self._response(time,derivative=True)
        return modal_temperature_gradient(r,phi,z,self.prepared.coefficients,self.prepared.roots,
                                          lambda m,n,depth:(value(m,n,depth),dz(m,n,depth)))

    def opd(self, x, y, time, *, num_ray_points=None, point_chunk_size=512):
        p = self.prepared.parameters
        return modal_refracted_opd(x,y,self.prepared.coefficients,self.prepared.roots,self._response(time),
                                  p.thickness,p.dn_dT,**_ray_settings(self.prepared,num_ray_points,point_chunk_size))

    def opd_series(self,x,y,times,*,num_ray_points=None,point_chunk_size=512):
        """Evaluate (time,*point_shape) while sharing spatial ray work."""
        times = np.asarray(times,dtype=float)
        if times.ndim!=1 or not times.size:
            raise ValueError('Require a nonempty one-dimensional time array')
        responses = [self._response(t) for t in times]
        p = self.prepared.parameters
        return modal_refracted_opd_series(x,y,self.prepared.coefficients,self.prepared.roots,
            lambda m,n,z:np.stack([response(m,n,z) for response in responses]),times.size,
            p.thickness,p.dn_dT,**_ray_settings(self.prepared,num_ray_points,point_chunk_size))

    def prepare_opd(self,time,*,num_ray_points=None,point_chunk_size=512):
        """Prepared scalar-time optical field for numerical aperture extrema."""
        p,response = self.prepared.parameters,self._response(time)
        evaluator = prepare_modal_ray_series(self.prepared.coefficients,self.prepared.roots,
            lambda m,n,z:response(m,n,z)[None,:],1,p.thickness,p.dn_dT,
            **_ray_settings(self.prepared,num_ray_points,point_chunk_size))
        return lambda x,y:evaluator(x,y)[0]
