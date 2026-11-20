"""Steady solution interface composed with the independent numerical kernels.

No verification, plotting, file writes or FEM work is triggered here.
All temperatures are rises above ambient; OPD follows the fixed refracted ray.
"""
from dataclasses import dataclass
from thermal_opd.steady import steady_temperature,steady_temperature_gradient,axial_response
from thermal_opd.opd import steady_refracted_opd,prepare_modal_ray_series,_ray_settings


@dataclass(frozen=True)
class SteadySolution:
    prepared: object

    def _thermal_args(self):
        p = self.prepared.parameters
        return (self.prepared.coefficients,self.prepared.roots,p.conductivity,p.thickness,p.h_eff,p.h_eff)

    def temperature(self,r,phi,z):
        return steady_temperature(r,phi,z,*self._thermal_args())

    def temperature_gradient(self,r,phi,z):
        return steady_temperature_gradient(r,phi,z,*self._thermal_args())

    def opd(self,x,y,*,num_ray_points=None,point_chunk_size=512):
        return steady_refracted_opd(x,y,*self._thermal_args(),self.prepared.parameters.dn_dT,
                                    **_ray_settings(self.prepared,num_ray_points,point_chunk_size))

    def prepare_opd(self,*,num_ray_points=None,point_chunk_size=512):
        """Prepare shared axial weights once for repeated aperture queries."""
        p,c,roots = self.prepared.parameters,self.prepared.coefficients,self.prepared.roots
        evaluator = prepare_modal_ray_series(c,roots,
            lambda m,n,z:axial_response(z,roots[m][n],c[m][n],p.conductivity,p.thickness,p.h_eff,p.h_eff)[0][None,:],
            1,p.thickness,p.dn_dT,**_ray_settings(self.prepared,num_ray_points,point_chunk_size))
        return lambda x,y:evaluator(x,y)[0]
