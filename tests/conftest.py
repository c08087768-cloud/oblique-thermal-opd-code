"""Controlled modal problems for independent numerical tests, without projection."""
import numpy as np
import pytest
from thermal_opd.model import PreparedModel, ModalResolution


@pytest.fixture
def modal_problem():
    def prepare(parameters, coefficients, roots, *, refraction_angle=0., ray_points=48):
        incidence=np.rad2deg(np.arcsin(parameters.substrate_refractive_index /
                                      parameters.incident_refractive_index*np.sin(refraction_angle)))
        M,N=max(coefficients),max(map(len,roots.values()))
        return PreparedModel(parameters,incidence,ModalResolution(M,N,80,max(16,2*M+1),ray_points),coefficients,roots)
    return prepare
