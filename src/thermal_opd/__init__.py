"""Public model/solution interfaces; optional FEM and plotting stay opt-in."""
from thermal_opd.model import ThermalModel,PreparedModel,ModalResolution
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.solutions import SteadySolution
from thermal_opd.transient import TransientSolution,PowerHistory

__all__=['ThermalModel','PreparedModel','ModalResolution','PhysicalParameters',
         'SteadySolution','TransientSolution','PowerHistory']
