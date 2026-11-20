"""Physical inputs for the manuscript, expressed in SI units.

Numerical settings (modal truncation and projection grids) belong in the
calling script. The incidence angle is also supplied by the calling script.
"""

from dataclasses import dataclass

from scipy.constants import Stefan_Boltzmann
from thermal_opd.materials import corning_7980_reference


_REFERENCE_INDEX, _REFERENCE_DN_DT = corning_7980_reference(1064e-9, 300.)


@dataclass(frozen=True)
class PhysicalParameters:
    """Nominal geometry, material, environment, and optical inputs.

    ``absorbed_power`` is assigned to the Gaussian on the full plane;
    power intercepted by the finite substrate must be evaluated separately.
    ``dn_dT`` and ``substrate_refractive_index`` come from the same absolute
    Corning7980 Sellmeier fit at 1064 nm and 300 K. Both are held constant;
    changing wavelength or reference temperature requires updating both.
    """

    radius: float = 12.7e-3              # m
    thickness: float = 5.0e-3            # m
    conductivity: float = 1.38           # W/(m K)
    # Added for transient runs, not values provided by the manuscript.
    # Heraeus Quartz Glass Plates datasheet: density 2.203 g/cm3 and
    # mean cp over 0..100 C. Nominal constant model, not a 300 K measurement.
    # https://www.heraeus-conamic.com/media/Media/Documents/Products_and_Solutions/SEMI/EN/Quartz_Glass_Plates_EN.pdf
    density: float = 2203.0              # kg/m^3
    specific_heat: float = 772.0         # J/(kg K), mean 0..100 C
    h_conv: float = 10.0                 # W/(m^2 K)
    emissivity: float = 0.85
    ambient_temperature: float = 300.0   # K
    absorbed_power: float = 110e-3       # W
    beam_radius_ratio: float = 0.30
    dn_dT: float = _REFERENCE_DN_DT      # 1/K, derivative of absolute index
    wavelength: float = 1064e-9          # m
    incident_refractive_index: float = 1.0  # Air approximation for Snell's law.
    # Leviton--Frey (2006), NASA NTRS20070018851 Table3; nominal fit value.
    substrate_refractive_index: float = _REFERENCE_INDEX

    @property
    def beam_radius(self) -> float:
        """Return the 1/e^2 intensity radius normal to beam propagation (m)."""
        return self.beam_radius_ratio * self.radius

    @property
    def thermal_diffusivity(self) -> float:
        """Constant-model k/(rho*cp), in m^2/s."""
        return self.conductivity / (self.density * self.specific_heat)

    @property
    def h_eff(self) -> float:
        """Return convection plus linearized radiation at ambient (W/(m^2 K))."""
        return (
            self.h_conv + 4 * self.emissivity * Stefan_Boltzmann
            * self.ambient_temperature**3
        )
