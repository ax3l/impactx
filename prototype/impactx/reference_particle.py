"""Pure-Python port of impactx::RefPart (src/particles/ReferenceParticle.H).

Tracks the reference particle attributes. Energy/momentum conventions match
ImpactX exactly: ``pt = -gamma`` (note the negative sign), momenta normalized
by ``m*c`` (i.e. ``beta*gamma``).
"""

import math

from . import _constants as const


class RefPart:
    """The reference particle of the beam."""

    def __init__(self):
        # path length / phase-space state (see ReferenceParticle.H)
        self.s = 0.0  # integrated orbit path length [m]
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.t = 0.0  # clock time * c [m]
        self.px = 0.0  # beta_x * gamma [unitless]
        self.py = 0.0
        self.pz = 0.0
        self.pt = 0.0  # energy, normalized by rest energy (== -gamma)
        self.mass = 0.0  # rest mass [kg]
        self.charge = 0.0  # charge [C]
        self.gyromagnetic_anomaly = 0.0
        self.sedge = 0.0  # value of s at entrance of the current element

    # --- species -----------------------------------------------------------
    def set_species(self, species_name):
        """Set charge, mass and gyromagnetic anomaly for a known species."""
        if species_name == "electron":
            qe, massE, g = -1.0, const.m_e / const.MeV_inv_c2, const.g_anomaly_electron
        elif species_name == "positron":
            qe, massE, g = 1.0, const.m_e / const.MeV_inv_c2, const.g_anomaly_electron
        elif species_name == "proton":
            qe, massE, g = 1.0, const.m_p / const.MeV_inv_c2, const.g_anomaly_proton
        elif species_name == "Hminus":
            qe = -1.0
            massE = (const.m_p + 2.0 * const.m_e) / const.MeV_inv_c2
            g = const.g_anomaly_proton
        else:
            raise RuntimeError(
                f"Unknown species: '{species_name}'. Known species: "
                "electron, positron, proton, Hminus."
            )
        self.set_charge_qe(qe)
        self.set_mass_MeV(massE)
        self.set_gyromagnetic_anomaly(g)
        return self

    # --- mass / charge -----------------------------------------------------
    def set_charge_qe(self, charge_qe):
        self.charge = charge_qe * const.q_e
        return self

    def charge_qe(self):
        return self.charge / const.q_e

    def set_mass_MeV(self, massE):
        if massE == 0.0:
            raise ValueError("set_mass_MeV: Mass cannot be zero!")
        self.mass = massE * const.MeV_inv_c2
        # re-scale pt and pz if an energy was already set
        if self.pt != 0.0:
            self.pt = -self.kin_energy_MeV() / massE - 1.0
            self.pz = math.sqrt(self.pt**2 - 1.0)
        return self

    def mass_MeV(self):
        return self.mass / const.MeV_inv_c2

    def set_gyromagnetic_anomaly(self, g):
        self.gyromagnetic_anomaly = g
        return self

    # --- energy ------------------------------------------------------------
    def set_kin_energy_MeV(self, kin_energy):
        if self.mass == 0.0:
            raise ValueError("set_kin_energy_MeV: Set mass first!")
        self.px = 0.0
        self.py = 0.0
        self.pt = -kin_energy / self.mass_MeV() - 1.0
        self.pz = math.sqrt(self.pt**2 - 1.0)
        return self

    def kin_energy_MeV(self):
        return self.mass_MeV() * (self.gamma() - 1.0)

    # --- derived quantities ------------------------------------------------
    def gamma(self):
        return -self.pt

    def beta(self):
        g = -self.pt
        return math.sqrt(1.0 - 1.0 / g**2)

    def beta_gamma(self):
        return math.sqrt(self.pt**2 - 1.0)

    def qm_ratio_SI(self):
        return self.charge / self.mass
