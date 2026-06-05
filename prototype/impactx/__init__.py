"""All-pyAMReX prototype of ImpactX.

A pure-Python reimplementation of the central ``ImpactX`` simulation class and
the AMReX classes it depends on, written entirely on top of pyAMReX. It is API
compatible with the compiled ``impactx`` package for the basic FODO example
(``examples/fodo/run_fodo.py``), which it can run unchanged.

Put this package's parent directory first on ``PYTHONPATH`` so that
``import impactx`` resolves here instead of the compiled package, e.g.::

    PYTHONPATH=/home/axel/src/impactx/prototype python examples/fodo/run_fodo.py
"""

import amrex.space3d as amr

from . import distribution, elements
from .amr_core_data import AmrCoreData
from .particle_container import ImpactXParticleContainer

__all__ = ["ImpactX", "distribution", "elements"]


class ImpactX:
    """The central ImpactX simulation object (pure-Python / pyAMReX)."""

    def __init__(self):
        if not amr.initialized():
            amr.initialize([])
        self._space_charge = False
        self.slice_step_diagnostics = True
        self.diagnostics = True
        self.lattice = elements.Lattice()
        self.periods = 1

        self.amr_data = None
        self._beam = None
        self._initialized = False

    # --- configuration -----------------------------------------------------
    @property
    def space_charge(self):
        return self._space_charge

    @space_charge.setter
    def space_charge(self, value):
        # accept bool (legacy) or string algorithm name, like the C++ binding
        if isinstance(value, bool):
            self._space_charge = value
        elif isinstance(value, str):
            self._space_charge = value.lower() not in ("false", "off", "")
        else:
            self._space_charge = bool(value)
        if self._space_charge:
            raise NotImplementedError(
                "This prototype currently supports space_charge = False only."
            )

    # --- grid / beam setup -------------------------------------------------
    def init_grids(self):
        """Initialize the AMReX grids via the AmrCore trampoline."""
        self.amr_data = AmrCoreData()
        self.amr_data.init_from_scratch(0.0)
        self._beam = ImpactXParticleContainer(self.amr_data)
        self._initialized = True

    @property
    def beam(self):
        if self._beam is None:
            raise RuntimeError("init_grids() must be called before accessing beam.")
        return self._beam

    def add_particles(self, bunch_charge, distr, npart, spin_distr=None):
        """Draw ``npart`` particles from ``distr`` and add them to the beam."""
        ref = self.beam.ref
        x, y, t, px, py, pt = distr.sample(npart)
        self.beam.add_n_particles(x, y, t, px, py, pt, ref.qm_ratio_SI(), bunch_charge)

    # --- run ---------------------------------------------------------------
    def _validate(self):
        ref = self.beam.ref
        if ref.kin_energy_MeV() == 0.0:
            raise RuntimeError("Reference particle energy not set.")
        if self.beam.num_particles() <= 0:
            raise RuntimeError("No particles in the beam.")
        if len(self.lattice) == 0:
            raise RuntimeError("Lattice is empty.")

    def track_particles(self):
        """Run the particle tracking loop over the lattice.

        Maintains a global slice-step counter that increments before each push,
        exactly as in the C++ tracking loop; diagnostic elements use it as their
        openPMD iteration index (so the first monitor writes iteration 1).
        """
        self._validate()
        step = 0
        for _period in range(self.periods):
            for element in self.lattice:
                for _slice in range(element.nslice):
                    step += 1
                    element.apply_slice(self.beam, step)

    # --- shutdown ----------------------------------------------------------
    def finalize(self):
        # flush any open diagnostics (BeamMonitor series)
        for element in self.lattice:
            fin = getattr(element, "finalize", None)
            if callable(fin):
                fin()
        self._beam = None
        self.amr_data = None
        self.lattice = elements.Lattice()
        if amr.initialized():
            amr.finalize()
