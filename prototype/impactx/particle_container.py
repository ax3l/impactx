"""Pure-Python port of impactx::ImpactXParticleContainer.

Wraps the pyAMReX ``ParticleContainer_pureSoA_11_0_polymorphic`` (the exact
ImpactX particle type: 11 real SoA components, 0 int), constructed from the
AmrCore's ParGDB -- the same construction path as the C++ code
``ImpactXParticleContainer(amr_core->GetParGDB())``.
"""

import numpy as np

import amrex.space3d as amr

from . import _constants as const
from .reference_particle import RefPart

#: SoA real component order (must match set_soa_compile_time_names below and
#: RealSoA in ImpactXParticleContainer.H)
SOA_REAL_NAMES = [
    "position_x",
    "position_y",
    "position_t",
    "momentum_x",
    "momentum_y",
    "momentum_t",
    "spin_x",
    "spin_y",
    "spin_z",
    "qm",
    "weighting",
]
# convenient index constants
IX, IY, IT, IPX, IPY, IPT, ISX, ISY, ISZ, IQM, IW = range(11)


class ImpactXParticleContainer:
    """The beam: an SoA particle container plus a reference particle."""

    def __init__(self, amr_core_data):
        self._amr = amr_core_data  # keep AmrCore (and its ParGDB) alive
        self.pc = amr.ParticleContainer_pureSoA_11_0_polymorphic(
            amr_core_data.get_par_gdb()
        )
        self.pc.arena = amr.The_Arena()
        self.pc.set_soa_compile_time_names(SOA_REAL_NAMES, [])
        self.ref = RefPart()

    # --- particle creation -------------------------------------------------
    def add_n_particles(self, x, y, t, px, py, pt, qm, bunch_charge):
        """Mirror ImpactXParticleContainer::AddNParticles for lev 0, grid 0.

        ``qm`` is the (uniform) charge-to-mass ratio; per-particle weighting is
        ``bunch_charge / q_e / npart``.
        """
        npart = len(x)
        proc = amr.ParallelDescriptor.MyProc()

        tile = self.pc.define_and_return_particle_tile(0, 0, 0)
        old = tile.num_particles
        tile.resize(old + npart)
        soa = tile.get_struct_of_arrays()

        def view(idx):
            return np.array(soa.get_real_data(idx), copy=False)

        view(IX)[old:] = x
        view(IY)[old:] = y
        view(IT)[old:] = t
        view(IPX)[old:] = px
        view(IPY)[old:] = py
        view(IPT)[old:] = pt
        view(ISX)[old:] = 0.0
        view(ISY)[old:] = 0.0
        view(ISZ)[old:] = 0.0
        view(IQM)[old:] = qm
        view(IW)[old:] = bunch_charge / const.q_e / npart

        idcpu = np.array(soa.get_idcpu_data(), copy=False)
        ids = np.arange(old + 1, old + npart + 1, dtype=np.int64)
        cpus = np.full(npart, proc, dtype=np.int32)
        amr.pack_ids(idcpu[old:], ids)
        amr.pack_cpus(idcpu[old:], cpus)

    # --- access ------------------------------------------------------------
    def soa_views(self):
        """Yield writable numpy views of (idcpu, [11 real arrays]) per tile.

        Views are zero-copy into AMReX memory; do not hold them across a
        structural change (resize/redistribute).
        """
        for lev in range(self.pc.finest_level + 1):
            for pti in self.pc.iterator(level=lev):
                soa = pti.soa()
                reals = [np.array(soa.get_real_data(i), copy=False) for i in range(11)]
                idcpu = np.array(soa.get_idcpu_data(), copy=False)
                yield idcpu, reals

    def min_max_positions(self):
        xmin = [np.inf] * 3
        xmax = [-np.inf] * 3
        for _idcpu, reals in self.soa_views():
            for d, idx in enumerate((IX, IY, IT)):
                if reals[idx].size:
                    xmin[d] = min(xmin[d], float(reals[idx].min()))
                    xmax[d] = max(xmax[d], float(reals[idx].max()))
        return xmin, xmax

    def num_particles(self):
        return self.pc.total_number_of_particles()
