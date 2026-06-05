"""Pure-Python port of impactx::initialization::AmrCoreData.

This subclasses the freshly-bound ``amrex.AmrCore`` (a pybind11 trampoline)
and overrides its pure-virtual methods, exactly mirroring the C++ class in
src/initialization/AmrCoreData.{H,cpp}. For the space-charge-free FODO case
only ``make_new_level_from_scratch`` is actually invoked (at level 0).
"""

import amrex.space3d as amr


class AmrCoreData(amr.AmrCore):
    """AMR core that owns the simulation grids and (later) space-charge fields.

    Built as ``one_box_per_rank`` with ``max_level = 0`` and a non-periodic,
    Cartesian level-0 geometry, matching ImpactX's default grid setup in
    src/initialization/InitAmrCore.cpp.
    """

    def __init__(
        self, n_cell=(16, 16, 16), prob_lo=(-1.0, -1.0, -1.0), prob_hi=(1.0, 1.0, 1.0)
    ):
        rb = amr.RealBox(
            prob_lo[0], prob_lo[1], prob_lo[2], prob_hi[0], prob_hi[1], prob_hi[2]
        )
        n_cell_v = amr.Vector_int(list(n_cell))
        ref_ratios = amr.Vector_IntVect([])  # max_level = 0 -> no refinement
        is_periodic = [0, 0, 0]
        coord = 0  # Cartesian

        super().__init__(rb, 0, n_cell_v, coord, ref_ratios, is_periodic)

        # per-level grids, populated by make_new_level_from_scratch
        self._levels = {}

        # space-charge data (unused while space_charge is off); kept for parity
        self.rho = {}
        self.phi = {}
        self.space_charge_field = {}

    # --- AmrCore pure virtuals --------------------------------------------
    def make_new_level_from_scratch(self, lev, time, ba, dm):
        # Without space charge there are no MultiFabs to allocate; we just
        # record the box array & distribution mapping for this level.
        self._levels[lev] = (ba, dm)

    def make_new_level_from_coarse(self, lev, time, ba, dm):
        raise NotImplementedError("MakeNewLevelFromCoarse: not implemented")

    def remake_level(self, lev, time, ba, dm):
        raise NotImplementedError("RemakeLevel: not implemented")

    def clear_level(self, lev):
        self._levels.pop(lev, None)
        self.rho.pop(lev, None)
        self.phi.pop(lev, None)
        self.space_charge_field.pop(lev, None)

    def error_est(self, lev, tags, time, ngrow):
        # No tagging: single-level simulation.
        pass
