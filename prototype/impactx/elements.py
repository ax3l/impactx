"""Pure-Python port of ImpactX lattice elements.

Provides the elements used by the FODO example: Drift, Quad and the
openPMD BeamMonitor diagnostic, plus a list-like lattice container. The
phase-space maps mirror src/elements/Drift.H and src/elements/Quad.H; the
reference-particle push is the straight-element (drift-like) map shared by
both.
"""

import math
import os

import numpy as np

from .particle_container import IPT, IPX, IPY, ISX, ISY, ISZ, IT, IW, IX, IY


def _advance_ref(ref, slice_ds):
    """Advance the reference particle through one straight slice."""
    step = slice_ds / math.sqrt(ref.pt**2 - 1.0)
    ref.x += step * ref.px
    ref.y += step * ref.py
    ref.z += step * ref.pz
    ref.t -= step * ref.pt
    ref.s += slice_ds


class _ThickElement:
    """Common base for thick elements with an ``ds`` and slicing."""

    def __init__(self, ds, nslice=1, name=None):
        self.ds = ds
        self.nslice = nslice
        self.name = name

    def _push_slice(self, reals, slice_ds, betgam2):
        raise NotImplementedError

    def apply_slice(self, beam, step):
        """Push the beam (and reference particle) through a single slice.

        Mirrors one iteration of the C++ in-element slice-step loop: advance the
        reference particle, then push all beam particles. ``betgam2`` derives
        from the (energy-invariant) reference ``pt``.
        """
        ref = beam.ref
        slice_ds = self.ds / self.nslice
        betgam2 = ref.pt**2 - 1.0
        _advance_ref(ref, slice_ds)
        for _idcpu, reals in beam.soa_views():
            self._push_slice(reals, slice_ds, betgam2)


class Drift(_ThickElement):
    """A field-free drift."""

    def __init__(self, ds, nslice=1, name=None, **kwargs):
        super().__init__(ds, nslice, name)

    def _push_slice(self, reals, slice_ds, betgam2):
        x, y, t = reals[IX], reals[IY], reals[IT]
        px, py, pt = reals[IPX], reals[IPY], reals[IPT]
        slice_bg = slice_ds / betgam2
        x += slice_ds * px
        y += slice_ds * py
        t += slice_bg * pt
        # px, py, pt unchanged


class Quad(_ThickElement):
    """A normal (magnetic) quadrupole, MADX strength convention ``k`` [1/m^2]."""

    def __init__(self, ds, k, nslice=1, name=None, **kwargs):
        super().__init__(ds, nslice, name)
        self.k = k

    def _push_slice(self, reals, slice_ds, betgam2):
        x, y, t = reals[IX], reals[IY], reals[IT]
        px, py, pt = reals[IPX], reals[IPY], reals[IPT]
        slice_bg = slice_ds / betgam2
        k = self.k

        if k == 0.0:
            x += slice_ds * px
            y += slice_ds * py
            t += slice_bg * pt
            return

        omega = math.sqrt(abs(k))
        s, c = math.sin(omega * slice_ds), math.cos(omega * slice_ds)
        sh, ch = math.sinh(omega * slice_ds), math.cosh(omega * slice_ds)

        if k > 0.0:
            # focusing in x, defocusing in y
            xo = c * x + (s / omega) * px
            pxo = -omega * s * x + c * px
            yo = ch * y + (sh / omega) * py
            pyo = omega * sh * y + ch * py
        else:
            # defocusing in x, focusing in y
            xo = ch * x + (sh / omega) * px
            pxo = omega * sh * x + ch * px
            yo = c * y + (s / omega) * py
            pyo = -omega * s * y + c * py

        x[:] = xo
        px[:] = pxo
        y[:] = yo
        py[:] = pyo
        t += slice_bg * pt


class BeamMonitor:
    """openPMD beam diagnostic, port of ImpactX diagnostics::BeamMonitor.

    Writes the full beam phase space to ``diags/openPMD/<name>.<ext>`` on each
    pass, one openPMD iteration per encounter in the lattice.
    """

    # record name -> SoA index, mirroring name2openPMD splitting of
    # "position_x" into record "position", component "x"
    _RECORDS = {
        "position": [("x", IX), ("y", IY), ("t", IT)],
        "momentum": [("x", IPX), ("y", IPY), ("t", IPT)],
        "spin": [("x", ISX), ("y", ISY), ("z", ISZ)],
    }
    _SCALARS = {"weighting": IW}

    def __init__(
        self, name, backend="default", encoding="g", period_sample_intervals=1
    ):
        self.name = name
        self.nslice = 1
        self.ds = 0.0
        self._backend = backend
        self._encoding = encoding
        self._series = None
        self._step = 0

    def _ensure_series(self):
        if self._series is not None:
            return
        import openpmd_api as io

        ext = {
            "default": "h5",
            "h5": "h5",
            "bp4": "bp",
            "bp": "bp",
            "json": "json",
        }.get(self._backend, self._backend)
        out_dir = os.path.join("diags", "openPMD")
        os.makedirs(out_dir, exist_ok=True)
        enc = {
            "g": io.Iteration_Encoding.group_based,
            "f": io.Iteration_Encoding.file_based,
            "v": io.Iteration_Encoding.variable_based,
        }[self._encoding]
        path = os.path.join(out_dir, f"{self.name}.{ext}")
        self._series = io.Series(path, io.Access.create)
        self._series.set_iteration_encoding(enc)
        self._io = io

    def apply_slice(self, beam, step):
        """Write the current beam to a new openPMD iteration keyed by ``step``."""
        self._step = step
        self._ensure_series()
        io = self._io

        # gather all particles across tiles into contiguous arrays
        cols = {i: [] for i in range(11)}
        ids = []
        for idcpu, reals in beam.soa_views():
            for i in range(11):
                cols[i].append(reals[i].copy())
            ids.append(np.array(idcpu, copy=True))
        data = {
            i: (np.concatenate(cols[i]) if cols[i] else np.zeros(0)) for i in range(11)
        }
        idcpu = np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint64)

        it = self._series.iterations[self._step]
        beam_sp = it.particles["beam"]

        for rec_name, comps in self._RECORDS.items():
            rec = beam_sp[rec_name]
            for comp_name, idx in comps:
                arr = np.asarray(data[idx], dtype=np.float64)
                rc = rec[comp_name]
                rc.reset_dataset(io.Dataset(arr.dtype, arr.shape))
                rc.store_chunk(arr)

        for sc_name, idx in self._SCALARS.items():
            arr = np.asarray(data[idx], dtype=np.float64)
            rc = beam_sp[sc_name][io.Record_Component.SCALAR]
            rc.reset_dataset(io.Dataset(arr.dtype, arr.shape))
            rc.store_chunk(arr)

        ids64 = np.asarray(idcpu, dtype=np.uint64)
        rc = beam_sp["id"][io.Record_Component.SCALAR]
        rc.reset_dataset(io.Dataset(ids64.dtype, ids64.shape))
        rc.store_chunk(ids64)

        # reference-particle attributes on the species (names match ImpactX)
        ref = beam.ref
        beam_sp.set_attribute("beta_ref", float(ref.beta()))
        beam_sp.set_attribute("gamma_ref", float(ref.gamma()))
        beam_sp.set_attribute("beta_gamma_ref", float(ref.beta_gamma()))
        for attr in ("s", "x", "y", "z", "t", "px", "py", "pz", "pt", "mass", "charge"):
            beam_sp.set_attribute(f"{attr}_ref", float(getattr(ref, attr)))

        self._series.flush()

    def finalize(self):
        if self._series is not None:
            self._series.flush()
            del self._series
            self._series = None


class Lattice(list):
    """The accelerator lattice: a plain list of elements.

    Subclasses ``list`` so ``extend``/``append``/iteration/``len`` all work as
    the FODO example expects (``sim.lattice.extend([...])``).
    """
