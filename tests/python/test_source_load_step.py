#!/usr/bin/env python3
#
# Copyright 2022-2026 The ImpactX Community
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

"""
Test that the Source element can load a selected step (openPMD iteration).
"""

from pathlib import Path

import pytest

from impactx import Config, ImpactX, distribution, elements, push

io = pytest.importorskip("openpmd_api")

if not Config.have_openpmd:
    pytest.skip("ImpactX was compiled without openPMD support", allow_module_level=True)

# the drift between the two beam monitors
DRIFT_DS = 0.25

# HDF5 as in the solenoid_restart example, else whatever this build provides
BACKEND = "h5" if Config.openpmd_backends.get("hdf5", False) else "default"

# exact for a single drift, but single precision accumulates a few epsilon
RTOL = 1.0e-12 if Config.precision == "DOUBLE" else 1.0e-5


def write_series(name, npart):
    """Track a beam through two monitors and return the written series' path.

    The lattice writes two steps: one at s=0 and one at s=DRIFT_DS.
    """
    sim = ImpactX()

    sim.particle_shape = 2
    sim.space_charge = False
    sim.slice_step_diagnostics = False
    sim.init_grids()

    ref = sim.beam.ref
    ref.set_species("proton").set_kin_energy_MeV(250.0)

    distr = distribution.Waterbag(
        lambdaX=1.559531175539e-3,
        lambdaY=2.205510139392e-3,
        lambdaT=1.0e-3,
        lambdaPx=6.41218345413e-4,
        lambdaPy=9.06819680526e-4,
        lambdaPt=1.0e-3,
    )
    sim.add_particles(1.0e-9, distr, npart)

    monitor = elements.BeamMonitor(name, backend=BACKEND)
    sim.lattice.extend([monitor, elements.Drift(name="d1", ds=DRIFT_DS), monitor])

    try:
        sim.track_particles()
    finally:
        # this closes the openPMD series, so that we can read it back below
        sim.finalize()

    files = sorted(Path("diags/openPMD").glob(f"{name}.*"))
    assert files, f"no openPMD series found for monitor '{name}'"

    return str(files[0])


def stored_steps(series_path):
    """The steps (openPMD iterations) stored in a series."""
    series = io.Series(series_path, io.Access.read_only)
    steps = sorted(series.iterations)
    series.close()

    return steps


@pytest.mark.manages_amrex
def test_source_load_step():
    """
    This tests that the Source element loads the step (openPMD iteration)
    selected with load_step, absolute as well as counted back from the last.
    """
    npart = 512
    series_path = write_series("mon_load_step", npart)

    steps = stored_steps(series_path)
    assert len(steps) == 2

    # read back into a fresh simulation
    sim = ImpactX()
    sim.particle_shape = 2
    sim.space_charge = False
    # keep init_grids from moving the diags directory we just wrote out of the way
    sim.diagnostics = False
    sim.init_grids()
    beam = sim.beam

    try:
        # the reference particle is restored per push, so we can read
        # several steps into the same particle container
        for load_step, s_expected in [
            (steps[0], 0.0),  # first step: absolute
            (steps[-1], DRIFT_DS),  # last step: absolute
            (-1, DRIFT_DS),  # last step: counted back
            (-2, 0.0),  # first step: counted back
        ]:
            source = elements.Source(
                "openPMD", series_path, load_step=load_step, name="source"
            )
            push(beam, source)
            assert beam.ref.s == pytest.approx(s_expected, rel=RTOL, abs=1.0e-12)
            assert beam.ref.kin_energy_MeV == pytest.approx(250.0, rel=RTOL)

        # the default is the last step in the file
        push(beam, elements.Source("openPMD", series_path))
        assert beam.ref.s == pytest.approx(DRIFT_DS, rel=RTOL)

        # a step that is not in the file lists the available steps
        missing = steps[-1] + 1
        with pytest.raises(RuntimeError, match="available steps"):
            push(beam, elements.Source("openPMD", series_path, load_step=missing))

        # counting back further than the first step in the file
        with pytest.raises(RuntimeError, match="available steps"):
            push(beam, elements.Source("openPMD", series_path, load_step=-3))
    finally:
        sim.finalize()


def test_source_load_step_serialization():
    """
    This tests that load_step is part of the element's repr and to_dict().
    """
    source = elements.Source("openPMD", "beam.h5", load_step=3)
    assert source.load_step == 3
    assert "load_step=3" in repr(source)

    d = source.to_dict()
    assert d["load_step"] == 3

    # default: the last step in the file
    assert elements.Source("openPMD", "beam.h5").load_step == -1

    source.load_step = -2
    assert source.load_step == -2
