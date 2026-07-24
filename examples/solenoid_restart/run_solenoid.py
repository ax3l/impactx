#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Marco Garten, Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import openpmd_api as io

from impactx import ImpactX, elements, push

source_path = "../solenoid.py/diags/openPMD/m1.h5"

sim = ImpactX()

# set numerical parameters and IO control
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

# load the particle bunch and the reference particle (a 250 MeV proton beam)
# from the final beam monitor output of the solenoid example
beam = sim.beam
push(beam, elements.Source("openPMD", source_path))

# check that the reference particle was restored exactly from the file metadata
ref = beam.ref
series = io.Series(source_path, io.Access.read_only)
last_step = list(series.iterations)[-1]
beam_md = series.iterations[last_step].particles["beam"]
for attr, value in [
    ("s_ref", ref.s),
    ("x_ref", ref.x),
    ("y_ref", ref.y),
    ("z_ref", ref.z),
    ("t_ref", ref.t),
    ("px_ref", ref.px),
    ("py_ref", ref.py),
    ("pz_ref", ref.pz),
    ("pt_ref", ref.pt),
    ("mass_ref", ref.mass),
    ("charge_ref", ref.charge),
    ("gyromagnetic_anomaly_ref", ref.gyromagnetic_anomaly),
]:
    assert value == beam_md.get_attribute(attr), attr
series.close()

# this is a 250 MeV proton at the end of the first solenoid channel (one period)
assert abs(ref.kin_energy_MeV - 250.0) < 1e-11
assert abs(ref.charge_qe - 1.0) < 1e-14
assert abs(ref.s - 3.820395) < 1e-11

# add beam diagnostics
m1 = elements.BeamMonitor("m1", backend="h5")

# design the accelerator lattice
sol1 = elements.Sol(name="sol1", ds=3.820395, ks=0.8223219329893234)
sim.lattice.append(m1)
sim.lattice.extend([sol1] * 3)
sim.lattice.append(m1)

# run simulation
sim.track_particles()

# clean shutdown
sim.finalize()
