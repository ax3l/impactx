#!/usr/bin/env python3
#
# Copyright 2022 ImpactX contributors
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#

import numpy as np
import pandas as pd
from scipy.stats import moment


def get_moments(beam):
    """Calculate standard deviations of beam position & momenta
    and emittance values

    Returns
    -------
    sigx, sigy, sigt, emittance_x, emittance_y, emittance_t
    """
    sigx = moment(beam["x"], moment=2)**0.5  # variance -> std dev.
    sigpx = moment(beam["px"], moment=2)**0.5
    sigy = moment(beam["y"], moment=2)**0.5
    sigpy = moment(beam["py"], moment=2)**0.5
    sigt = moment(beam["t"], moment=2)**0.5
    sigpt = moment(beam["pt"], moment=2)**0.5

    epstrms = beam.cov(ddof=0)
    emittance_x = (sigx**2 * sigpx**2 - epstrms["x"]["px"]**2)**0.5
    emittance_y = (sigy**2 * sigpy**2 - epstrms["y"]["py"]**2)**0.5
    emittance_t = (sigt**2 * sigpt**2 - epstrms["t"]["pt"]**2)**0.5

    return (
        sigx, sigy, sigt,
        emittance_x, emittance_y, emittance_t)


# initial/final beam on rank zero
initial = pd.read_csv("diags/initial_beam.txt.0.0", delimiter=r"\s+")
final = pd.read_csv("diags/output_beam.txt.0.0", delimiter=r"\s+")

# compare number of particles
num_particles = 10000
assert num_particles == len(initial)
assert num_particles == len(final)

print("Initial Beam:")
sigx, sigy, sigt, \
    emittance_x, emittance_y, emittance_t = get_moments(initial)
print(f"  sigx={sigx:e} sigy={sigy:e} sigt={sigt:e}")
print(f"  emittance_x={emittance_x:e} emittance_y={emittance_y:e} emittance_t={emittance_t:e}")

atol = 1.0  # a big number
rtol = num_particles**-0.5  # from random sampling of a smooth distribution
print(f"  rtol={rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx, sigy, sigt,
     emittance_x, emittance_y, emittance_t],
    [1.006098e-03, 9.973850e-04, 1.001599e-03,
     9.958199e-07, 9.976684e-07, 9.896235e-07],
    rtol=rtol,
    atol=atol
)


print("")
print("Final Beam:")
sigx, sigy, sigt, \
    emittance_x, emittance_y, emittance_t = get_moments(final)
print(f"  sigx={sigx:e} sigy={sigy:e} sigt={sigt:e}")
print(f"  emittance_x={emittance_x:e} emittance_y={emittance_y:e} emittance_t={emittance_t:e}")

atol = 1.0  # a big number
rtol = num_particles**-0.5  # from random sampling of a smooth distribution
print(f"  rtol={rtol} (ignored: atol~={atol})")

assert np.allclose(
    [sigx, sigy, sigt,
     emittance_x, emittance_y, emittance_t],
    [9.893263e-04, 9.991726e-04, 9.915850e-04,
     9.958199e-07, 9.976684e-07, 9.896235e-07],
    rtol=rtol,
    atol=atol
)
