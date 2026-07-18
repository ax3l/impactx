# -*- coding: utf-8 -*-
#
# Copyright 2022-2026 The ImpactX Community
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
"""Tests for the tracking progress bar / status line.

The progress bar is emitted from C++ (``src/tracking/ProgressBar.*``). Because it
writes to the process' real ``stdout`` (via ``amrex::Print``), we exercise it by
running a tiny simulation as a subprocess and capturing its output. Running under a
pipe makes ``stdout`` non-interactive, which is exactly the batch/dashboard case.
"""

import os
import re
import subprocess
import sys

import pytest

try:
    import pty
except ImportError:  # pragma: no cover - Windows has no pty
    pty = None

requires_pty = pytest.mark.skipif(
    pty is None or not hasattr(os, "openpty"),
    reason="requires a POSIX pseudo-terminal",
)

# A tiny tracking run with a known step total:
#   sum(nslice) = 3 + 2 = 5, periods = 2  ->  total = 10 forward steps
DRIVER = r"""
import sys

from impactx import ImpactX, distribution, elements

progress = sys.argv[1] if len(sys.argv) > 1 else "auto"
ascii_flag = len(sys.argv) > 2 and sys.argv[2] == "1"
space_charge = sys.argv[3] if len(sys.argv) > 3 else "off"

sim = ImpactX()
sim.particle_shape = 2
sim.diagnostics = False
sim.slice_step_diagnostics = False

if space_charge != "off":
    # mesh-based space charge -> a per-slice MLMG Poisson solve that prints residuals
    sim.max_level = 0
    sim.n_cell = [16, 16, 24]
    sim.blocking_factor_x = [16]
    sim.blocking_factor_y = [16]
    sim.blocking_factor_z = [8]
    sim.space_charge = space_charge
    sim.dynamic_size = True
    sim.prob_relative = [3.0]
else:
    sim.space_charge = False

sim.init_grids()

# progress bar controls (ParmParse-backed, need AMReX initialized)
sim.progress = progress
sim.progress_ascii = ascii_flag

# a 2 GeV electron reference particle
ref = sim.beam.ref
ref.set_species("electron").set_kin_energy_MeV(2.0e3)

# a small waterbag beam
distr = distribution.Waterbag(
    lambdaX=1.0e-4,
    lambdaY=1.0e-4,
    lambdaT=1.0e-3,
    lambdaPx=1.0e-5,
    lambdaPy=1.0e-5,
    lambdaPt=1.0e-3,
    muxpx=0.0,
    muypy=0.0,
    mutpt=0.0,
)
sim.add_particles(1.0e-9, distr, 1000)

sim.lattice.extend(
    [
        elements.Drift(name="d1", ds=1.0, nslice=3),
        elements.Quad(name="q1", ds=1.0, k=1.0, nslice=2),
    ]
)
sim.periods = 2

sim.track_particles()
sim.finalize()
"""

TOTAL = (3 + 2) * 2  # sum(nslice) * periods  ->  total forward steps
TOTAL_S = (1.0 + 1.0) * 2  # sum(ds) * periods  ->  total path length s in meters


def _run(tmp_path, *args):
    script = tmp_path / "progress_driver.py"
    script.write_text(DRIVER)
    # capture raw bytes (not text=True): universal-newline mode would translate
    # the live bar's carriage returns "\r" into "\n" and hide them from the test.
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(tmp_path),
        capture_output=True,
        timeout=300,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, (
        f"driver failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    )
    return stdout


def _run_pty(tmp_path, *args):
    """Run the driver with stdout attached to a pseudo-terminal (isatty() is True).

    Returns the raw byte stream (including the live bar's carriage returns and ANSI
    escapes) decoded as text.
    """
    script = tmp_path / "progress_driver.py"
    script.write_text(DRIVER)

    controller, worker = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=str(tmp_path),
        stdout=worker,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(worker)  # only the child keeps the worker end open

    chunks = []
    while True:
        try:
            data = os.read(controller, 65536)
        except OSError:  # controller raises EIO at EOF on Linux
            break
        if not data:
            break
        chunks.append(data)
    os.close(controller)
    proc.wait(timeout=300)

    out = b"".join(chunks).decode("utf-8", errors="replace")
    assert proc.returncode == 0, f"driver failed (rc={proc.returncode}):\n{out}"
    return out


def test_progress_banner_non_interactive(tmp_path):
    """A piped run emits per-step banner lines carrying the exact total.

    This is the contract the Trame dashboard relies on
    (``dashboard/Run/executor.py`` parses ``step=<N> of <T>``).
    """
    out = _run(tmp_path, "auto")

    # no live carriage-return bar should leak into a pipe
    assert "\r" not in out

    steps = re.findall(r"\+\+\+\+ Starting step=(\d+) of (\d+)", out)
    assert steps, f"no banner lines found in:\n{out}"

    # every line reports the exact total = periods * sum(nslice)
    assert {int(total) for _, total in steps} == {TOTAL}

    # exactly one banner per step, counting 1..TOTAL in order
    assert [int(step) for step, _ in steps] == list(range(1, TOTAL + 1))

    # explicit token endpoints
    assert f"++++ Starting step=1 of {TOTAL}" in out
    assert f"++++ Starting step={TOTAL} of {TOTAL}" in out

    # the path length s is reported against the exact total (periods * sum(ds))
    assert re.search(rf"s=\d+\.\d+/{TOTAL_S:.2f} m", out), out


def test_progress_live_forced_ascii(tmp_path):
    """``progress=on`` forces the live bar even into a pipe; ASCII keeps it deterministic."""
    out = _run(tmp_path, "on", "1")

    # live frames are drawn with a carriage return using the ASCII renderer
    assert "\r" in out
    assert "Tracking [" in out

    # progress is shown in path length s against the exact total
    assert re.search(rf"s=\d+\.\d+/{TOTAL_S:.2f} m", out), out

    # the final frame reaches 100% at s = total length (bar fully filled)
    assert re.search(
        rf"Tracking \[#+\]  s={TOTAL_S:.2f}/{TOTAL_S:.2f} m  100%  time ", out
    ), out

    # live mode must not also emit the per-step banner
    assert "++++ Starting step=" not in out


@requires_pty
def test_progress_live_bar_coexists_with_solver(tmp_path):
    """On a real terminal, the live bar stays pinned while a per-slice solver prints.

    With mesh space charge the MLMG Poisson solve prints residuals every slice. The
    bar installs a filtering stream buffer (see src/tracking/ProgressBar.cpp) so that
    output erases the bar, scrolls above it, and the bar is redrawn on the new bottom
    line -- rather than being shredded or suppressed.
    """
    out = _run_pty(tmp_path, "auto", "0", "3D")

    # the live bar and the solver output are both present (coexistence)
    assert "Tracking " in out, out
    assert "MLMG:" in out, out

    # the bar is kept at the bottom via ANSI "erase entire line" (\x1b[2K)
    assert "\x1b[2K" in out

    # the bar is redrawn after solver output (repinned below the scrolled MLMG lines)
    assert out.rfind("Tracking ") > out.find("MLMG:"), out

    # and it still reaches 100% at s = total length
    assert re.search(r"100%  time ", out), out
