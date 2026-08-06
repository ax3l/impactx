#!/usr/bin/env python3
#
# Copyright 2022-2026 ImpactX contributors
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

"""
Tests for the ``min_model`` fidelity floor of the MAD-X -> ImpactX translation.

ImpactX implements most elements at up to three levels of fidelity: linear,
paraxial (``Chr*``) and exact (``Exact*``), see ``docs/source/theory/assumptions.rst``.
The MAD-X translator picks the cheapest ImpactX element that represents the
MAD-X input. ``min_model`` raises the *lower gate* of that choice without
otherwise changing it. Three properties are asserted here:

1. A floor never lowers a tier. An element that already needs a richer model,
   such as a skew quadrupole that only exists as ``ExactMultipole``, keeps it at
   ``min_model="linear"``.
2. Where a tier is not implemented, the floor rounds *up*. ImpactX has no
   ``ChrSbend``, so ``min_model="paraxial"`` on a plain SBEND yields the exact
   model rather than falling back to the linear one.
3. Where no model reaches the floor at all, the translation still succeeds and
   warns once. SOLENOID has only the ideal, hard-edge ``Sol``.

References:
    - https://impactx.readthedocs.io/en/latest/usage/python.html#impactx.elements.KnownElementsList.load_file
    - MAD-X manual: https://madx.web.cern.ch/webguide/manual.html
"""

import math
import warnings

import pytest

from impactx import elements
from impactx.element_models import MODEL_TIERS, select_model
from impactx.madx_to_impactx import (
    MADXImpactXTranslatorWarning,
    lattice,
    read_lattice,
)


def _translate(elems, **kwargs):
    """Translate parsed MAD-X element dicts through lattice(), muting warnings."""
    if isinstance(elems, dict):
        elems = [elems]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lattice(elems, **kwargs)


def _types(beamline):
    """Element class names of a translated beamline."""
    return [type(e).__name__ for e in beamline]


def _only(beamline, cls):
    """Return the single element of type ``cls`` in a translated beamline."""
    matches = [e for e in beamline if isinstance(e, cls)]
    assert len(matches) == 1, _types(beamline)
    return matches[0]


# ---------------------------------------------------------------------------
# The tier picker itself
# ---------------------------------------------------------------------------


def test_select_model_picks_cheapest_at_or_above_floor():
    """The cheapest implemented tier that satisfies the floor wins."""
    full = {tier: tier for tier in MODEL_TIERS}
    assert select_model(full, "linear") == ("linear", "linear")
    assert select_model(full, "paraxial") == ("paraxial", "paraxial")
    assert select_model(full, "exact") == ("exact", "exact")


def test_select_model_rounds_up_through_a_missing_tier():
    """A family without a paraxial model serves a paraxial floor from its exact one."""
    holed = {"linear": "linear", "exact": "exact"}
    assert select_model(holed, "linear") == ("linear", "linear")
    assert select_model(holed, "paraxial") == ("exact", "exact")
    assert select_model(holed, "exact") == ("exact", "exact")


def test_select_model_falls_back_below_an_unreachable_floor():
    """With nothing at or above the floor, the most faithful model available is used."""
    linear_only = {"linear": "linear"}
    assert select_model(linear_only, "exact") == ("linear", "linear")


# ---------------------------------------------------------------------------
# Element families with a full or partial tier ladder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "min_model, expected",
    [("linear", "Drift"), ("paraxial", "ChrDrift"), ("exact", "ExactDrift")],
)
def test_drift_tiers(min_model, expected):
    """DRIFT has all three tiers, so the floor is met exactly."""
    beamline = _translate(
        {"name": "d1", "type": "drift", "l": 0.5}, min_model=min_model
    )
    assert _types(beamline) == [expected]
    assert beamline[0].ds == pytest.approx(0.5)


@pytest.mark.parametrize(
    "min_model, expected",
    [("linear", "Quad"), ("paraxial", "ChrQuad"), ("exact", "ExactQuad")],
)
def test_quadrupole_tiers(min_model, expected):
    """QUADRUPOLE has all three tiers. Strength and rotation survive the promotion."""
    beamline = _translate(
        {"name": "q1", "type": "quadrupole", "l": 0.3, "k1": 2.0, "tilt": 0.25},
        min_model=min_model,
    )
    assert _types(beamline) == [expected]
    quad = beamline[0]
    assert quad.ds == pytest.approx(0.3)
    assert quad.k == pytest.approx(2.0)
    assert quad.rotation == pytest.approx(0.25 * 180.0 / math.pi)
    if min_model != "linear":
        # MAD-X k convention: k in m^(-2), which plain Quad always assumes
        assert quad.unit == 0


@pytest.mark.parametrize(
    "min_model, expected",
    [("linear", "Sbend"), ("paraxial", "ExactSbend"), ("exact", "ExactSbend")],
)
def test_sbend_tiers_round_up(min_model, expected):
    """There is no ChrSbend: a paraxial floor rounds up to the exact bend."""
    ds, angle = 2.0, 0.1
    beamline = _translate(
        {"name": "b1", "type": "sbend", "l": ds, "angle": angle},
        min_model=min_model,
    )
    assert _types(beamline) == [expected]
    bend = beamline[0]
    assert bend.ds == pytest.approx(ds)
    if expected == "Sbend":
        assert bend.to_dict()["rc"] == pytest.approx(ds / angle)
    else:
        # note: the ExactSbend constructor takes phi in degrees, but the
        # property returns radians (ImpactX issue #1367)
        assert bend.phi == pytest.approx(angle)


@pytest.mark.parametrize(
    "min_model, expected",
    [("linear", "CFbend"), ("paraxial", "ExactCFbend"), ("exact", "ExactCFbend")],
)
def test_combined_function_bend_tiers_round_up(min_model, expected):
    """A bend with K1 uses the combined-function rung of the ladder."""
    ds, angle, k1 = 2.0, 0.1, 0.5
    beamline = _translate(
        {"name": "b1", "type": "sbend", "l": ds, "angle": angle, "k1": k1},
        min_model=min_model,
    )
    assert _types(beamline) == [expected]
    fields = beamline[0].to_dict()
    if expected == "CFbend":
        assert fields["rc"] == pytest.approx(ds / angle)
        assert fields["k"] == pytest.approx(k1)
    else:
        assert fields["k_normal"][0] == pytest.approx(angle / ds)  # curvature 1/rc
        assert fields["k_normal"][1] == pytest.approx(k1)


@pytest.mark.parametrize(
    "min_model, expected",
    [("linear", "linear"), ("paraxial", "nonlinear"), ("exact", "nonlinear")],
)
def test_dipedge_fringe_model_follows_min_model(min_model, expected):
    """DipEdge switches its fringe model by argument. "nonlinear" is its exact tier."""
    beamline = _translate(
        {
            "name": "de",
            "type": "dipedge",
            "h": 0.0966,
            "e1": 0.0483,
            "hgap": 0.01,
            "fint": 0.5,
        },
        min_model=min_model,
    )
    edge = _only(beamline, elements.DipEdge)
    assert edge.model == expected
    # physical parameters are untouched by the model choice
    assert edge.psi == pytest.approx(0.0483)
    assert edge.rc == pytest.approx(1.0 / 0.0966)


def test_bend_edges_follow_min_model():
    """The DipEdges a bend emits at its faces honor the floor, too."""
    elem = {
        "name": "b1",
        "type": "sbend",
        "l": 2.0,
        "angle": 0.1,
        "e1": 0.05,
        "e2": 0.05,
        "hgap": 0.01,
        "fint": 0.5,
    }
    beamline = _translate(dict(elem), min_model="linear")
    assert _types(beamline) == ["DipEdge", "Sbend", "DipEdge"]
    assert [e.model for e in beamline if isinstance(e, elements.DipEdge)] == [
        "linear",
        "linear",
    ]

    beamline = _translate(dict(elem), min_model="exact")
    assert _types(beamline) == ["DipEdge", "ExactSbend", "DipEdge"]
    assert [e.model for e in beamline if isinstance(e, elements.DipEdge)] == [
        "nonlinear",
        "nonlinear",
    ]


def test_synthetic_drifts_follow_min_model():
    """Drifts the translator adds itself (here: around a thin MONITOR) follow the floor."""
    beamline = _translate(
        {"name": "m1", "type": "monitor", "l": 0.4}, min_model="exact"
    )
    assert _types(beamline) == ["ExactDrift", "BeamMonitor"]


# ---------------------------------------------------------------------------
# A floor never lowers a tier
# ---------------------------------------------------------------------------


def test_skew_quadrupole_stays_exact_at_linear_floor():
    """A skew quadrupole only exists as ExactMultipole. The floor must not downgrade it."""
    beamline = _translate(
        {"name": "q1", "type": "quadrupole", "l": 0.3, "k1": 2.0, "k1s": 0.4},
        min_model="linear",
    )
    assert _types(beamline) == ["ExactMultipole"]


def test_sextupole_stays_exact_at_linear_floor():
    """ImpactX has no thick linear sextupole. SEXTUPOLE stays exact at any floor."""
    beamline = _translate(
        {"name": "s1", "type": "sextupole", "l": 0.2, "k2": 3.0}, min_model="linear"
    )
    assert _types(beamline) == ["ExactMultipole"]


# ---------------------------------------------------------------------------
# Unreachable floors warn, once
# ---------------------------------------------------------------------------


def test_solenoid_warns_once_when_floor_is_unreachable():
    """SOLENOID has only the linear Sol model: translate it anyway and warn once."""
    solenoids = [
        {"name": f"s{i}", "type": "solenoid", "l": 0.5, "ks": 1.0} for i in range(3)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beamline = lattice(solenoids, min_model="exact")

    assert _types(beamline) == ["Sol", "Sol", "Sol"]
    floor_warnings = [
        w
        for w in caught
        if issubclass(w.category, MADXImpactXTranslatorWarning)
        and "SOLENOID" in str(w.message)
        and "exact" in str(w.message)
    ]
    assert len(floor_warnings) == 1, [str(w.message) for w in caught]


def test_solenoid_does_not_warn_at_linear_floor():
    """The default floor is reachable everywhere, so it never triggers the warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beamline = lattice([{"name": "s1", "type": "solenoid", "l": 0.5, "ks": 1.0}])

    assert _types(beamline) == ["Sol"]
    assert not [w for w in caught if "no 'linear'" in str(w.message)]


# ---------------------------------------------------------------------------
# Validation and end-to-end plumbing
# ---------------------------------------------------------------------------


def test_invalid_min_model_raises():
    with pytest.raises(ValueError, match="min_model"):
        lattice([{"name": "d1", "type": "drift", "l": 0.5}], min_model="quadratic")


def test_read_lattice_forwards_min_model(tmp_path):
    """min_model reaches the translator through the file-reading entry point."""
    deck = tmp_path / "fodo.madx"
    deck.write_text(
        "BEAM, PARTICLE=ELECTRON, ENERGY=5.0;\n"
        "D1: DRIFT, L=0.25;\n"
        "Q1: QUADRUPOLE, L=0.5, K1=1.0;\n"
        "MYLINE: LINE=(D1, Q1, D1);\n"
        "USE, SEQUENCE=MYLINE;\n"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = read_lattice(str(deck))
        exact = read_lattice(str(deck), min_model="exact")

    assert _types(default) == ["Drift", "Quad", "Drift"]
    assert _types(exact) == ["ExactDrift", "ExactQuad", "ExactDrift"]


def test_load_file_forwards_min_model(tmp_path):
    """The public KnownElementsList.load_file entry point forwards min_model."""
    deck = tmp_path / "fodo.madx"
    deck.write_text(
        "BEAM, PARTICLE=ELECTRON, ENERGY=5.0;\n"
        "D1: DRIFT, L=0.25;\n"
        "Q1: QUADRUPOLE, L=0.5, K1=1.0;\n"
        "MYLINE: LINE=(D1, Q1, D1);\n"
        "USE, SEQUENCE=MYLINE;\n"
    )

    lat = elements.KnownElementsList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat.load_file(str(deck), nslice=2, min_model="paraxial")

    assert _types(lat) == ["ChrDrift", "ChrQuad", "ChrDrift"]
    assert lat[1].nslice == 2


def test_load_file_rejects_invalid_min_model(tmp_path):
    deck = tmp_path / "fodo.madx"
    deck.write_text(
        "BEAM, PARTICLE=ELECTRON, ENERGY=5.0;\n"
        "D1: DRIFT, L=0.25;\n"
        "MYLINE: LINE=(D1);\n"
        "USE, SEQUENCE=MYLINE;\n"
    )

    lat = elements.KnownElementsList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="min_model"):
            lat.load_file(str(deck), min_model="chromatic")


# ---------------------------------------------------------------------------
# PALS reader (same floor, same vocabulary)
# ---------------------------------------------------------------------------


def test_pals_min_model():
    """The PALS reader shares the tier tables. Its default output is unchanged."""
    pals = pytest.importorskip("pals")

    beamline = pals.BeamLine(
        name="fodo",
        line=[
            pals.Drift(name="d1", length=0.25),
            pals.Quadrupole(
                name="q1",
                length=0.5,
                MagneticMultipoleP=pals.MagneticMultipoleParameters(Kn1=1.0),
            ),
        ],
    )

    from impactx.pals_to_impactx import read_lattice as pals_read_lattice

    # the linear `Quad` has no `unit` argument yet (ImpactX issue #798), so the
    # cheapest quadrupole model available to PALS is the paraxial one
    assert _types(pals_read_lattice(beamline)) == ["Drift", "ChrQuad"]
    assert _types(pals_read_lattice(beamline, min_model="exact")) == [
        "ExactDrift",
        "ExactQuad",
    ]
