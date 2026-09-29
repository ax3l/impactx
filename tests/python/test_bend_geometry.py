#!/usr/bin/env python3
#
# Copyright 2022-2026 The ImpactX Community
#
# Authors: Axel Huebl, Chad Mitchell
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

"""The geometry of a bend: ds plus one of rc, phi or B, or phi together with B."""

import math
import warnings

import numpy as np
import pytest

from impactx import ImpactX, RefPart, elements

BENDS = {
    "Sbend": lambda **geometry: elements.Sbend(**geometry),
    "ExactSbend": lambda **geometry: elements.ExactSbend(**geometry),
    "CFbend": lambda **geometry: elements.CFbend(k=0.3, **geometry),
}


def electron(kin_energy_MeV=1.0e3):
    ref = RefPart()
    ref.set_species("electron").set_kin_energy_MeV(kin_energy_MeV)
    return ref


def geometries(ds, rc, ref):
    """The same bend of length ds and radius rc, specified each way"""
    phi = math.degrees(ds / rc)
    B = ref.rigidity_Tm / rc
    return {
        "rc": dict(ds=ds, rc=rc),
        "phi": dict(ds=ds, phi=phi),
        "B": dict(ds=ds, B=B),
        # a fixed angle: ds does not set the arc length, so pick a different one
        "phi_B": dict(ds=2.0 * ds, phi=phi, B=B),
    }


def track_reference(element, kin_energy_MeV=1.0e3):
    """The reference particle after the element"""
    sim = ImpactX()
    sim.particle_shape = 2
    sim.slice_step_diagnostics = False
    sim.diagnostics = False
    sim.init_grids()
    ref = sim.beam.ref
    ref.set_species("electron").set_kin_energy_MeV(kin_energy_MeV)
    sim.lattice.append(element)
    sim.track_reference(ref)
    result = dict(x=ref.x, z=ref.z, px=ref.px, pz=ref.pz, t=ref.t)
    sim.finalize()
    return result


@pytest.mark.parametrize("kind", BENDS.keys())
@pytest.mark.parametrize("rc", [10.0, -10.0])
def test_every_specification_gives_the_same_bend(kind, rc):
    ref = electron()
    maps = {}
    orbits = {}
    for name, geometry in geometries(0.5, rc, ref).items():
        bend = BENDS[kind](**geometry)
        assert bend.signed_rc(ref) == pytest.approx(rc, rel=1e-12), name
        maps[name] = np.array(bend.transfer_map(ref))
        orbits[name] = track_reference(bend)

    for name in ["phi", "B", "phi_B"]:
        np.testing.assert_allclose(maps[name], maps["rc"], rtol=1e-10, atol=1e-14)
        # phi_B has its own ds: s differs, but not the orbit
        for key, value in orbits["rc"].items():
            assert orbits[name][key] == pytest.approx(value, rel=1e-10, abs=1e-14), (
                name,
                key,
            )


def test_a_fixed_angle_is_kept_at_any_energy():
    """A cyclotron half turn: the radius grows with the energy, the angle stays 180 degrees"""
    radii = []
    for kin_energy_MeV in [1.0, 10.0]:
        half_turn = elements.ExactSbend(ds=0.25, phi=180.0, B=1.0)
        orbit = track_reference(half_turn, kin_energy_MeV)
        pz0 = electron(kin_energy_MeV).beta_gamma
        assert orbit["pz"] == pytest.approx(-pz0, rel=1e-12)
        assert orbit["px"] == pytest.approx(0.0, abs=1e-12 * pz0)
        radii.append(abs(half_turn.signed_rc(electron(kin_energy_MeV))))
    assert radii[1] > radii[0]


def test_a_field_bends_by_the_charge():
    electron_ref = electron()
    positron_ref = RefPart()
    positron_ref.set_species("positron").set_kin_energy_MeV(1.0e3)

    by_field = elements.Sbend(ds=0.5, B=0.5)
    assert by_field.signed_rc(positron_ref) == pytest.approx(
        -by_field.signed_rc(electron_ref), rel=1e-12
    )

    by_radius = elements.Sbend(ds=0.5, rc=10.0)
    assert by_radius.signed_rc(positron_ref) == by_radius.signed_rc(electron_ref)


@pytest.mark.parametrize("kind", BENDS.keys())
@pytest.mark.parametrize(
    "geometry",
    [
        dict(),
        dict(rc=10.0, phi=2.0),
        dict(rc=10.0, B=0.5),
        dict(rc=10.0, phi=2.0, B=0.5),
    ],
    ids=["none", "rc_phi", "rc_B", "rc_phi_B"],
)
def test_one_specification_is_required(kind, geometry):
    with pytest.raises(ValueError, match="rc"):
        BENDS[kind](ds=0.5, **geometry)


@pytest.mark.parametrize("kind", BENDS.keys())
def test_legacy_zero_field_with_an_angle_is_the_angle_alone(kind):
    with pytest.warns(DeprecationWarning, match="B=0.0 together with phi"):
        legacy = BENDS[kind](ds=0.5, phi=3.0, B=0.0)
    assert legacy.B is None
    assert legacy.to_dict(**_in_degrees(kind)) == BENDS[kind](ds=0.5, phi=3.0).to_dict(
        **_in_degrees(kind)
    )


def _in_degrees(kind):
    """ExactSbend.to_dict() gives phi in radians unless asked for degrees"""
    return dict(in_degrees=True) if kind == "ExactSbend" else {}


@pytest.mark.parametrize("kind", ["Sbend", "CFbend"])
def test_parameters_that_do_not_specify_the_bend_are_none(kind):
    bend = BENDS[kind](ds=0.5, phi=3.0, B=0.5)
    assert (bend.rc, bend.phi, bend.B) == (None, 3.0, 0.5)
    d = bend.to_dict()
    assert (d["rc"], d["phi"], d["B"]) == (None, 3.0, 0.5)


@pytest.mark.parametrize("kind", BENDS.keys())
def test_a_property_changes_a_parameter_of_the_specification(kind):
    bend = BENDS[kind](ds=0.5, rc=10.0)
    bend.rc = 12.0
    assert bend.signed_rc(electron()) == 12.0

    with pytest.raises(ValueError, match="set_geometry"):
        bend.B = 0.5
    with pytest.raises(ValueError, match="set_geometry"):
        bend.rc = None
    assert (bend.rc, bend.phi, bend.B) == (12.0, None, None)


@pytest.mark.parametrize("kind", BENDS.keys())
def test_set_geometry_changes_the_specification(kind):
    ref = electron()
    bend = BENDS[kind](ds=0.5, rc=10.0)

    bend.set_geometry(rc=None, B=0.5)
    assert (bend.rc, bend.B) == (None, 0.5)
    assert bend.signed_rc(ref) == pytest.approx(ref.rigidity_Tm / 0.5, rel=1e-12)

    # the remaining parameters are kept: adding phi gives a fixed angle
    bend.set_geometry(phi=90.0)
    assert bend.B == 0.5
    assert bend.to_dict(**_in_degrees(kind))["phi"] == pytest.approx(90.0)

    # a rejected change leaves the bend as it was
    with pytest.raises(ValueError):
        bend.set_geometry(rc=10.0)
    assert (bend.rc, bend.B) == (None, 0.5)

    with pytest.raises(TypeError, match="unexpected keyword"):
        bend.set_geometry(angle=10.0)


@pytest.mark.parametrize("kind", BENDS.keys())
def test_copy_changes_the_specification_together(kind):
    bend = BENDS[kind](ds=0.5, rc=10.0, name="b")
    by_angle = bend.copy(rc=None, phi=3.0)
    assert by_angle.to_dict(**_in_degrees(kind))["phi"] == pytest.approx(3.0)
    assert by_angle.rc is None
    assert bend.rc == 10.0  # the original is unchanged

    with pytest.raises(ValueError):
        bend.copy(phi=3.0)  # rc and phi


@pytest.mark.parametrize("kind", BENDS.keys())
@pytest.mark.parametrize("spec", ["rc", "phi", "B", "phi_B"])
def test_dicts_round_trip(kind, spec):
    geometry = geometries(0.5, 10.0, electron())[spec]
    lattice = elements.KnownElementsList()
    lattice.append(BENDS[kind](name="b", **geometry))

    restored = elements.KnownElementsList()
    restored.from_dicts(lattice.to_dicts())
    assert restored[0].to_dict(**_in_degrees(kind)) == pytest.approx(
        lattice[0].to_dict(**_in_degrees(kind))
    )


@pytest.mark.parametrize("kind", BENDS.keys())
@pytest.mark.parametrize(
    "geometry", [dict(rc=0.0), dict(phi=0.0), dict(B=0.0)], ids=["rc", "phi", "B"]
)
def test_a_straight_bend_is_a_drift_of_its_orbit(kind, geometry):
    ref = electron()
    straight = BENDS[kind](ds=0.5, **geometry)
    assert math.isinf(straight.signed_rc(ref))

    orbit = track_reference(straight)
    drift = track_reference(elements.Drift(ds=0.5))
    for key, value in drift.items():
        assert orbit[key] == pytest.approx(value, rel=1e-12, abs=1e-14), key


def test_a_straight_cfbend_is_a_quadrupole():
    ref = electron()
    np.testing.assert_allclose(
        np.array(elements.CFbend(ds=0.5, rc=0.0, k=0.3).transfer_map(ref)),
        np.array(elements.Quad(ds=0.5, k=0.3).transfer_map(ref)),
        rtol=1e-12,
        atol=1e-15,
    )


def test_input_file_geometry(tmp_path):
    """Input files take the same rc, phi and B keys"""
    inputs = tmp_path / "inputs"
    inputs.write_text(
        "lattice.elements = geo_sb_rc geo_sb_phi geo_esb_b geo_esb_phib geo_cf_phi\n"
        "geo_sb_rc.type = sbend\ngeo_sb_rc.ds = 0.5\ngeo_sb_rc.rc = 10.0\n"
        "geo_sb_phi.type = sbend\ngeo_sb_phi.ds = 0.5\ngeo_sb_phi.phi = 3.0\n"
        "geo_esb_b.type = sbend_exact\ngeo_esb_b.ds = 0.5\ngeo_esb_b.B = 0.5\n"
        "geo_esb_phib.type = sbend_exact\ngeo_esb_phib.ds = 0.25\n"
        "geo_esb_phib.phi = 180.0\ngeo_esb_phib.B = 1.0\n"
        "geo_cf_phi.type = cfbend\ngeo_cf_phi.ds = 0.5\ngeo_cf_phi.phi = 3.0\n"
        "geo_cf_phi.k = 0.3\n"
    )

    sim = ImpactX()
    sim.load_inputs_file(str(inputs))
    sim.particle_shape = 2
    sim.slice_step_diagnostics = False
    sim.diagnostics = False
    sim.init_grids()
    sim.init_lattice_elements_from_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # ExactSbend.to_dict()
        specs = [(el.rc, el.to_dict().get("B"), el.name) for el in sim.lattice]
    assert specs == [
        (10.0, None, "geo_sb_rc"),
        (None, None, "geo_sb_phi"),
        (None, 0.5, "geo_esb_b"),
        (None, 1.0, "geo_esb_phib"),
        (None, None, "geo_cf_phi"),
    ]
    assert sim.lattice[1].phi == pytest.approx(3.0)
    assert sim.lattice[4].phi == pytest.approx(3.0)

    sim.finalize()
