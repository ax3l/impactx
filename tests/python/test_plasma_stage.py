#!/usr/bin/env python3
#
# Copyright 2022-2023 The ImpactX Community
#
# Authors: Axel Huebl
# License: BSD-3-Clause-LBNL
#
# -*- coding: utf-8 -*-

import pytest

from impactx import elements


def test_plasma_stage_creation():
    """Test creating PlasmaStage elements with different wakefield models"""

    # Test all wakefield models using the enum
    models = [
        elements.PlasmaStage.WakefieldModel.none,
        elements.PlasmaStage.WakefieldModel.simple_blowout,
        elements.PlasmaStage.WakefieldModel.custom_blowout,
        elements.PlasmaStage.WakefieldModel.focusing_blowout,
        elements.PlasmaStage.WakefieldModel.cold_fluid_1d,
        elements.PlasmaStage.WakefieldModel.quasistatic_2d,
    ]

    for model in models:
        print(f"Testing PlasmaStage with wakefield_model: {model}")
        # Use positional arguments for the first three
        plasma_stage = elements.PlasmaStage(
            0.1,  # length
            1e20,  # density
            model,  # wakefield_model (enum)
            0,
            0,
            0,
            0,
            0,
            1,
            f"plasma_{model.name}",
        )
        assert plasma_stage.m_wakefield_model == model


def test_plasma_stage_properties():
    """Test property setters and getters"""
    plasma_stage = elements.PlasmaStage(
        0.1, 1e20, elements.PlasmaStage.WakefieldModel.simple_blowout
    )
    plasma_stage.nr = 32
    plasma_stage.nxi = 64
    plasma_stage.dr = 2e-5
    plasma_stage.dxi = 2e-5
    assert plasma_stage.nr == 32
    assert plasma_stage.nxi == 64
    assert plasma_stage.dr == 2e-5
    assert plasma_stage.dxi == 2e-5


def test_plasma_stage_invalid_model():
    """Test that invalid wakefield model raises an error"""
    with pytest.raises(Exception):
        elements.PlasmaStage(0.1, 1e20, 999)
    print("\u2713 Correctly caught error for invalid model")


if __name__ == "__main__":
    print("Testing PlasmaStage Python bindings...")

    test_plasma_stage_creation()
    test_plasma_stage_properties()
    test_plasma_stage_invalid_model()

    print("\n✓ All tests passed!")
