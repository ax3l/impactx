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

    # Test all wakefield models
    models = [
        "none",
        "simple_blowout",
        "custom_blowout",
        "focusing_blowout",
        "cold_fluid_1d",
        "quasistatic_2d",
    ]

    for model in models:
        print(f"Testing PlasmaStage with wakefield_model: {model}")

        # Create PlasmaStage element
        plasma_stage = elements.PlasmaStage(
            wakefield_model=model,
            density=1e20,  # 1e20 m^-3
            length=0.1,  # 0.1 m
            ds=0.01,  # 0.01 m slice length
            name=f"plasma_{model}",
        )

        # Test properties
        assert plasma_stage.wakefield_model == model
        assert plasma_stage.density == 1e20
        assert plasma_stage.length == 0.1
        assert plasma_stage.ds == 0.01
        assert plasma_stage.name == f"plasma_{model}"

        # Test string representation
        repr_str = repr(plasma_stage)
        print(f"  Repr: {repr_str}")
        assert "PlasmaStage" in repr_str
        assert model in repr_str

        # Test to_dict method
        dict_data = plasma_stage.to_dict()
        print(f"  Dict: {dict_data}")
        assert dict_data["wakefield_model"] == model
        assert dict_data["density"] == 1e20
        assert dict_data["length"] == 0.1

        print(f"  ✓ {model} model works correctly")


def test_plasma_stage_properties():
    """Test property setters and getters"""

    plasma_stage = elements.PlasmaStage(
        wakefield_model="simple_blowout", density=1e20, length=0.1
    )

    # Test property setters
    plasma_stage.density = 2e20
    assert plasma_stage.density == 2e20

    plasma_stage.length = 0.2
    assert plasma_stage.length == 0.2

    plasma_stage.slice_ds = 0.02
    assert plasma_stage.slice_ds == 0.02

    plasma_stage.betgam2 = 100.0
    assert plasma_stage.betgam2 == 100.0

    plasma_stage.slice_bg = 10.0
    assert plasma_stage.slice_bg == 10.0

    print("✓ Property setters and getters work correctly")


def test_plasma_stage_invalid_model():
    """Test that invalid wakefield model raises an error"""

    with pytest.raises(Exception):
        # Pass an invalid enum value (e.g., 999)
        elements.PlasmaStage(0.1, 1e20, 999)
    print("\u2713 Correctly caught error for invalid model")


if __name__ == "__main__":
    print("Testing PlasmaStage Python bindings...")

    test_plasma_stage_creation()
    test_plasma_stage_properties()
    test_plasma_stage_invalid_model()

    print("\n✓ All tests passed!")
