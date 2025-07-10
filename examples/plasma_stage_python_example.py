#!/usr/bin/env python3
"""
Example: Using PlasmaStage element in ImpactX simulation

This example demonstrates how to create and use a PlasmaStage element
with different wakefield models in an ImpactX simulation.
"""

from impactx import elements


def create_plasma_stage_example():
    """Create a simple lattice with a plasma stage element"""

    # Create the lattice
    lattice = elements.KnownElementsList()

    # Add a drift before the plasma stage
    drift = elements.Drift(ds=0.1, name="drift_before")
    lattice.append(drift)

    # Create plasma stage with simple blowout model
    plasma_stage = elements.PlasmaStage(
        wakefield_model="simple_blowout",
        density=1e20,  # 1e20 m^-3 plasma density
        length=0.2,  # 0.2 m plasma length
        ds=0.01,  # 0.01 m slice length
        name="plasma_stage",
    )
    lattice.append(plasma_stage)

    # Add a drift after the plasma stage
    drift_after = elements.Drift(ds=0.1, name="drift_after")
    lattice.append(drift_after)

    return lattice


def create_plasma_stage_comparison():
    """Create lattices with different plasma stage models for comparison"""

    models = ["none", "simple_blowout", "focusing_blowout"]
    lattices = {}

    for model in models:
        lattice = elements.KnownElementsList()

        # Add a drift before
        drift = elements.Drift(ds=0.05, name="drift_before")
        lattice.append(drift)

        # Create plasma stage with current model
        plasma_stage = elements.PlasmaStage(
            wakefield_model=model,
            density=1e20,
            length=0.1,
            ds=0.01,
            name=f"plasma_{model}",
        )
        lattice.append(plasma_stage)

        # Add a drift after
        drift_after = elements.Drift(ds=0.05, name="drift_after")
        lattice.append(drift_after)

        lattices[model] = lattice

    return lattices


def print_lattice_info(lattice, name="Lattice"):
    """Print information about a lattice"""
    print(f"\n{name}:")
    print(f"  Number of elements: {len(lattice)}")

    for i, element in enumerate(lattice):
        element_type = type(element).__name__
        element_name = element.name if hasattr(element, "name") else "unnamed"
        print(f"  {i + 1}. {element_type}: {element_name}")

        # Print specific info for PlasmaStage
        if isinstance(element, elements.PlasmaStage):
            print(f"     - Wakefield model: {element.wakefield_model}")
            print(f"     - Plasma density: {element.density:.2e} m^-3")
            print(f"     - Length: {element.length:.3f} m")


def demonstrate_plasma_stage_usage():
    """Demonstrate various ways to use PlasmaStage elements"""

    print("=== PlasmaStage Element Demonstration ===\n")

    # Example 1: Simple plasma stage
    print("1. Simple PlasmaStage with simple_blowout model:")
    lattice1 = create_plasma_stage_example()
    print_lattice_info(lattice1, "Simple Plasma Stage Lattice")

    # Example 2: Comparison of different models
    print("\n2. Comparison of different wakefield models:")
    lattices = create_plasma_stage_comparison()

    for model, lattice in lattices.items():
        print_lattice_info(lattice, f"Lattice with {model} model")

    # Example 3: Creating individual elements
    print("\n3. Creating individual PlasmaStage elements:")

    # None model (no wakefield effects)
    plasma_none = elements.PlasmaStage(
        wakefield_model="none", density=1e20, length=0.1, name="plasma_none"
    )
    print(f"  - None model: {plasma_none}")

    # Simple blowout model
    plasma_simple = elements.PlasmaStage(
        wakefield_model="simple_blowout", density=1e20, length=0.1, name="plasma_simple"
    )
    print(f"  - Simple blowout: {plasma_simple}")

    # Focusing blowout model
    plasma_focusing = elements.PlasmaStage(
        wakefield_model="focusing_blowout",
        density=1e20,
        length=0.1,
        name="plasma_focusing",
    )
    print(f"  - Focusing blowout: {plasma_focusing}")

    # Example 4: Property access and modification
    print("\n4. Property access and modification:")
    plasma = elements.PlasmaStage(
        wakefield_model="simple_blowout", density=1e20, length=0.1
    )

    print(f"  Initial density: {plasma.density:.2e} m^-3")
    plasma.density = 2e20
    print(f"  Modified density: {plasma.density:.2e} m^-3")

    print(f"  Initial length: {plasma.length:.3f} m")
    plasma.length = 0.2
    print(f"  Modified length: {plasma.length:.3f} m")

    # Example 5: Dictionary representation
    print("\n5. Dictionary representation:")
    plasma_dict = plasma.to_dict()
    for key, value in plasma_dict.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demonstrate_plasma_stage_usage()
