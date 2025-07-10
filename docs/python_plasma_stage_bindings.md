# PlasmaStage Python Bindings

This document describes the Python bindings for the PlasmaStage element in ImpactX.

## Overview

The PlasmaStage element simulates the interaction of a particle beam with a plasma, applying wakefield effects based on the specified model. It is implemented as a thick element that can be sliced for accurate simulation.

## Available Wakefield Models

The PlasmaStage element supports the following wakefield models:

- `"none"`: No wakefield effects (pass-through)
- `"simple_blowout"`: Simple blowout wakefield model
- `"custom_blowout"`: Custom blowout wakefield model
- `"focusing_blowout"`: Focusing blowout wakefield model
- `"cold_fluid_1d"`: 1D cold fluid model
- `"quasistatic_2d"`: 2D quasistatic model

## Constructor

```python
impactx.elements.PlasmaStage(
    wakefield_model,    # str: Type of wakefield model
    density,           # float: Plasma density in m^-3
    length,            # float: Plasma stage length in m
    ds=0,             # float: Segment length in m (optional)
    dx=0,             # float: Horizontal offset in m (optional)
    dy=0,             # float: Vertical offset in m (optional)
    rotation=0,       # float: Rotation angle in degrees (optional)
    name=None         # str: Optional element name (optional)
)
```

## Properties

The PlasmaStage element exposes the following properties:

### Read/Write Properties

- `wakefield_model` (str): Type of wakefield model
- `density` (float): Plasma density in m^-3
- `length` (float): Plasma stage length in m
- `slice_ds` (float): Slice length in m
- `betgam2` (float): Beta*gamma squared
- `slice_bg` (float): Slice beta*gamma

### Read-Only Properties

- `ds` (float): Segment length in m
- `name` (str): Element name
- `nslice` (int): Number of slices

## Methods

### `__repr__()`
Returns a string representation of the element.

### `to_dict()`
Returns a dictionary representation of the element with all its properties.

## Usage Examples

### Basic Usage

```python
import impactx

# Create a simple plasma stage
plasma_stage = impactx.elements.PlasmaStage(
    wakefield_model="simple_blowout",
    density=1e20,  # 1e20 m^-3
    length=0.1,    # 0.1 m
    name="my_plasma_stage"
)

# Access properties
print(f"Model: {plasma_stage.wakefield_model}")
print(f"Density: {plasma_stage.density:.2e} m^-3")
print(f"Length: {plasma_stage.length:.3f} m")

# Modify properties
plasma_stage.density = 2e20
plasma_stage.length = 0.2
```

### Creating a Lattice

```python
import impactx

# Create a lattice with plasma stage
lattice = impactx.elements.KnownElementsList()

# Add elements to the lattice
lattice.append(impactx.elements.Drift(ds=0.1, name="drift_before"))
lattice.append(impactx.elements.PlasmaStage(
    wakefield_model="simple_blowout",
    density=1e20,
    length=0.2,
    name="plasma_stage"
))
lattice.append(impactx.elements.Drift(ds=0.1, name="drift_after"))

# Print lattice information
for i, element in enumerate(lattice):
    print(f"{i+1}. {type(element).__name__}: {element.name}")
```

### Comparing Different Models

```python
import impactx

# Create plasma stages with different models
models = ["none", "simple_blowout", "focusing_blowout"]
plasma_stages = {}

for model in models:
    plasma_stages[model] = impactx.elements.PlasmaStage(
        wakefield_model=model,
        density=1e20,
        length=0.1,
        name=f"plasma_{model}"
    )

# Compare properties
for model, stage in plasma_stages.items():
    print(f"{model}: {stage}")
```

## Error Handling

The PlasmaStage constructor will raise an exception if an invalid wakefield model is provided:

```python
try:
    plasma_stage = impactx.elements.PlasmaStage(
        wakefield_model="invalid_model",
        density=1e20,
        length=0.1
    )
except Exception as e:
    print(f"Error: {e}")
```

## Integration with ImpactX

The PlasmaStage element integrates seamlessly with other ImpactX elements and can be used in:

- Particle tracking simulations
- Lattice design and optimization
- Wakefield effect studies
- Plasma acceleration research

## Physics Notes

- The plasma density should be specified in m^-3
- The length parameter determines the total plasma stage length
- The ds parameter controls the slice length for accurate simulation
- Different wakefield models implement different physics approximations
- The element automatically computes physics constants based on the beam parameters

## See Also

- [PlasmaStage C++ Documentation](../src/elements/PlasmaStage.H)
- [ImpactX Elements Overview](../src/elements/All.H)
- [Python Bindings Source](../src/python/elements.cpp)
