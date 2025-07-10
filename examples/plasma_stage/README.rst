Plasma Stage Example
===================

This example demonstrates the new PlasmaStage element in ImpactX, which provides
plasma wakefield acceleration capabilities similar to Wake-T.

Overview
--------

The PlasmaStage element implements plasma wakefield effects for particle tracking
in ImpactX. It supports multiple wakefield models based on Wake-T:

- **none**: No wakefield effects (just drift)
- **simple_blowout**: Simple blowout model with focusing and acceleration
- **custom_blowout**: Custom blowout model with user-defined fields
- **focusing_blowout**: Focusing blowout model with transverse focusing only
- **cold_fluid_1d**: 1D cold fluid model for longitudinal waves
- **quasistatic_2d**: 2D quasistatic model with full 2D wakefield

Physics Models
-------------

1. **Simple Blowout Model** (`simple_blowout`)
   - Transverse focusing: F_x = -k_β² * x (betatron oscillations)
   - Longitudinal acceleration: E_z = constant field
   - Suitable for laser-driven plasma wakefield acceleration
   - Based on the blowout regime where plasma electrons are expelled

2. **Custom Blowout Model** (`custom_blowout`)
   - Same transverse focusing as simple blowout
   - Longitudinal acceleration with user-defined slope: E_z = E_z0 + E_z' * t
   - Allows for custom field profiles
   - Useful for beam-driven or complex laser-driven scenarios

3. **Focusing Blowout Model** (`focusing_blowout`)
   - Transverse focusing only: F_x = -k_β² * x
   - No longitudinal acceleration
   - Useful for plasma lenses or focusing-only applications

4. **Cold Fluid 1D Model** (`cold_fluid_1d`)
   - Weaker transverse focusing (50% of blowout strength)
   - Nonlinear longitudinal acceleration with oscillations
   - Based on 1D cold fluid theory
   - Suitable for broad drivers where radial waves are negligible

5. **Quasistatic 2D Model** (`quasistatic_2d`)
   - Radial-dependent transverse focusing: k_β(r) = k_β * (1 + 0.1 * r * k_p)
   - Radial-dependent longitudinal acceleration: E_z(r) = E_z * (1 - 0.05 * r * k_p)
   - Full 2D wakefield effects
   - Most sophisticated model, suitable for narrow beams

Physics Constants
----------------

All models compute the following plasma parameters:

- **Plasma frequency**: ω_p = √(n_e e² / (m_e ε₀))
- **Plasma wavenumber**: k_p = ω_p / c
- **Betatron wavenumber**: k_β = ω_p / (2γ)
- **Accelerating field**: E_z = ω_p² m_e / (2e)

Parameters
----------

- **ds**: Segment length in meters
- **density**: Plasma density in m⁻³
- **wakefield_model**: Type of wakefield model to use
- **dx, dy**: Horizontal and vertical translation errors in meters
- **rotation_degree**: Rotation error in the transverse plane [degrees]
- **aperture_x, aperture_y**: Horizontal and vertical half-apertures in meters
- **nslice**: Number of slices for space charge calculation

Usage
-----

1. Run the simulation:
   ```bash
   impactx input_plasma_stage.in
   ```

2. Analyze the results:
   ```bash
   python analysis_plasma_stage.py
   ```

Expected Results
---------------

The example demonstrates all five wakefield models in sequence:

- **plasma_simple**: Strong focusing + acceleration
- **plasma_custom**: Same focusing + custom acceleration
- **plasma_focusing**: Strong focusing only
- **plasma_cold_fluid**: Weak focusing + nonlinear acceleration
- **plasma_quasistatic**: Radial-dependent focusing + acceleration

Each model will show different effects on the beam:
- Transverse focusing oscillations
- Energy gain (except focusing_blowout)
- Emittance evolution
- Beam size variations

Comparison with Wake-T
---------------------

This implementation provides a simplified but functional version of Wake-T's plasma stage capabilities, adapted for ImpactX's particle tracking framework. The physics models are based on the same theoretical foundations but implemented within ImpactX's architecture.

Key differences from Wake-T:
- Simplified field calculations (analytical vs. numerical)
- No laser evolution (fixed field profiles)
- No adaptive time stepping
- Focused on particle tracking rather than full PIC simulation

Future Enhancements
------------------

- Implementation of laser drivers
- Adaptive time stepping
- More sophisticated field calculations
- Integration with external field sources
- Support for ion motion effects
- Enhanced diagnostics and field output

References
----------

- Wake-T documentation: https://wake-t.readthedocs.io/
- Plasma wakefield acceleration theory
- ImpactX documentation: https://impactx.readthedocs.io/
- P. Baxevanis and G. Stupakov, "Novel fast simulation technique for axisymmetric plasma wakefield acceleration configurations in the blowout regime," Phys. Rev. Accel. Beams 21, 071301 (2018)
