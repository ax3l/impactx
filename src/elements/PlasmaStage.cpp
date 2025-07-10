/* Copyright 2022-2023 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */

#include "PlasmaStage.H"

#include <stdexcept>
#include <iostream>
#include <vector>


namespace impactx::elements
{
    std::string
    PlasmaStage::wakefield_model_name (WakefieldModel const & model)
    {
        switch (model)
        {
            case WakefieldModel::none:
                return "none";
            case WakefieldModel::simple_blowout:
                return "simple_blowout";
            case WakefieldModel::custom_blowout:
                return "custom_blowout";
            case WakefieldModel::focusing_blowout:
                return "focusing_blowout";
            case WakefieldModel::cold_fluid_1d:
                return "cold_fluid_1d";
            case WakefieldModel::quasistatic_2d:
                return "quasistatic_2d";
            default:
                throw std::runtime_error("Unknown wakefield model!");
        }
    }

void PlasmaStage::operator() (
    amrex::ParticleReal & AMREX_RESTRICT x,
    amrex::ParticleReal & AMREX_RESTRICT y,
    amrex::ParticleReal & AMREX_RESTRICT t,
    amrex::ParticleReal & AMREX_RESTRICT px,
    amrex::ParticleReal & AMREX_RESTRICT py,
    amrex::ParticleReal & AMREX_RESTRICT pt,
    uint64_t & AMREX_RESTRICT idcpu,
    [[maybe_unused]] RefPart const & AMREX_RESTRICT refpart
) const
{
    using namespace amrex::literals; // for _rt and _prt

    // shift due to alignment errors of the element
    shift_in(x, y, px, py);

    // initialize output values
    amrex::ParticleReal xout = x;
    amrex::ParticleReal yout = y;
    amrex::ParticleReal tout = t;
    amrex::ParticleReal pxout = px;
    amrex::ParticleReal pyout = py;
    amrex::ParticleReal ptout = pt;
    amrex::ParticleReal k_beta_ds = 0.0_prt;
    amrex::ParticleReal cos_kds = 1.0_prt;
    amrex::ParticleReal sin_kds = 0.0_prt;
    amrex::ParticleReal r = 0.0_prt;
    amrex::ParticleReal k_beta_r = 0.0_prt;

    // Apply plasma wakefield effects based on model
    switch (m_wakefield_model)
    {
        case WakefieldModel::none:
            // No wakefield effects - just drift
            xout = x + m_slice_ds * px;
            yout = y + m_slice_ds * py;
            tout = t + m_slice_bg * pt;
            break;

        case WakefieldModel::simple_blowout:
            // Simple blowout model with focusing and acceleration
            k_beta_ds = m_k_beta * m_slice_ds;
            cos_kds = std::cos(k_beta_ds);
            sin_kds = std::sin(k_beta_ds);

            // Apply focusing in x and y
            xout = cos_kds * x + sin_kds / m_k_beta * px;
            pxout = -m_k_beta * sin_kds * x + cos_kds * px;

            yout = cos_kds * y + sin_kds / m_k_beta * py;
            pyout = -m_k_beta * sin_kds * y + cos_kds * py;

            // Longitudinal acceleration (constant field)
            ptout = pt;  // No relative energy change
            tout = t + m_slice_bg * ptout;
            break;

        case WakefieldModel::custom_blowout:
            // Custom blowout model with user-defined fields
            k_beta_ds = m_k_beta * m_slice_ds;
            cos_kds = std::cos(k_beta_ds);
            sin_kds = std::sin(k_beta_ds);

            xout = cos_kds * x + sin_kds / m_k_beta * px;
            pxout = -m_k_beta * sin_kds * x + cos_kds * px;

            yout = cos_kds * y + sin_kds / m_k_beta * py;
            pyout = -m_k_beta * sin_kds * y + cos_kds * py;

            ptout = pt;  // No relative energy change
            tout = t + m_slice_bg * ptout;
            break;

        case WakefieldModel::focusing_blowout:
            // Focusing blowout model - only transverse focusing
            k_beta_ds = m_k_beta * m_slice_ds;
            cos_kds = std::cos(k_beta_ds);
            sin_kds = std::sin(k_beta_ds);

            xout = cos_kds * x + sin_kds / m_k_beta * px;
            pxout = -m_k_beta * sin_kds * x + cos_kds * px;

            yout = cos_kds * y + sin_kds / m_k_beta * py;
            pyout = -m_k_beta * sin_kds * y + cos_kds * py;

            tout = t + m_slice_bg * pt;
            break;

        case WakefieldModel::cold_fluid_1d:
            // 1D cold fluid model - longitudinal waves only
            k_beta_ds = m_k_beta * m_slice_ds * 0.5_prt;
            cos_kds = std::cos(k_beta_ds);
            sin_kds = std::sin(k_beta_ds);

            xout = cos_kds * x + sin_kds / (m_k_beta * 0.5_prt) * px;
            pxout = -m_k_beta * 0.5_prt * sin_kds * x + cos_kds * px;

            yout = cos_kds * y + sin_kds / (m_k_beta * 0.5_prt) * py;
            pyout = -m_k_beta * 0.5_prt * sin_kds * y + cos_kds * py;

            ptout = pt;  // No relative energy change
            tout = t + m_slice_bg * ptout;
            break;

        case WakefieldModel::quasistatic_2d:
            // Initialize solver if needed
            if (!m_quasistatic2d_solver)
                m_quasistatic2d_solver = std::make_shared<Quasistatic2DWakefield>(
                    m_density, m_nr, m_nxi, m_dr, m_dxi);
            // Call the solver steps (stubs for now)
            m_quasistatic2d_solver->deposit_charge(/* TODO: pass particles */);
            m_quasistatic2d_solver->solve_bubble_boundary();
            m_quasistatic2d_solver->compute_fields();
            m_quasistatic2d_solver->interpolate_fields(/* TODO: pass particles */);
            // TODO: apply interpolated fields to particles
            // TODO: advance reference particle
            break;

        default:
            // Fallback to drift
            xout = x + m_slice_ds * px;
            yout = y + m_slice_ds * py;
            tout = t + m_slice_bg * pt;
            break;
    }

    // assign updated values
    x = xout;
    y = yout;
    t = tout;
    px = pxout;
    py = pyout;
    pt = ptout;

    // apply transverse aperture
    apply_aperture(x, y, idcpu);

    // undo shift due to alignment errors of the element
    shift_out(x, y, px, py);
}

void Quasistatic2DWakefield::deposit_charge(/* beam particles, etc. */) {}
void Quasistatic2DWakefield::solve_bubble_boundary() {}
void Quasistatic2DWakefield::compute_fields() {}
void Quasistatic2DWakefield::interpolate_fields(/* particle positions, output fields */) {}

} // namespace impactx::elements
