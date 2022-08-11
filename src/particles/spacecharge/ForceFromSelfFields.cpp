/* Copyright 2022 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Marco Garten
 * License: BSD-3-Clause-LBNL
 */
#include "ForceFromSelfFields.H"

#include <ablastr/fields/PoissonSolver.H>

#include <AMReX_BLProfiler.H>
#include <AMReX_REAL.H>       // for ParticleReal
#include <AMReX_SPACE.H>      // for AMREX_D_DECL



namespace impactx::spacecharge
{
    void ForceFromSelfFields (
        std::unordered_map<int, amrex::MultiFab> & phi,

        std::unordered_map<int, amrex::MultiFab> & scf_x,
        std::unordered_map<int, amrex::MultiFab> & scf_y,
        std::unordered_map<int, amrex::MultiFab> & scf_z,
        const amrex::Vector<amrex::Geometry>& geom
    )
    {
        using namespace amrex::literals;


        int const finest_level = phi.size() - 1u;
        // loop over refinement levels
        for (int lev = 0; lev <= finest_level; ++lev) {

            const auto &gm = geom[lev];
            const auto dx = gm.CellSizeArray();

            const amrex::GpuArray<amrex::Real, 3> inv2dr {AMREX_D_DECL(2._rt/dx[0], 2._rt/dx[1], 2._rt/dx[2])};

            // reset the values in space_charge_force to zero
            scf_x.at(lev).setVal(0.);
            scf_y.at(lev).setVal(0.);
            scf_z.at(lev).setVal(0.);

            for (amrex::MFIter mfi(phi[lev]); mfi.isValid(); ++mfi) {

                amrex::Box bx = mfi.validbox();
                const auto phi_arr = (phi[lev])[mfi].array();

                amrex::MultiFab &force_x_at_level = scf_x.at(lev);
                amrex::MultiFab &force_y_at_level = scf_y.at(lev);
                amrex::MultiFab &force_z_at_level = scf_z.at(lev);

                auto scf_arr_x = force_x_at_level[mfi].array();
                auto scf_arr_y = force_y_at_level[mfi].array();
                auto scf_arr_z = force_z_at_level[mfi].array();

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    scf_arr_x(i,j,k) = inv2dr[0] * (phi_arr(i-1,j,k) - phi_arr(i+1,j,k));
                    scf_arr_y(i,j,k) = inv2dr[1] * (phi_arr(i,j-1,k) - phi_arr(i,j+1,k));
                    scf_arr_z(i,j,k) = inv2dr[2] * (phi_arr(i,j,k-1) - phi_arr(i,j,k+1));
                });

            }

        }


    }
} // impactx::spacecharge
