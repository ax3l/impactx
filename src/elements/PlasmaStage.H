/* Copyright 2022-2023 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_PLASMASTAGE_H
#define IMPACTX_PLASMASTAGE_H

#include "particles/ImpactXParticleContainer.H"
#include "mixin/alignment.H"
#include "mixin/beamoptic.H"
#include "mixin/thick.H"
#include "mixin/lineartransport.H"
#include "mixin/named.H"
#include "mixin/nofinalize.H"
#include "mixin/pipeaperture.H"

#include <AMReX_Extension.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <stdexcept>
#include <functional>
#include <cmath> // M_PI
#include <memory> // For std::unique_ptr


namespace impactx::elements
{
    // Forward declaration
    class Quasistatic2DWakefield;

    struct PlasmaStage
    : public mixin::Named,
      public mixin::BeamOptic<PlasmaStage>,
      public mixin::LinearTransport<PlasmaStage>,
      public mixin::Thick,
      public mixin::Alignment,
      public mixin::PipeAperture,
      public mixin::NoFinalize
    {
        static constexpr auto type = "PlasmaStage";
        using PType = ImpactXParticleContainer::ParticleType;

        // Plasma wakefield models
        enum WakefieldModel
        {
            none,
            simple_blowout,
            custom_blowout,
            focusing_blowout,
            cold_fluid_1d,
            quasistatic_2d
        };

        static std::string
        wakefield_model_name (WakefieldModel const & model);

        static WakefieldModel
        wakefield_model_from_string (std::string const & model) {
            if (model == "none") return WakefieldModel::none;
            if (model == "simple_blowout") return WakefieldModel::simple_blowout;
            if (model == "custom_blowout") return WakefieldModel::custom_blowout;
            if (model == "focusing_blowout") return WakefieldModel::focusing_blowout;
            if (model == "cold_fluid_1d") return WakefieldModel::cold_fluid_1d;
            if (model == "quasistatic_2d") return WakefieldModel::quasistatic_2d;
            throw std::runtime_error("Unknown wakefield_model: " + model);
        }

        /** A plasma stage element that applies plasma wakefield effects
         *
         * @param ds Segment length in m
         * @param density Plasma density in m^(-3)
         * @param wakefield_model Wakefield model to use
         * @param dx horizontal translation error in m
         * @param dy vertical translation error in m
         * @param rotation_degree rotation error in the transverse plane [degrees]
         * @param aperture_x horizontal half-aperture in m
         * @param aperture_y vertical half-aperture in m
         * @param nslice number of slices used for the application of space charge
         * @param name a user defined and not necessarily unique name of the element
         */
        PlasmaStage (
            amrex::ParticleReal ds,
            amrex::ParticleReal density,
            WakefieldModel wakefield_model = WakefieldModel::simple_blowout,
            amrex::ParticleReal dx = 0,
            amrex::ParticleReal dy = 0,
            amrex::ParticleReal rotation_degree = 0,
            amrex::ParticleReal aperture_x = 0,
            amrex::ParticleReal aperture_y = 0,
            int nslice = 1,
            std::optional<std::string> name = std::nullopt
        )
        : Named(std::move(name)),
          Thick(ds, nslice),
          Alignment(dx, dy, rotation_degree),
          PipeAperture(aperture_x, aperture_y),
          m_wakefield_model(wakefield_model),
          m_density(density)
        {
            using namespace amrex::literals; // for _rt and _prt

            if (density <= 0_prt) {
                throw std::runtime_error("PlasmaStage: density must be > 0!");
            }
        }

        // Rule of five for unique_ptr
        PlasmaStage(const PlasmaStage&) = default;
        PlasmaStage& operator=(const PlasmaStage&) = default;
        PlasmaStage(PlasmaStage&&) = default;
        PlasmaStage& operator=(PlasmaStage&&) = default;

        /** Push all particles */
        using BeamOptic::operator();

        /** Compute and cache the constants for the push.
         *
         * In particular, used to pre-compute and cache variables that are
         * independent of the individually tracked particle.
         *
         * @param refpart reference particle
         */
        void compute_constants (RefPart const & refpart)
        {
            using namespace amrex::literals; // for _rt and _prt
            using amrex::Math::powi;

            Alignment::compute_constants(refpart);

            // length of the current slice
            m_slice_ds = m_ds / nslice();

            // find beta*gamma^2
            m_betgam2 = powi<2>(refpart.pt) - 1.0_prt;

            m_slice_bg = m_slice_ds / m_betgam2;

            // Compute plasma frequency and related constants
            // Plasma frequency: omega_p = sqrt(n_e * e^2 / (m_e * epsilon_0))
            // where n_e is electron density, e is elementary charge,
            // m_e is electron mass, epsilon_0 is vacuum permittivity
            constexpr amrex::ParticleReal e = 1.602176634e-19_prt;  // elementary charge [C]
            constexpr amrex::ParticleReal m_e = 9.1093837015e-31_prt;  // electron mass [kg]
            constexpr amrex::ParticleReal epsilon_0 = 8.8541878128e-12_prt;  // vacuum permittivity [F/m]
            constexpr amrex::ParticleReal c = 299792458.0_prt;  // speed of light [m/s]

            m_omega_p = std::sqrt(m_density * e * e / (m_e * epsilon_0));
            m_k_p = m_omega_p / c;  // plasma wavenumber

            // Compute constants for different wakefield models
            switch (m_wakefield_model)
            {
                case WakefieldModel::simple_blowout:
                    // Simple blowout model: focusing and acceleration
                    m_k_beta = m_omega_p / (2.0_prt * std::sqrt(m_betgam2 + 1.0_prt));
                    m_e_z_p = m_omega_p * m_omega_p * m_e / (2.0_prt * e);  // accelerating field
                    m_l_p = 2.0_prt * M_PI * c / m_omega_p;  // plasma wavelength
                    break;

                case WakefieldModel::custom_blowout:
                    // Custom blowout model: user-defined fields
                    m_k_beta = m_omega_p / (2.0_prt * std::sqrt(m_betgam2 + 1.0_prt));
                    m_e_z_p = m_omega_p * m_omega_p * m_e / (2.0_prt * e);
                    m_e_z_slope = 0.0_prt;  // user can modify this
                    break;

                case WakefieldModel::focusing_blowout:
                    // Focusing blowout model: only transverse focusing
                    m_k_beta = m_omega_p / (2.0_prt * std::sqrt(m_betgam2 + 1.0_prt));
                    break;

                case WakefieldModel::cold_fluid_1d:
                    // 1D cold fluid model: longitudinal waves only
                    m_k_beta = m_omega_p / (2.0_prt * std::sqrt(m_betgam2 + 1.0_prt));
                    m_e_z_p = m_omega_p * m_omega_p * m_e / (2.0_prt * e);
                    break;

                case WakefieldModel::quasistatic_2d:
                    // 2D quasistatic model: full 2D wakefield
                    m_k_beta = m_omega_p / (2.0_prt * std::sqrt(m_betgam2 + 1.0_prt));
                    m_e_z_p = m_omega_p * m_omega_p * m_e / (2.0_prt * e);
                    break;

                case WakefieldModel::none:
                default:
                    // No wakefield effects
                    m_k_beta = 0.0_prt;
                    m_e_z_p = 0.0_prt;
                    break;
            }
        }

        /** This is a plasma stage functor, so that a variable of this type can be used like a plasma stage function.
         *
         * The @see compute_constants method must be called before pushing particles through this operator.
         *
         * @param x particle position in x
         * @param y particle position in y
         * @param t particle position in t
         * @param px particle momentum in x
         * @param py particle momentum in y
         * @param pt particle momentum in t
         * @param idcpu particle global index
         * @param refpart reference particle (unused)
         */
        void operator() (
            amrex::ParticleReal & AMREX_RESTRICT x,
            amrex::ParticleReal & AMREX_RESTRICT y,
            amrex::ParticleReal & AMREX_RESTRICT t,
            amrex::ParticleReal & AMREX_RESTRICT px,
            amrex::ParticleReal & AMREX_RESTRICT py,
            amrex::ParticleReal & AMREX_RESTRICT pt,
            uint64_t & AMREX_RESTRICT idcpu,
            [[maybe_unused]] RefPart const & AMREX_RESTRICT refpart
        ) const;

        /** This pushes the reference particle.
         *
         * @param[in,out] refpart reference particle
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (RefPart & AMREX_RESTRICT refpart) const
        {
            using namespace amrex::literals; // for _rt and _prt
            using amrex::Math::powi;

            // assign input reference particle values
            amrex::ParticleReal const x = refpart.x;
            amrex::ParticleReal const px = refpart.px;
            amrex::ParticleReal const y = refpart.y;
            amrex::ParticleReal const py = refpart.py;
            amrex::ParticleReal const z = refpart.z;
            amrex::ParticleReal const pz = refpart.pz;
            amrex::ParticleReal const t = refpart.t;
            amrex::ParticleReal const pt = refpart.pt;
            amrex::ParticleReal const s = refpart.s;

            // length of the current slice
            amrex::ParticleReal const slice_ds = m_ds / nslice();

            // assign intermediate parameter
            amrex::ParticleReal const step = slice_ds / std::sqrt(powi<2>(pt) - 1.0_prt);

            // advance position and momentum (drift through plasma)
            refpart.x = x + step * px;
            refpart.y = y + step * py;
            refpart.z = z + step * pz;

            // Apply wakefield effects to reference particle based on model
            amrex::ParticleReal pt_new = pt;
            amrex::ParticleReal t_new = t;

            switch (m_wakefield_model)
            {
                case WakefieldModel::simple_blowout:
                case WakefieldModel::custom_blowout:
                case WakefieldModel::cold_fluid_1d:
                case WakefieldModel::quasistatic_2d: {
                    // Apply longitudinal acceleration
                    amrex::ParticleReal const E_z = m_e_z_p * slice_ds;
                    pt_new = pt + E_z;
                    t_new = t - step * pt_new;
                    break;
                }

                case WakefieldModel::focusing_blowout:
                    // Only transverse focusing, no longitudinal acceleration
                    t_new = t - step * pt;
                    break;

                case WakefieldModel::none:
                default:
                    // No wakefield effects
                    t_new = t - step * pt;
                    break;
            }

            refpart.t = t_new;
            refpart.pt = pt_new;

            // advance integrated path length
            refpart.s = s + slice_ds;
        }

        /** This pushes the covariance matrix. */
        using LinearTransport::operator();

        /** This function returns the linear transport map.
         *
         * @param[in] refpart reference particle
         * @returns 6x6 transport matrix
         */
        AMREX_GPU_HOST AMREX_FORCE_INLINE
        Map6x6
        transport_map (RefPart const & AMREX_RESTRICT /*refpart*/) const
        {
            throw std::runtime_error("PlasmaStage does not implement a linear transport map.");
        }

        WakefieldModel m_wakefield_model; //! wakefield model type
        amrex::ParticleReal m_density; //! plasma density [m^(-3)]
        amrex::ParticleReal m_omega_p; //! plasma frequency [rad/s]
        amrex::ParticleReal m_k_p; //! plasma wavenumber [m^(-1)]
        amrex::ParticleReal m_k_beta; //! betatron wavenumber [m^(-1)]
        amrex::ParticleReal m_e_z_p; //! longitudinal electric field [V/m]
        amrex::ParticleReal m_e_z_slope; //! longitudinal field slope [V/m^2]
        amrex::ParticleReal m_l_p; //! plasma wavelength [m]
        amrex::ParticleReal m_slice_ds; //! slice length [m]
        amrex::ParticleReal m_betgam2; //! beta*gamma^2
        amrex::ParticleReal m_slice_bg; //! slice length / (beta*gamma^2)

        // Add a shared_ptr to the 2D quasistatic solver (optional, can be shared_ptr or value)
        mutable std::shared_ptr<Quasistatic2DWakefield> m_quasistatic2d_solver;

        // Parameters for the (r, xi) grid for the 2D quasistatic solver
        int m_nr = 64; // number of radial grid points
        int m_nxi = 128; // number of longitudinal grid points
        amrex::ParticleReal m_dr = 1.0e-5; // radial grid spacing [m]
        amrex::ParticleReal m_dxi = 1.0e-5; // longitudinal grid spacing [m]

    };

    // Skeleton for the 2D quasistatic wakefield solver
    class Quasistatic2DWakefield {
    public:
        // Grid parameters
        int nr, nxi;
        amrex::ParticleReal dr, dxi;

        // Constructor: set up grid, plasma/beam params, etc.
        Quasistatic2DWakefield(
            amrex::ParticleReal plasma_density,
            int nr_, int nxi_, amrex::ParticleReal dr_, amrex::ParticleReal dxi_
        ) : nr(nr_), nxi(nxi_), dr(dr_), dxi(dxi_) {}

        // Deposit beam charge onto (r, xi) grid
        void deposit_charge(/* beam particles, etc. */);

        // Solve bubble boundary ODE for each xi
        void solve_bubble_boundary();

        // Compute fields (E_r, E_z, B_theta) on grid
        void compute_fields();

        // Interpolate fields to particle positions
        void interpolate_fields(/* particle positions, output fields */);

        // (Optionally) Advance reference particle
        void advance_reference(/* ref particle */);

        // Grid and field storage (to be defined)
        // std::vector<std::vector<amrex::ParticleReal>> rho, Er, Ez, Btheta, ...;

        // ... other members ...
    };

} // namespace impactx

#endif // IMPACTX_PLASMASTAGE_H
