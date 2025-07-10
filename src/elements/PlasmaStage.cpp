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

} // namespace impactx
