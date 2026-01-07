/* Copyright 2022-2026 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors:  Chad Mitchell, Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_UPDATE_TRANSFER_MAP_H
#define IMPACTX_UPDATE_TRANSFER_MAP_H

#include "ImpactX.H"

#include "elements/All.H"
#include "elements/mixin/lineartransport.H"
#include "particles/ReferenceParticle.H"

#include <AMReX_REAL.H>

#include <string>
#include <type_traits>
#include <variant>


namespace impactx
{
    void
    ImpactX::update_transfer_map (
        RefPart & ref,
        elements::KnownElements const & element_variant
    )
    {
        BL_PROFILE("impactx::diagnostics::transfer_map");

        // options
        bool fallback_identity_map = false;  // TODO expose to user

        // extract element transport map, handle fallbacks
        Map6x6 element_transport_map = Map6x6::Identity();
        std::visit([&ref, &element_transport_map, &fallback_identity_map](auto&& el)
        {
            using Element = std::decay_t<decltype(el)>;
            std::string not_impl_msg = "Undefined transfer map in lattice for element ";
            if (el.has_name()) not_impl_msg += el.name() + " ";
            not_impl_msg += std::string("of type ") + Element::type;

            if constexpr (std::is_base_of_v<elements::mixin::LinearTransport<Element>, Element>) {
                try {
                    element_transport_map = el.transport_map(ref);
                } catch (std::exception const & e) {
                    if (!fallback_identity_map) {
                        throw std::runtime_error(not_impl_msg);
                    }
                }
            } else {
                if (!fallback_identity_map) {
                    throw std::runtime_error(not_impl_msg);
                }
            }
        }, element_variant);

        // advance linear transfer map
        m_linear_transfer_map = m_linear_transfer_map * element_transport_map;
        // TODO: shorthand needs https://github.com/AMReX-Codes/amrex/pull/4880 from AMReX 26.02+
        // m_linear_transfer_map *= element_transport_map;
    }

} // namespace impactx

#endif // IMPACTX_REDUCED_BEAM_CHARACTERISTICS
