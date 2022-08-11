/* Copyright 2022 The Regents of the University of California, through Lawrence
 *           Berkeley National Laboratory (subject to receipt of any required
 *           approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * This file is part of ImpactX.
 *
 * Authors: Marco Garten
 * License: BSD-3-Clause-LBNL
 */
#ifndef IMPACTX_FORCEFROMSELFFIELDS_H
#define IMPACTX_FORCEFROMSELFFIELDS_H

#include "particles/ImpactXParticleContainer.H"

#include <AMReX_MultiFab.H>

#include <unordered_map>


namespace impactx::spacecharge
{
    /** Calculate the space charge force field from the electric potential
     *
     * This resets the values in scf_<component> to zero and then calculates the space
     * charge force field.
     *
     * @param[inout] phi scalar potential per level
     * @param[inout] scf_x space charge force component for x per level
     * @param[inout] scf_y space charge force component for y per level
     * @param[inout] scf_z space charge force component for z per level
     * @param[in] geom geometry object
     */
    void ForceFromSelfFields (
        std::unordered_map<int, amrex::MultiFab> & phi,
        std::unordered_map<int, amrex::MultiFab> & scf_x,
        std::unordered_map<int, amrex::MultiFab> & scf_y,
        std::unordered_map<int, amrex::MultiFab> & scf_z,
        const amrex::Vector<amrex::Geometry>& geom
    );

} // namespace impactx

#endif // IMPACTX_FORCEFROMSELFFIELDS_H
