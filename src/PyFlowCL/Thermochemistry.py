"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Thermochemistry.py

"""

__copyright__ = """
Copyright (c) 2022 Jonathan F. MacArt
"""

__license__ = """
 Permission is hereby granted, free of charge, to any person 
 obtaining a copy of this software and associated documentation 
 files (the "Software"), to deal in the Software without 
 restriction, including without limitation the rights to use, 
 copy, modify, merge, publish, distribute, sublicense, and/or 
 sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be 
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import numpy as np


# -------------------------------------------------------------
# General-purpose functions
# -------------------------------------------------------------

# Compute the internal energy from the conserved state q
def get_internal_energy_q(q, interior=False):
    if interior:
        return (( q['rhoE'].interior() -
                  0.5*( q['rhoU'].interior()**2 +
                        q['rhoV'].interior()**2 +
                        q['rhoW'].interior()**2 )/q['rho'].interior() )
                / q['rho'].interior() )

    else:
        return (( q['rhoE'].var -
                  0.5*( q['rhoU'].var**2 +
                        q['rhoV'].var**2 +
                        q['rhoW'].var**2 )/q['rho'].var )
                / q['rho'].var )


# Tensor version of internal energy function (works only for 2D)
def get_internal_energy_q_tensor(q, interior=False):
    if interior:
        return (( q[3][5:-5,5:-5,:] -
                  0.5*( q[1][5:-5,5:-5,:]**2 +
                        q[2][5:-5,5:-5,:]**2)/q[0][5:-5,5:-5,:])  ## FIX! INDICES HARDCODED!
                / q[0][5:-5,5:-5,:] )

    else:
        return (( q[3] -
                  0.5*( q[1]**2 +
                        q[2]**2)/q[0])
                / q[0])


# Get additional scalar names from the InputConfig
def get_add_scalar_names(cfg, sc_names, sc_names_prim):
    if hasattr(cfg, 'additional_scalars'):
        for name in cfg.additional_scalars:
            # Include rho in name for conserved quantities
            sc_names.append('rho' + name)
            sc_names_prim.append(name)
    return

# Get mass fractions from the conserved variables
def get_massfractions(q, names, interior=False):
    Y = []
    if interior:
        for name in names:
            Y.append(q[name].interior() / q['rho'].interior())
    else:
        for name in names:
            Y.append(q[name].var / q['rho'].var)

    return Y

# -------------------------------------------------------------
# Specific EOS+Transport classes
# -------------------------------------------------------------

# -------------------------------------------------------------
# Dimensionless calorically perfect gas
#
class Perfect_Gas_Nondim:
    def __init__(self, cfg):
        self.dimensional = False
        self.combustion  = False
        
        # Data needed by member functions
        self.gamma = cfg.gamma
        self.Re    = cfg.Re
        self.Ma    = cfg.Ma
        self.Pr    = cfg.Pr

        # Reference state
        self.p0    = 1.0 / self.gamma
        self.rho0  = 1.0
        self.T0    = 1.0 / (self.gamma - 1.0)
        self.U0    = 1.0
        self.L0    = 1.0
        self.mu    = 1.0

        # For initial CFL/DN estimates
        self.base_cs = 1.0 / self.Ma
        self.base_mu = self.mu / self.Re

        # Pressure multiplier
        self.P_fac = 1.0 / self.Ma**2

        # Scalars
        self.sc_names_prim = []
        self.sc_names      = []
        get_add_scalar_names(cfg, self.sc_names, self.sc_names_prim)
        self.num_sc = len(self.sc_names)

        # Source terms (0.0 for nonreacting)
        self.srcSC = []
        for isc in range(self.num_sc):
            self.srcSC.append(0.0)

    def get_density(self, p, T, SC=None):
        return self.gamma * p / ((self.gamma - 1.0) * T)

    def get_internal_energy_RPY(self, rho, p, Y=None):
        return p / ((self.gamma - 1.0) * rho * self.Ma**2)

    def get_internal_energy_TY(self, T, Y=None):
        return T / (self.gamma * self.Ma**2)

    def get_T_internal_energy(self, e):
        return self.gamma * self.Ma**2 * e

    def get_T(self, q):
        e = get_internal_energy_q(q)
        return self.get_T_internal_energy(e)

    def get_TPE(self, q, Y=None, interior=False):
        e = get_internal_energy_q(q, interior)
        if interior:
            p = (self.gamma - 1.0) * self.Ma**2 * q['rho'].interior() * e
        else:
            p = (self.gamma - 1.0) * self.Ma**2 * q['rho'].var * e
        T = self.gamma * self.Ma**2 * e
        return T, p, e

    def get_TPE_tensor(self, q, Y=None, interior=False):
        e = get_internal_energy_q_tensor(q, interior)
        if interior:
            p = (self.gamma - 1.0) * self.Ma**2 * q[0][5:-5,5:-5,:] * e ## FIX! HARDCODED!
        else:
            p = (self.gamma - 1.0) * self.Ma**2 * q[0] * e
        T = self.gamma * self.Ma**2 * e
        return T, p, e

    def get_P_rho_internal_energy(self, rho, e):
        return (self.gamma - 1.0) * self.Ma**2 * rho * e

    def get_mu_kappa(self, T):
        mu    = self.base_mu * ((self.gamma - 1.0) * T)**0.7
        kappa = mu / (self.Pr * self.Ma**2)
        return mu, kappa

    def get_soundspeed_T(self, T):
        return torch.sqrt((self.gamma - 1.0)*T) / self.Ma
    
    def get_soundspeed_rp(self, rho, p):
        return torch.sqrt(self.gamma * p / rho) / self.Ma

    def get_soundspeed_q(self, q):
        rho = q['rho'].interior()
        e = get_internal_energy_q(q, True)
        p = self.get_P_rho_internal_energy(rho, e)
        return self.get_soundspeed_rp(rho, p)

    def get_gamma_TY(self, T, Y):
        return self.gamma

    def get_species_diff_coeff(self, T, SC):
        # Need to add transport coefficients here for any species
        # other than mixture fraction (Zmix)
        return 0.0

    def get_species_production_rates(self, rho, T, SC):
        return self.srcSC

    

# -------------------------------------------------------------
# Dimensional calorically perfect gas
#
class Perfect_Gas_Dim:
    def __init__(self, cfg):
        self.dimensional = True
        self.combustion  = False
        
        # Data needed by member functions
        self.gamma = cfg.gamma  # Ratio of specific heats
        self.Rgas  = cfg.Rgas   # Specific gas constant [J/(kg*K)]
        
        self.mu    = cfg.mu     # Viscosity [Pa*s]
        self.Pr    = cfg.Pr     # Prandtl number (const)

        # Reference state
        self.L0    = cfg.L0     # Reference length scale [m]
        self.U0    = cfg.U0     # Reference velocity [m/s]
        self.T0    = cfg.T0     # Reference temperature [K]
        
        if hasattr(cfg, 'p0'):
            self.p0    = cfg.p0     # Reference pressure [N/m^2]
            self.rho0  = self.p0 / (self.Rgas * self.T0)

        else:
            self.rho0  = cfg.rho0
            self.p0    = self.rho0 * (self.Rgas * self.T0)

        # Computed thermodynamic properties
        self.cv = self.Rgas / (self.gamma - 1.0)
        self.cp = self.gamma * self.cv

        # For initial CFL/DN estimates
        self.base_cs = np.sqrt(self.gamma * self.Rgas * self.T0)
        self.base_mu = self.mu

        # Pressure multiplier
        self.P_fac = 1.0

        # Scalars
        self.sc_names_prim = []
        self.sc_names      = []
        get_add_scalar_names(cfg, self.sc_names, self.sc_names_prim)
        self.num_sc = len(self.sc_names)

        # Source terms (0.0 for nonreacting)
        self.srcSC = []
        for isc in range(self.num_sc):
            self.srcSC.append(0.0)

    def get_density(self, p, T, SC=None):
        return p / (self.Rgas * T)

    def get_internal_energy_RPY(self, rho, p, Y=None):
        T = p / (rho * self.Rgas)
        return T * self.cv

    def get_internal_energy_TY(self, T, Y=None):
        return T * self.cv

    def get_T_internal_energy(self, e):
        return e / self.cv

    def get_T(self, q):
        e = get_internal_energy_q(q)
        T = e / self.cv
        return T

    def get_TPE(self, q, Y=None, interior=False):
        e = get_internal_energy_q(q, interior)
        T = e / self.cv
        if interior:
            p = q['rho'].interior() * self.Rgas * T
        else:
            p = q['rho'].var * self.Rgas * T
        return T, p, e

    def get_TPE_tensor(self, q, Y=None, interior=False):
        e = get_internal_energy_q_tensor(q, interior)
        T = e / self.cv
        if interior:
            p = q[0][5:-5,5:-5,:] * self.Rgas * T
        else:
            p = q[0] * self.Rgas * T
        return T, p, e

    def get_P_rho_internal_energy(self, rho, e):
        return (self.gamma - 1.0) * rho * e

    def get_mu_kappa(self, T):
        # Sutherland's viscosity law (source: Viscous Fluid Flow by White)
        # assuming air
        # mu_ref = 1.716e-5       # [Ns/m^2]
        # k_ref = 0.0241          # [W/m/K]
        # T_ref = 273             # [K]
        # S = 111                 # [K]; Sutherland constant
        # sutherland = (T/T_ref)**1.5*(T_ref + S)/(T + S)
        # mu = mu_ref*sutherland
        # kappa = k_ref*sutherland
        
        # Viscocity exponent-law 
        Tref = 1000 #273.15#K
        omeg = 0.734 # Viscocity exponent
        mu = self.base_mu * (T/Tref)**omeg 
        # mu =  self.base_mu * torch.ones_like(T)
        # E
        
        # mu    = self.mu * ((self.gamma - 1.0) * T)**0.7
        kappa = mu * self.cp / self.Pr
        
        return mu, kappa

    def get_soundspeed_q(self, q):
        T = get_internal_energy_q(q, interior=True) / self.cv
        return torch.sqrt(self.gamma * self.Rgas * T)

    def get_soundspeed_T(self, T):
        return torch.sqrt(self.gamma * self.Rgas * T)
    
    def get_soundspeed_rp(self, rho, p):
        return torch.sqrt(self.gamma * p / rho)

    def get_gamma_TY(self, T, Y):
        return self.gamma

    def get_species_diff_coeff(self, T, SC):
        # Need to add transport coefficients here for any species
        # other than mixture fraction (Zmix)
        return 0.0

    def get_species_production_rates(self, rho, T, SC):
        return self.srcSC



# -------------------------------------------------------------
# Dimensional finite-rate chemistry
#   (A wrapper to Pyrometheus-generated thermochemistry)
#
class Finitechem:
    def __init__(self, cfg):
        self.dimensional = True
        self.combustion  = True

        # Pointer to Pyrometheus thermochemistry object
        self.TC    = cfg.TC
        
        # Data needed by member functions
        self.mu    = cfg.mu     # Viscosity [Pa*s]
        self.Pr    = cfg.Pr     # Prandtl number (const)

        # Reference state
        self.L0    = cfg.L0     # Reference length scale [m]
        self.U0    = cfg.U0     # Reference velocity [m/s]
        self.T0    = cfg.T0     # Reference temperature [K]
        self.p0    = cfg.p0     # Reference pressure [N/m^2]
        self.Y0    = cfg.Y0     # Reference species mass fractions
        self.rho0  = self.get_density(self.p0, self.T0, self.Y0)

        # For initial CFL/DN estimates
        #self.base_cs = np.sqrt(self.gamma * self.Rgas * self.T0)
        self.base_cs = 343.0 #self.get_soundspeed_YT(self.Y0, self.T0) ###### FIX
        self.base_mu = self.mu

        # Pressure multiplier
        self.P_fac = 1.0

        # Scalars
        self.sc_names_prim = []
        self.sc_names = []
        for name in self.TC.species_names:
            self.sc_names.append('rhoY_'+name)
            self.sc_names_prim.append('Y_'+name)
        get_add_scalar_names(cfg, self.sc_names, self.sc_names_prim)
        self.num_sc = len(self.sc_names)

    def get_density(self, p, T, SC):
        return self.TC.get_density(p, T, SC[:self.TC.num_species])

    def get_internal_energy_RPY(self, rho, p, Y):
        R = self.get_gas_constant(Y)
        T = p / (rho * R)
        return self.get_internal_energy_TY(T, Y)

    def get_internal_energy_TY(self, T, Y):
        return self.TC.get_mixture_internal_energy_mass(T, Y)

    def get_TY(self, q, interior=False):
        e = get_internal_energy_q(q, interior)
        Y = get_massfractions(q, self.sc_names, interior)
        return self.TC.get_temperature(e, self.T0, Y, do_energy=True), Y

    def get_TPE(self, q, Y=None, interior=False):
        # Function called from RHS where we already have Y (mass fractions)
        # Internal energy
        e = get_internal_energy_q(q, interior)

        # Mass fractions
        if Y is None:
            Y = get_massfractions(q, self.sc_names, interior)
        else:
            Y = Y[:self.TC.num_species]
            
        # Temperature
        T = self.TC.get_temperature(e, self.T0, Y, do_energy=True)

        # Pressure
        if interior:
            p = self.TC.get_pressure(q['rho'].interior(), T, Y)
        else:
            p = self.TC.get_pressure(q['rho'].var, T, Y)
            
        return T, p, e

    def get_gas_constant(self, Y):
        return self.TC.get_specific_gas_constant(Y)

    def get_soundspeed_q(self, q):
        T, Y  = self.get_TY(q, interior=True)
        return self.get_soundspeed_YT(Y, T)

    def get_gamma_TY(self, T, Y):
        ## NOTE: NEED more efficient way to do this. Evaluates NASA polynomials twice!!
        cp = self.TC.get_mixture_specific_heat_cp_mass(T, Y)
        cv = self.TC.get_mixture_specific_heat_cv_mass(T, Y)
        return cp/cv

    def get_soundspeed_YT(self, Y, T):
        ## NOTE: NEED more efficient way to do this. Evaluates NASA polynomials twice!!
        cp = self.TC.get_mixture_specific_heat_cp_mass(T, Y)
        cv = self.TC.get_mixture_specific_heat_cv_mass(T, Y)
        gamma_R = cp*cp/cv - 1.0
        return torch.sqrt(gamma_R * T)

    def get_species_production_rates(self, rho, T, SC):
        ## JFM TODO: Need to implement production rates for non-Y species!
        return self.TC.get_net_production_rates(rho, T, SC[:self.TC.num_species])

    def get_mu_kappa(self, T):
        mu = self.mu * torch.ones_like(T) #* ((self.gamma - 1.0) * T)**0.7   ## JFM FIX
        kappa = mu / self.Pr #mu * self.cp / self.Pr
        return mu, kappa

    def get_species_diff_coeff(self, T, SC):
        diff = []
        for isc in range(self.num_sc):
            diff.append( self.mu * torch.ones_like(T) ) ## JFM FIX

        return diff
