"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Metrics.py

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

import copy

import torch
import numpy as np

# --------------------------------------------------
# Functions needed for MUSCL / HLLE fluxes
# --------------------------------------------------
def get_interp_ratios(q_im1, q_i, q_ip1, q_ip2):
    eps = 1e-12
    Delta_im_12 = q_i - q_im1
    Delta_ip_12 = q_ip1 - q_i
    Delta_ip_32 = q_ip2 - q_ip1
    
    # r+_{i-1/2}
    rp_im_12 = Delta_ip_12 / (Delta_im_12 + eps)

    # r-_{i+3/2}
    rm_ip_32 = Delta_ip_12 / (Delta_ip_32 + eps)

    return rp_im_12, rm_ip_32


def limiter(r):
    # Minmod
    phi = torch.maximum(torch.zeros_like(r),
                        torch.minimum(r, torch.ones_like(r)))
    # Superbee
    #ones = torch.ones_like(r)
    #phi = torch.maximum(torch.zeros_like(r),
    #                    torch.maximum(torch.minimum(2*r, ones),
    #                                  torch.minimum(r, 2*ones)))
    return phi


def global_monotonicity(v_im1, v_i, v_ip1, vL, vR):
    # Signs of original and reconstructed states
    #s_orig   = torch.sign(v_ip1 - v_i)
    #s_interp = torch.sign(vR - vL)
    sL = torch.sign(v_i - v_im1)
    sR = torch.sign(v_ip1 - v_i)

    # Mask is true where signs match
    #mask = (s_orig == s_interp)
    mask = (sL == sR)

    # Correct L/R states where signs do not match
    vL_new = mask*vL + ~mask*0.5*(vL + vR)
    vR_new = mask*vR + ~mask*vL_new

    #vL_new = 0.5*(vL + vR)  ## Need this for cylinder to get smooth solutions!
    #vR_new = vL_new

    #vL_new = vL
    #vR_new = vR

    return vL_new, vR_new


def get_MUSCL_LR(q_im1, q_i, q_ip1, q_ip2, xiL, xiR, species=None):  ### NOTE eps=0 : piecewise linear
    eps = 0.0
    if (species is None):
        # Normal single-variable reconstruction
        # Interpolation ratios
        rL, rR = get_interp_ratios(q_im1, q_i, q_ip1, q_ip2)

        # Left and right states
        qL = q_i   + 0.5*eps*(1.0-xiL) * limiter(rL) * (q_i - q_im1)
        qR = q_ip1 - 0.5*eps*(1.0-xiR) * limiter(rR) * (q_ip2 - q_ip1)

        #return qL, qR
        return global_monotonicity(q_im1, q_i, q_ip1, qL, qR)
    else:
        # Combined reconstruction of mixture density & species partial densities
        Y_im1 = species[0]
        Y_i   = species[1]
        Y_ip1 = species[2]
        Y_ip2 = species[3]
        nspecies = len(Y_i)

        # Get most limiting reconstruction
        # Density
        rL, rR = get_interp_ratios(q_im1, q_i, q_ip1, q_ip2)
        limiter_L = limiter(rL)
        limiter_R = limiter(rR)
        
        # Species partial densities
        for isc in range(nspecies):
            rL, rR = get_interp_ratios(Y_im1[isc], Y_i[isc], Y_ip1[isc], Y_ip2[isc])
            limiter_L = torch.minimum(limiter_L, limiter(rL))
            limiter_R = torch.minimum(limiter_R, limiter(rR))

        # Left and right states
        # Density
        qL = q_i   + 0.5*eps*(1.0-xiL) * limiter_L * (q_i - q_im1)
        qR = q_ip1 - 0.5*eps*(1.0-xiR) * limiter_R * (q_ip2 - q_ip1)
        
        # Species partial densities
        YL = []
        YR = []
        for isc in range(nspecies):
            YL.append( Y_i[isc]   + 0.5*eps*(1.0-xiL) * limiter_L * (Y_i[isc]   - Y_im1[isc]) )
            YR.append( Y_ip1[isc] - 0.5*eps*(1.0-xiR) * limiter_R * (Y_ip2[isc] - Y_ip1[isc]) )

        return qL, qR, YL, YR


def HLLE(FL, FR, FS, SL, SR):
    zeros = torch.zeros_like(FL)
    ones  = torch.ones_like(FL)
    
    return ( FL * torch.heaviside( SL, zeros) +
             FS * torch.heaviside(-SL, ones ) * torch.heaviside(SR, ones) +
             FR * torch.heaviside(-SR, zeros) )


# --------------------------------------------------
# Functions needed for 1st-order upwind fluxes
# --------------------------------------------------
def Upwind(F_im1, F_i, F_ip1, SL, SR):
    return SR*(F_i - F_im1) - SL*(F_ip1 - F_i)


# --------------------------------------------------
# Functions needed for Steger-Warming fluxes
# --------------------------------------------------
def jac_matmul(S_inv, R_inv, C_inv, L, C, R, S):
    mat = torch.matmul(S_inv, R_inv)
    mat = torch.matmul(mat, C_inv)
    mat = torch.matmul(mat, L)
    mat = torch.matmul(mat, C)
    mat = torch.matmul(mat, R)
    mat = torch.matmul(mat, S)
    return mat

def jac_matmul_2(R, D, L):
    mat = torch.matmul(R, D)
    mat = torch.matmul(mat, L)
    return mat

def interp_im(phi):
    return 0.5*(phi[1:,...] + phi[:-1,...])

def interp_jm(phi):
    return 0.5*(phi[:,1:,...] + phi[:,:-1,...])

def interp_im_weighted(phi, w1):
    #w2 = 1.0 - w1
    #return w2*phi[1:,...] + w1*phi[:-1,...]
    phi_L = phi[:-1,...]
    phi_R = phi[1:,...]
    diff = phi_L - phi_R
    return w1*diff + phi_R, phi_L - w1*diff
    
def interp_jm_weighted(phi, w1):
    #w2 = 1.0 - w1
    #return w2*phi[:,1:,...] + w1*phi[:,:-1,...]
    phi_L = phi[:,:-1,...]
    phi_R = phi[:,1:,...]
    diff = phi_L - phi_R
    return w1*diff + phi_R, phi_L - w1*diff

def Euler_split_flux_Jac(u, v, c, vn, g, nx, ny, DA_m, DA_p, RA, LA):
    gm1 = g - 1.0
    ov_gm1 = 1.0/gm1

    c2 = c*c
    c2_gm1 = c2*ov_gm1
    
    ek = 0.5*(u*u + v*v)
    H  = c2_gm1 + ek

    # Eigenvalues
    L1 = vn - c
    L2 = vn
    L3 = vn + c
    L1m = 0.5*(L1 - torch.abs(L1))
    L2m = 0.5*(L2 - torch.abs(L2))
    L3m = 0.5*(L3 - torch.abs(L3))
    L1p = 0.5*(L1 + torch.abs(L1))
    L2p = 0.5*(L2 + torch.abs(L2))
    L3p = 0.5*(L3 + torch.abs(L3))

    DA_m[...,0,0] = L1m
    DA_m[...,1,1] = L2m
    DA_m[...,2,2] = L2m
    DA_m[...,3,3] = L3m

    DA_p[...,0,0] = L1p
    DA_p[...,1,1] = L2p
    DA_p[...,2,2] = L2p
    DA_p[...,3,3] = L3p

    # Right eigenvectors
    RA[...,0,0] = 1.0
    RA[...,0,1] = nx
    RA[...,0,2] = ny
    RA[...,0,3] = 1.0
    RA[...,1,0] = u - c*nx
    RA[...,1,1] = u*nx + 1.0 - nx**2
    RA[...,1,2] = u*ny - nx*ny
    RA[...,1,3] = u + c*nx
    RA[...,2,0] = v - c*ny
    RA[...,2,1] = v*nx - nx*ny
    RA[...,2,2] = v*ny + 1.0 - ny**2
    RA[...,2,3] = v + c*ny
    RA[...,3,0] = H - vn*c
    RA[...,3,1] = (ek - vn)*nx + u
    RA[...,3,2] = (ek - vn)*ny + v
    RA[...,3,3] = H + vn*c

    # Left eigenvectors
    LA[...,0,0] = ek + vn*c*ov_gm1
    LA[...,0,1] = -u - c*ov_gm1*nx
    LA[...,0,2] = -v - c*ov_gm1*ny
    LA[...,0,3] = 1.0
    LA[...,1,0] = 2.0*(c2_gm1 - ek)*nx + 2.0*c2_gm1*(vn*nx - u)
    LA[...,1,1] = 2.0*nx*u + 2.0*c2_gm1*(1.0 - nx**2)
    LA[...,1,2] = 2.0*nx*v - 2.0*c2_gm1*nx*ny
    LA[...,1,3] = -2.0*nx
    LA[...,2,0] = 2.0*(c2_gm1 - ek)*ny + 2.0*c2_gm1*(vn*ny - v)
    LA[...,2,1] = 2.0*ny*u - 2.0*c2_gm1*nx*ny
    LA[...,2,2] = 2.0*ny*v + 2.0*c2_gm1*(1.0 - ny**2)
    LA[...,2,3] = -2.0*ny
    LA[...,3,0] = ek - vn*c*ov_gm1
    LA[...,3,1] = c*ov_gm1*nx - u
    LA[...,3,2] = c*ov_gm1*ny - v
    LA[...,3,3] = 1.0

    LA *= 0.5*gm1/c2[...,None,None]

    return

def Euler_split_flux_Jac_2(u, v, c, vn, g, nx, ny, Q, A, direction, save_jac):
    # Second version, no in-place ops, differentiable
    gm1 = g - 1.0
    ov_gm1 = 1.0/gm1

    c2 = c*c
    c2_gm1 = c2*ov_gm1
    
    ek = 0.5*(u*u + v*v)
    H  = c2_gm1 + ek

    # Eigenvalues
    D0 = vn - c
    D1 = vn
    D3 = vn + c
    if direction=='-':
        D_0 = 0.5*(D0 - torch.abs(D0))
        D_1 = 0.5*(D1 - torch.abs(D1))
        D_2 = D_1
        D_3 = 0.5*(D3 - torch.abs(D3))
    elif direction=='+':
        D_0 = 0.5*(D0 + torch.abs(D0))
        D_1 = 0.5*(D1 + torch.abs(D1))
        D_2 = D_1
        D_3 = 0.5*(D3 + torch.abs(D3))
    else:
        raise Exception('Euler_split_flux_Jac_2: Direction not recognized')

    # Right eigenvectors
    R_00 = 1.0
    R_01 = nx
    R_02 = ny
    R_03 = 1.0
    R_10 = u - c*nx
    R_11 = u*nx + 1.0 - nx**2
    R_12 = u*ny - nx*ny
    R_13 = u + c*nx
    R_20 = v - c*ny
    R_21 = v*nx - nx*ny
    R_22 = v*ny + 1.0 - ny**2
    R_23 = v + c*ny
    R_30 = H - vn*c
    R_31 = (ek - vn)*nx + u
    R_32 = (ek - vn)*ny + v
    R_33 = H + vn*c

    # Left eigenvectors
    L_00 = ek + vn*c*ov_gm1
    L_01 = -u - c*ov_gm1*nx
    L_02 = -v - c*ov_gm1*ny
    L_03 = 1.0
    L_10 = 2.0*(c2_gm1 - ek)*nx + 2.0*c2_gm1*(vn*nx - u)
    L_11 = 2.0*nx*u + 2.0*c2_gm1*(1.0 - nx**2)
    L_12 = 2.0*nx*v - 2.0*c2_gm1*nx*ny
    L_13 = -2.0*nx
    L_20 = 2.0*(c2_gm1 - ek)*ny + 2.0*c2_gm1*(vn*ny - v)
    L_21 = 2.0*ny*u - 2.0*c2_gm1*nx*ny
    L_22 = 2.0*ny*v + 2.0*c2_gm1*(1.0 - ny**2)
    L_23 = -2.0*ny
    L_30 = ek - vn*c*ov_gm1
    L_31 = c*ov_gm1*nx - u
    L_32 = c*ov_gm1*ny - v
    L_33 = 1.0

    L_fac = 0.5*gm1/c2

    # F = (R*D*L) * Q
    A_00 = (R_00*D_0*L_00 + R_01*D_1*L_10 + R_02*D_2*L_20 + R_03*D_3*L_30)
    A_01 = (R_00*D_0*L_01 + R_01*D_1*L_11 + R_02*D_2*L_21 + R_03*D_3*L_31)
    A_02 = (R_00*D_0*L_02 + R_01*D_1*L_12 + R_02*D_2*L_22 + R_03*D_3*L_32)
    A_03 = (R_00*D_0*L_03 + R_01*D_1*L_13 + R_02*D_2*L_23 + R_03*D_3*L_33)

    A_10 = (R_10*D_0*L_00 + R_11*D_1*L_10 + R_12*D_2*L_20 + R_13*D_3*L_30)
    A_11 = (R_10*D_0*L_01 + R_11*D_1*L_11 + R_12*D_2*L_21 + R_13*D_3*L_31)
    A_12 = (R_10*D_0*L_02 + R_11*D_1*L_12 + R_12*D_2*L_22 + R_13*D_3*L_32)
    A_13 = (R_10*D_0*L_03 + R_11*D_1*L_13 + R_12*D_2*L_23 + R_13*D_3*L_33)

    A_20 = (R_20*D_0*L_00 + R_21*D_1*L_10 + R_22*D_2*L_20 + R_23*D_3*L_30)
    A_21 = (R_20*D_0*L_01 + R_21*D_1*L_11 + R_22*D_2*L_21 + R_23*D_3*L_31)
    A_22 = (R_20*D_0*L_02 + R_21*D_1*L_12 + R_22*D_2*L_22 + R_23*D_3*L_32)
    A_23 = (R_20*D_0*L_03 + R_21*D_1*L_13 + R_22*D_2*L_23 + R_23*D_3*L_33)

    A_30 = (R_30*D_0*L_00 + R_31*D_1*L_10 + R_32*D_2*L_20 + R_33*D_3*L_30)
    A_31 = (R_30*D_0*L_01 + R_31*D_1*L_11 + R_32*D_2*L_21 + R_33*D_3*L_31)
    A_32 = (R_30*D_0*L_02 + R_31*D_1*L_12 + R_32*D_2*L_22 + R_33*D_3*L_32)
    A_33 = (R_30*D_0*L_03 + R_31*D_1*L_13 + R_32*D_2*L_23 + R_33*D_3*L_33)

    F_0 = A_00*Q[0] + A_01*Q[1] + A_02*Q[2] + A_03*Q[3]
    F_1 = A_10*Q[0] + A_11*Q[1] + A_12*Q[2] + A_13*Q[3]
    F_2 = A_20*Q[0] + A_21*Q[1] + A_22*Q[2] + A_23*Q[3]
    F_3 = A_30*Q[0] + A_31*Q[1] + A_32*Q[2] + A_33*Q[3]

    # Save Jacobian if needed
    if save_jac:
        A[0,0,...] = A_00
        A[0,1,...] = A_01
        A[0,2,...] = A_02
        A[0,3,...] = A_03
        A[1,0,...] = A_10
        A[1,1,...] = A_11
        A[1,2,...] = A_12
        A[1,3,...] = A_13
        A[2,0,...] = A_20
        A[2,1,...] = A_21
        A[2,2,...] = A_22
        A[2,3,...] = A_23
        A[3,0,...] = A_30
        A[3,1,...] = A_31
        A[3,2,...] = A_32
        A[3,3,...] = A_33
        A *= L_fac[None,None,...]

    return torch.stack((F_0, F_1, F_2, F_3), dim=0) * L_fac[None,...]
    


# ------------------------------------------------------
# Collocated 4th-order CD schemes for uniform grids
#   Periodic boundary conditions in \xi
#   No-slip wall at -\eta
#   Farfield boundary at +\eta
#   Could improve with higher-order \eta boundary schemes
#   z : periodic, strictly rectilinear
# ------------------------------------------------------
class central_4th_periodicRectZ:
    def __init__(self, grid, decomp):
        # Working precision
        self.WP = decomp.WP
        
        # Grid spacings
        # Xi
        if (decomp.nx > 1):
            self.d_xi    = grid.d_xi
            self.d_xi4_i = 1.0/grid.d_xi**4
        # Eta
        self.d_eta    = grid.d_eta
        self.d_eta4_i = 1.0/grid.d_eta**4
        # Z
        if (decomp.nz > 1):
            self.d_z    = grid.d_z
            self.d_z4_i = 1.0/grid.d_z**4

        # Save BC info
        self.periodic_xi = grid.periodic_xi
        self.BC_eta_top = grid.BC_eta_top
        self.BC_eta_bot = grid.BC_eta_bot

        if (grid.BC_eta_top=='periodic' and grid.BC_eta_bot=='periodic'):
            self.periodic_eta = True
        else:
            self.periodic_eta = False

        # This task's location in the communicator
        self.device = decomp.device
        self.iproc = decomp.iproc; self.npx = decomp.npx
        self.jproc = decomp.jproc; self.npy = decomp.npy
        self.kproc = decomp.kproc; self.npz = decomp.npz

        # Local interior indices
        self.nx_ = decomp.nx_
        self.ny_ = decomp.ny_
        self.nz_ = decomp.nz_
        self.imin_ = decomp.imin_;  self.imax_ = decomp.imax_+1
        self.jmin_ = decomp.jmin_;  self.jmax_ = decomp.jmax_+1
        self.kmin_ = decomp.kmin_;  self.kmax_ = decomp.kmax_+1

        # Size of extended interior
        self.noveri = int(np.ceil( 0.5*decomp.nover ))
        self.nxi_ = decomp.nx_ + 2*self.noveri
        self.nyi_ = decomp.ny_ + 2*self.noveri
        self.nzi_ = decomp.nz_ + 2*self.noveri
        
        # Indices for extended interiors
        #    |     Overlap     |      Interior    |     Overlap     |
        #    |      nover      |        nx_       |      nover      |
        #    |        | noveri |                  | noveri |        |
        #    |--------|++++++++|==================|++++++++|--------|
        #  imino_   imini_   imin_              imax_    imaxi_   imaxo_
        #
        self.imini_ = self.imin_-self.noveri; self.imaxi_ = self.imax_+self.noveri
        self.jmini_ = self.jmin_-self.noveri; self.jmaxi_ = self.jmax_+self.noveri
        self.kmini_ = self.kmin_-self.noveri; self.kmaxi_ = self.kmax_+self.noveri
            
        # Enforce X dimensionality
        self.nx = decomp.nx
        if (self.nx==1):
            self.nxi_ = 1
            self.imini_ = 0
            self.imaxi_ = 1

        # Enforce Z dimensionality
        self.nz = decomp.nz
        if (self.nz==1):
            self.nzi_ = 1
            self.kmini_ = 0
            self.kmaxi_ = 1

        # Metrics are initialized without grid transforms
        self.have_transforms = False

        # Euler fluxes: compute one additional point to the left
        self.imin_b = self.imin_-1;  self.imax_b = self.imax_
        self.jmin_b = self.jmin_-1;  self.jmax_b = self.jmax_
        self.kmin_b = self.kmin_-1;  self.kmax_b = self.kmax_

        # Euler fluxes: Default is not to save the Jacobian
        self.save_jac = False

        return

    
    def set_transforms(self,grid,decomp):
        # Save the grid transforms including overlaps
        self.xi_x_EX  = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
        self.eta_x_EX = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
        self.xi_y_EX  = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
        self.eta_y_EX = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
        self.xi_lap_coeff  = torch.zeros((decomp.nx_,decomp.ny_,1),dtype=self.WP).to(self.device)
        self.eta_lap_coeff = torch.zeros((decomp.nx_,decomp.ny_,1),dtype=self.WP).to(self.device)
        self.cross_lap_coeff = torch.zeros((decomp.nx_,decomp.ny_,1),dtype=self.WP).to(self.device)
        self.xi_x_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_x
        self.eta_x_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_x
        self.xi_y_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_y
        self.eta_y_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_y
        decomp.communicate_border_2D( self.xi_x_EX  )
        decomp.communicate_border_2D( self.eta_x_EX )
        decomp.communicate_border_2D( self.xi_y_EX  )
        decomp.communicate_border_2D( self.eta_y_EX )

        # Euler fluxes: Determinant of the Jacobian matrix of the inverse grid transformation
        # Already have these in Grid.py, but recompute here to include the overlaps
        self.Jac       = torch.abs(self.xi_x_EX * self.eta_y_EX - self.xi_y_EX * self.eta_x_EX)
        self.inv_Jac   = 1.0 / self.Jac
        self.xi_x_Jac  = self.xi_x_EX * self.inv_Jac
        self.xi_y_Jac  = self.xi_y_EX * self.inv_Jac
        self.eta_x_Jac = self.eta_x_EX * self.inv_Jac
        self.eta_y_Jac = self.eta_y_EX * self.inv_Jac

        if grid.have_midpoint_transforms:
            # Interpolated transforms for Euler fluxes - needed also outside domain boundaries
            # x-midpoints
            self.xi_x_xm_EX  = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.eta_x_xm_EX = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.xi_y_xm_EX  = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.eta_y_xm_EX = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.xi_x_xm_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_x_xm
            self.eta_x_xm_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_x_xm
            self.xi_y_xm_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_y_xm
            self.eta_y_xm_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_y_xm
            decomp.communicate_border_2D( self.xi_x_xm_EX  )
            decomp.communicate_border_2D( self.eta_x_xm_EX )
            decomp.communicate_border_2D( self.xi_y_xm_EX  )
            decomp.communicate_border_2D( self.eta_y_xm_EX )
            
            # y-midpoints
            self.xi_x_ym_EX  = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.eta_x_ym_EX = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.xi_y_ym_EX  = torch.zeros((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.eta_y_ym_EX = torch.ones ((decomp.nxo_,decomp.nyo_,1),dtype=self.WP).to(self.device)
            self.xi_x_ym_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_x_ym
            self.eta_x_ym_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_x_ym
            self.xi_y_ym_EX [self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.xi_y_ym
            self.eta_y_ym_EX[self.imin_:self.imax_,self.jmin_:self.jmax_,0] = grid.eta_y_ym
            decomp.communicate_border_2D( self.xi_x_ym_EX  )
            decomp.communicate_border_2D( self.eta_x_ym_EX )
            decomp.communicate_border_2D( self.xi_y_ym_EX  )
            decomp.communicate_border_2D( self.eta_y_ym_EX )
        
            self.Jac_xm = torch.abs(self.xi_x_xm_EX * self.eta_y_xm_EX - self.xi_y_xm_EX * self.eta_x_xm_EX)
            self.Jac_ym = torch.abs(self.xi_x_ym_EX * self.eta_y_ym_EX - self.xi_y_ym_EX * self.eta_x_ym_EX)
            self.inv_Jac_xm = 1.0 / self.Jac_xm
            self.inv_Jac_ym = 1.0 / self.Jac_ym

            # Needed for Steger-Warming fluxes
            self.x_eta_xm_EX = -self.xi_y_xm_EX * self.inv_Jac_xm
            self.y_eta_xm_EX =  self.xi_x_xm_EX * self.inv_Jac_xm

            self.x_xi_ym_EX =  self.eta_y_ym_EX * self.inv_Jac_ym
            self.y_xi_ym_EX = -self.eta_x_ym_EX * self.inv_Jac_ym
            
        else:
            self.inv_Jac_xm = self.inv_Jac
            self.inv_Jac_ym = self.inv_Jac

        self.have_transforms = True

        # For Steger-Warming fluxes
        # 
        shape = [4, 4,
                 self.imax_-self.imin_+1,
                 self.jmax_-self.jmin_,
                 self.kmax_-self.kmin_]
        self.Ap = torch.zeros(shape, dtype=self.WP).to(self.device)
        self.Am = torch.zeros(shape, dtype=self.WP).to(self.device)
        
        shape = [4, 4,
                 self.imax_-self.imin_,
                 self.jmax_-self.jmin_+1,
                 self.kmax_-self.kmin_]
        self.Bp = torch.zeros(shape, dtype=self.WP).to(self.device)
        self.Bm = torch.zeros(shape, dtype=self.WP).to(self.device)
        
        # Cell-face normal vectors
        # X
        self.norm_xx = self.xi_x_xm_EX[self.imin_-1:self.imax_,self.jmin_:self.jmax_,:].clone()
        self.norm_xy = self.xi_y_xm_EX[self.imin_-1:self.imax_,self.jmin_:self.jmax_,:].clone()
        norm_m_x = torch.sqrt(self.norm_xx**2 + self.norm_xy**2)
        self.norm_xx /= norm_m_x
        self.norm_xy /= norm_m_x
        # Y
        self.norm_yx = self.eta_x_ym_EX[self.imin_:self.imax_,self.jmin_-1:self.jmax_,:].clone()
        self.norm_yy = self.eta_y_ym_EX[self.imin_:self.imax_,self.jmin_-1:self.jmax_,:].clone()
        norm_m_y = torch.sqrt(self.norm_yx**2 + self.norm_yy**2)
        self.norm_yx /= norm_m_y
        self.norm_yy /= norm_m_y

        # Cell-face areas
        self.Sx = torch.sqrt(self.x_eta_xm_EX[self.imin_-1:self.imax_,self.jmin_:self.jmax_,:]**2 +
                             self.y_eta_xm_EX[self.imin_-1:self.imax_,self.jmin_:self.jmax_,:]**2 )
        
        self.Sy = torch.sqrt(self.x_xi_ym_EX[self.imin_:self.imax_,self.jmin_-1:self.jmax_,:]**2 +
                             self.y_xi_ym_EX[self.imin_:self.imax_,self.jmin_-1:self.jmax_,:]**2 )
        

        # These are only needed on the interior
        self.xi_xx, _ = self.grad_node(self.xi_x_EX, compute_dy=False, force_2D=True)[:2]
        self.eta_xx, _ = self.grad_node(self.eta_x_EX, compute_dy=False, force_2D=True)[:2]
        _, self.xi_yy = self.grad_node(self.xi_y_EX, compute_dx=False, force_2D=True)[:2]
        _, self.eta_yy = self.grad_node(self.eta_y_EX, compute_dx=False, force_2D=True)[:2]
        xi_lap_coeff = self.xi_x_EX ** 2 + self.xi_y_EX ** 2
        eta_lap_coeff = self.eta_x_EX ** 2 + self.eta_y_EX ** 2
        cross_lap_coeff = 2 * (self.xi_x_EX * self.eta_x_EX + self.xi_y_EX * self.eta_y_EX)
        self.xi_lap_coeff[:, :, 0] = xi_lap_coeff[self.imin_:self.imax_,self.jmin_:self.jmax_,0]
        self.eta_lap_coeff[:, :, 0] = eta_lap_coeff[self.imin_:self.imax_,self.jmin_:self.jmax_,0]
        self.cross_lap_coeff[:, :, 0] = cross_lap_coeff[self.imin_:self.imax_,self.jmin_:self.jmax_,0]
        return


    def expand_overlaps(self,u):
        # Expands the overlaps of computed derivatives at non-periodic
        # boundaries to the extended-interior overlaps. Used when a
        # first derivative will be reused to compute a second
        # derivative.
        if ((not self.periodic_xi) and (self.nx > 1)):
            if (self.iproc==0):
                nx,ny,nz = u.shape
                u = torch.cat( (torch.zeros((self.noveri,ny,nz),
                                            dtype=self.WP).to(self.device), u), dim=0 )
                    
            if (self.iproc==self.npx-1):
                nx,ny,nz = u.shape
                u = torch.cat( (u, torch.zeros((self.noveri,ny,nz),
                                               dtype=self.WP).to(self.device)), dim=0 )
                
        if (not self.periodic_eta):
            if (self.jproc==0):
                nx,ny,nz = u.shape
                u = torch.cat( (torch.zeros((nx,self.noveri,nz),
                                            dtype=self.WP).to(self.device), u), dim=1 )
                
            if (self.jproc==self.npy-1):
                nx,ny,nz = u.shape
                u = torch.cat( (u, torch.zeros((nx,self.noveri,nz),
                                               dtype=self.WP).to(self.device)), dim=1 )

        return u


    def full2ext(self,u):
        # Input:  u with full overlaps      (nxo_,nyo_,nzo_)
        # Output: view to extended interior (nxi_,nyi_,nzi_)
        return u[self.imini_:self.imaxi_,
                 self.jmini_:self.jmaxi_,
                 self.kmini_:self.kmaxi_]
    
    def full2int(self,u):
        # Input:  u with full overlaps  (nxo_,nyo_,nzo_)
        # Output: view to true interior (nx_,ny_,nz_)
        return u[self.imin_:self.imax_,
                 self.jmin_:self.jmax_,
                 self.kmin_:self.kmax_]

    def ext2int(self,u):
        # Input:  u with extended interior (nxi_,nyi_,nzi_)
        # Output: view to true interior    (nx_,ny_,nz_)
        if (self.nx > 1):
            imin_ = self.noveri; imax_ = -self.noveri
        else:
            imin_ = 0; imax_ = 1
        if (self.nz > 1):
            kmin_ = self.noveri; kmax_ = -self.noveri
        else:
            kmin_ = 0; kmax_ = 1
        return u[imin_:imax_,
                 self.noveri:-self.noveri,
                 kmin_:kmax_]

    
    def enforce_neumann(self,u_eta):
        # Enforce homogeneous Neumann conditions
        if (self.BC_eta_bot=='wall' and self.jproc==0):
            u_eta[:,0,:] = 0.0
        if (self.BC_eta_top=='wall' and self.jproc==self.npy-1):
            u_eta[:,-1,:] = 0.0
                
        return u_eta
    

    #@profile
    def grad_node( self,
                   u,
                   compute_dx = True,
                   compute_dy = True,
                   compute_dz = True,
                   Neumann = False,
                   compute_extended = False,
                   extended_input = False,
                   force_2D = False ):

        # Get indices
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = self.kmin_
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = self.kmax_
        #nx_ = self.nx_
        #ny_ = self.ny_
        #nz_ = self.nz_
        
        if (compute_extended):
            # Compute derivatives on extended interior
            # Input u has full overlaps (nxo_,nyo_,nzo_)
            imin_ = self.imini_; jmin_ = self.jmini_; kmin_ = self.kmini_
            imax_ = self.imaxi_; jmax_ = self.jmaxi_; kmax_ = self.kmaxi_
            #nx_ = self.nxi_
            #ny_ = self.nyi_
            #nz_ = self.nzi_

            # Edge cases: truncate the extended interior for
            # non-periodic boundaries
            if (not self.periodic_xi):
                if (self.iproc==0):          imin_ = self.imin_
                if (self.iproc==self.npx-1): imax_ = self.imax_
            if (not self.periodic_eta):
                if (self.jproc==0):          jmin_ = self.jmin_
                if (self.jproc==self.npy-1): jmax_ = self.jmax_
                
        elif (extended_input):
            # Compute derivatives on true interior
            # Input u has extended interior (nxi_,nyi_,nzi_)
            # e.g., for second derivatives
            imin_ = self.noveri; imax_ = -self.noveri
            jmin_ = self.noveri; jmax_ = -self.noveri
            kmin_ = self.noveri; kmax_ = -self.noveri
            if (self.nx==1):
                imin_ = 0; imax_ = 1
            if (self.nz==1):
                kmin_ = 0; kmax_ = 1

        elif (force_2D):
            kmin_ = 0; kmax_ = 1

        # Indices for grid transforms
        if (compute_extended):
            # Will be applied after expanding edge cases, so use full
            # extended interior
            imin_g = self.imini_; jmin_g = self.jmini_
            imax_g = self.imaxi_; jmax_g = self.jmaxi_
        else:
            # True interior only
            imin_g = self.imin_; jmin_g = self.jmin_
            imax_g = self.imax_; jmax_g = self.jmax_

        #
        # NEED TO ACCOUNT FOR EDGE CASES!!
        #
        #u_xi  = torch.zeros((nx_,ny_,nz_), dtype=self.WP).to(self.device)
        #u_tmp = torch.zeros((nx_,ny_,nz_), dtype=self.WP).to(self.device)
        #u_eta = torch.zeros((nx_,ny_,nz_), dtype=self.WP).to(self.device)
            
        if (compute_dx or compute_dy):
            # Xi
            if (self.nx > 1):
                if (not self.periodic_xi and (self.iproc==0 or self.iproc==self.npx-1)):
                    if (self.iproc==0 and self.npx>1):
                        # Left non-periodic boundary
                        u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
                        u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
                        u2   = u[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_]
                        u3m1 = u[imin_+3:imax_+1,jmin_:jmax_,kmin_:kmax_]
                        u1m3 = u[imin_+1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                        u4m0 = u[imin_+4:imax_+2,jmin_:jmax_,kmin_:kmax_]
                        u0m4 = u[imin_  :imax_-2,jmin_:jmax_,kmin_:kmax_]
                        u_xi = torch.cat(( (u1 - u0)/self.d_xi,
                                           (u2 - u0)/(2.0*self.d_xi),
                                           ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_xi ), dim=0)
                    elif (self.iproc==self.npx-1 and self.npx>1):
                        # Right non-periodic boundary
                        u3m1 = u[imin_+1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                        u1m3 = u[imin_-1:imax_-3,jmin_:jmax_,kmin_:kmax_]
                        u4m0 = u[imin_+2:imax_  ,jmin_:jmax_,kmin_:kmax_]
                        u0m4 = u[imin_-2:imax_-4,jmin_:jmax_,kmin_:kmax_]
                        um3  = u[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_]
                        um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                        um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]
                        u_xi = torch.cat(( ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_xi,
                                           (um1 - um3)/(2.0*self.d_xi),
                                           (um1 - um2)/self.d_xi ), dim=0)
                    else:
                        # Non-periodic xi and npx=1
                        u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
                        u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
                        u2   = u[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_]
                        u3m1 = u[imin_+3:imax_-1,jmin_:jmax_,kmin_:kmax_]
                        u1m3 = u[imin_+1:imax_-3,jmin_:jmax_,kmin_:kmax_]
                        u4m0 = u[imin_+4:imax_  ,jmin_:jmax_,kmin_:kmax_]
                        u0m4 = u[imin_  :imax_-4,jmin_:jmax_,kmin_:kmax_]
                        um3  = u[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_]
                        um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                        um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]
                        u_xi = torch.cat(( (u1 - u0)/self.d_xi,
                                           (u2 - u0)/(2.0*self.d_xi),
                                           ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_xi,
                                           (um1 - um3)/(2.0*self.d_xi),
                                           (um1 - um2)/self.d_xi ), dim=0)
                else:
                    ul  = u[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    ull = u[imin_-2:imax_-2,jmin_:jmax_,kmin_:kmax_]

                    ur  = u[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
                    urr = u[imin_+2:imax_+2,jmin_:jmax_,kmin_:kmax_]

                    u_xi  = ( 2.0*(ur - ul)/3.0 - 
                              (urr - ull)/12.0 )/self.d_xi
            else:
                u_xi = 0.0

            # Eta
            if (not self.periodic_eta and (self.jproc==0 or self.jproc==self.npy-1)):
                if (self.jproc==0 and self.npy>1):
                    # Bottom non-periodic boundary
                    u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
                    u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
                    u2   = u[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_]
                    u3m1 = u[imin_:imax_,jmin_+3:jmax_+1,kmin_:kmax_]
                    u1m3 = u[imin_:imax_,jmin_+1:jmax_-1,kmin_:kmax_]
                    u4m0 = u[imin_:imax_,jmin_+4:jmax_+2,kmin_:kmax_]
                    u0m4 = u[imin_:imax_,jmin_  :jmax_-2,kmin_:kmax_]
                    u_eta = torch.cat(( (u1 - u0)/self.d_eta,
                                        (u2 - u0)/(2.0*self.d_eta),
                                        ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_eta ), dim=1)
                elif (self.jproc==self.npy-1 and self.npy>1):
                    # Top non-periodic boundary
                    u3m1 = u[imin_:imax_,jmin_+1:jmax_-1,kmin_:kmax_]
                    u1m3 = u[imin_:imax_,jmin_-1:jmax_-3,kmin_:kmax_]
                    u4m0 = u[imin_:imax_,jmin_+2:jmax_  ,kmin_:kmax_]
                    u0m4 = u[imin_:imax_,jmin_-2:jmax_-4,kmin_:kmax_]
                    um3  = u[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_]
                    um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
                    um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
                    u_eta = torch.cat(( ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_eta,
                                        (um1 - um3)/(2.0*self.d_eta),
                                        (um1 - um2)/self.d_eta ), dim=1)
                else:
                    # Non-periodic eta and npy=1
                    u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
                    u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
                    u2   = u[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_]
                    u3m1 = u[imin_:imax_,jmin_+3:jmax_-1,kmin_:kmax_]
                    u1m3 = u[imin_:imax_,jmin_+1:jmax_-3,kmin_:kmax_]
                    u4m0 = u[imin_:imax_,jmin_+4:jmax_  ,kmin_:kmax_]
                    u0m4 = u[imin_:imax_,jmin_  :jmax_-4,kmin_:kmax_]
                    um3  = u[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_]
                    um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
                    um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
                    u_eta = torch.cat(( (u1 - u0)/self.d_eta,
                                        (u2 - u0)/(2.0*self.d_eta),
                                        ( 2.0*(u3m1 - u1m3)/3.0 - (u4m0 - u0m4)/12.0 )/self.d_eta,
                                        (um1 - um3)/(2.0*self.d_eta),
                                        (um1 - um2)/self.d_eta ), dim=1)
            else:
                ul  = u[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
                ull = u[imin_:imax_,jmin_-2:jmax_-2,kmin_:kmax_]

                ur  = u[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
                urr = u[imin_:imax_,jmin_+2:jmax_+2,kmin_:kmax_]

                u_eta  = ( 2.0*(ur - ul)/3.0 - 
                           (urr - ull)/12.0 )/self.d_eta

            # Enforce Neumann boundary conditions
            if (Neumann):
                u_eta = self.enforce_neumann(u_eta)

            # Extend overlaps to full extended interior
            if (compute_extended):
                if (self.nx > 1): u_xi = self.expand_overlaps(u_xi)
                u_eta = self.expand_overlaps(u_eta)

            # Compute du/dx, du/dy
            if (self.nx > 1 and compute_dx):
                if self.have_transforms:
                    du_dx = ( self.xi_x_EX [imin_g:imax_g,jmin_g:jmax_g,:] * u_xi  +
                              self.eta_x_EX[imin_g:imax_g,jmin_g:jmax_g,:] * u_eta )
                else:
                    du_dx = u_xi
            else:
                du_dx = None
            if compute_dy:
                if self.have_transforms:
                    du_dy = ( self.xi_y_EX [imin_g:imax_g,jmin_g:jmax_g,:] * u_xi  +
                              self.eta_y_EX[imin_g:imax_g,jmin_g:jmax_g,:] * u_eta )
                else:
                    du_dy = u_eta
            else:
                du_dy = None

        else:
            # Not computing dx or dy
            du_dx = None
            du_dy = None
            
        # Z: periodic, rectilinear
        if (self.nz > 1 and compute_dz):
            ul  = u[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1]
            ull = u[imin_:imax_,jmin_:jmax_,kmin_-2:kmax_-2]
            
            ur  = u[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1]
            urr = u[imin_:imax_,jmin_:jmax_,kmin_+2:kmax_+2]

            du_dz  = ( 2.0*(ur - ul)/3.0 - 
                       (urr - ull)/12.0 )/self.d_z

            if (compute_extended):
                du_dz = self.expand_overlaps(du_dz)

            # # relese memory
            # del ul; del ull; del ur; del urr;

        else:
            du_dz = None

        return du_dx,du_dy,du_dz


    def Euler_Fluxes( self,
                      rho,
                      rhoU,
                      rhoV,
                      rhoW,
                      rhoE,
                      EOS ):

        # Expand the conserved quantities at boundaries
        #  --> TODO: 3D
        rho_EX  = rho [self.imin_b-1:self.imax_b+2,self.jmin_b-1:self.jmax_b+2,:]
        rhoU_EX = rhoU[self.imin_b-1:self.imax_b+2,self.jmin_b-1:self.jmax_b+2,:]
        rhoV_EX = rhoV[self.imin_b-1:self.imax_b+2,self.jmin_b-1:self.jmax_b+2,:]
        rhoW_EX = rhoW[self.imin_b-1:self.imax_b+2,self.jmin_b-1:self.jmax_b+2,:]
        rhoE_EX = rhoE[self.imin_b-1:self.imax_b+2,self.jmin_b-1:self.jmax_b+2,:]
        ##  --> TODO: Scalars
        var_list = [rho_EX,
                    rhoU_EX,
                    rhoV_EX,
                    rhoW_EX,
                    rhoE_EX]

        MUSCL_flux = False
        
        # Periodic boundaries and domain interiors are already handled by overlaps
        imin = jmin = kmin =  2
        if MUSCL_flux:
            imax = jmax = kmax = -1
        else:
            imax = jmax = kmax = -2
            
        if False: #(not self.periodic_xi):
            if self.iproc==0:
                # Left non-periodic boundary
                for var in var_list:
                    var[imin-1,:,:] = var[imin,:,:]
            if self.iproc==self.npx-1:
                # Right non-periodic boundary
                for var in var_list:
                    var[-2:,:,:] = var[-3:-2,:,:]
                    
        if False:# (not self.periodic_eta):
            if self.jproc==0:
                # Bottom non-periodic boundary
                #for var in var_list:
                #    var[:,jmin-1,:] = var[:,jmin,:]  # Upwind: not actually accessing imin-1/jmin-1 at walls
                    
                # Adiabatic walls: de/dn = 0
                rhoKE_EX = 0.5*(rhoU_EX**2 + rhoV_EX**2 + rhoW_EX**2)/rho_EX
                e_EX = (rhoE_EX - rhoKE_EX)/rho_EX
                e_EX[:,:jmin,:] = e_EX[:,jmin:jmin+1,:]
                
                # Reset rho*E
                rhoE_EX = rho_EX * e_EX + rhoKE_EX
                
            #if self.jproc==self.npy-1:
            #    # Top non-periodic boundary
            #    for var in var_list:
            #        var[:,-2:,:] = var[:,-3:-2,:]

        # Inviscid fluxes - x
        if MUSCL_flux:
            q_im1 = (rho_EX [imin-2:imax-2,jmin:jmax-1,:],
                     rhoU_EX[imin-2:imax-2,jmin:jmax-1,:],
                     rhoV_EX[imin-2:imax-2,jmin:jmax-1,:],
                     rhoW_EX[imin-2:imax-2,jmin:jmax-1,:],
                     rhoE_EX[imin-2:imax-2,jmin:jmax-1,:])
            q_i   = (rho_EX [imin-1:imax-1,jmin:jmax-1,:],
                     rhoU_EX[imin-1:imax-1,jmin:jmax-1,:],
                     rhoV_EX[imin-1:imax-1,jmin:jmax-1,:],
                     rhoW_EX[imin-1:imax-1,jmin:jmax-1,:],
                     rhoE_EX[imin-1:imax-1,jmin:jmax-1,:])
            q_ip1 = (rho_EX [imin  :imax  ,jmin:jmax-1,:],
                     rhoU_EX[imin  :imax  ,jmin:jmax-1,:],
                     rhoV_EX[imin  :imax  ,jmin:jmax-1,:],
                     rhoW_EX[imin  :imax  ,jmin:jmax-1,:],
                     rhoE_EX[imin  :imax  ,jmin:jmax-1,:])
            q_ip2 = (rho_EX [imin+1:,jmin:jmax-1,:],
                     rhoU_EX[imin+1:,jmin:jmax-1,:],
                     rhoV_EX[imin+1:,jmin:jmax-1,:],
                     rhoW_EX[imin+1:,jmin:jmax-1,:],
                     rhoE_EX[imin+1:,jmin:jmax-1,:])

            fx = self.MUSCL_HLLE_Flux('x', q_im1, q_i, q_ip1, q_ip2, EOS)

        else:
            q_im1 = (rho_EX [imin-1:imax-1,jmin:jmax,:],
                     rhoU_EX[imin-1:imax-1,jmin:jmax,:],
                     rhoV_EX[imin-1:imax-1,jmin:jmax,:],
                     rhoW_EX[imin-1:imax-1,jmin:jmax,:],
                     rhoE_EX[imin-1:imax-1,jmin:jmax,:])
            q_i   = (rho_EX [imin  :imax  ,jmin:jmax,:],
                     rhoU_EX[imin  :imax  ,jmin:jmax,:],
                     rhoV_EX[imin  :imax  ,jmin:jmax,:],
                     rhoW_EX[imin  :imax  ,jmin:jmax,:],
                     rhoE_EX[imin  :imax  ,jmin:jmax,:])
            q_ip1 = (rho_EX [imin+1:imax+1,jmin:jmax,:],
                     rhoU_EX[imin+1:imax+1,jmin:jmax,:],
                     rhoV_EX[imin+1:imax+1,jmin:jmax,:],
                     rhoW_EX[imin+1:imax+1,jmin:jmax,:],
                     rhoE_EX[imin+1:imax+1,jmin:jmax,:])
            
            div_fx = self.Upwind_1st_Flux('x', q_im1, q_i, q_ip1, EOS)
            
        # Inviscid fluxes - y
        if MUSCL_flux:
            q_im1 = (rho_EX [imin:imax-1,jmin-2:jmax-2,:],
                     rhoU_EX[imin:imax-1,jmin-2:jmax-2,:],
                     rhoV_EX[imin:imax-1,jmin-2:jmax-2,:],
                     rhoW_EX[imin:imax-1,jmin-2:jmax-2,:],
                     rhoE_EX[imin:imax-1,jmin-2:jmax-2,:])
            q_i   = (rho_EX [imin:imax-1,jmin-1:jmax-1,:],
                     rhoU_EX[imin:imax-1,jmin-1:jmax-1,:],
                     rhoV_EX[imin:imax-1,jmin-1:jmax-1,:],
                     rhoW_EX[imin:imax-1,jmin-1:jmax-1,:],
                     rhoE_EX[imin:imax-1,jmin-1:jmax-1,:])
            q_ip1 = (rho_EX [imin:imax-1,jmin  :jmax  ,:],
                     rhoU_EX[imin:imax-1,jmin  :jmax  ,:],
                     rhoV_EX[imin:imax-1,jmin  :jmax  ,:],
                     rhoW_EX[imin:imax-1,jmin  :jmax  ,:],
                     rhoE_EX[imin:imax-1,jmin  :jmax  ,:])
            q_ip2 = (rho_EX [imin:imax-1,jmin+1:,:],
                     rhoU_EX[imin:imax-1,jmin+1:,:],
                     rhoV_EX[imin:imax-1,jmin+1:,:],
                     rhoW_EX[imin:imax-1,jmin+1:,:],
                     rhoE_EX[imin:imax-1,jmin+1:,:])

            fy = self.MUSCL_HLLE_Flux('y', q_im1, q_i, q_ip1, q_ip2, EOS)
        else:
            q_im1 = (rho_EX [imin:imax,jmin-1:jmax-1,:],
                     rhoU_EX[imin:imax,jmin-1:jmax-1,:],
                     rhoV_EX[imin:imax,jmin-1:jmax-1,:],
                     rhoW_EX[imin:imax,jmin-1:jmax-1,:],
                     rhoE_EX[imin:imax,jmin-1:jmax-1,:])
            q_i   = (rho_EX [imin:imax,jmin  :jmax  ,:],
                     rhoU_EX[imin:imax,jmin  :jmax  ,:],
                     rhoV_EX[imin:imax,jmin  :jmax  ,:],
                     rhoW_EX[imin:imax,jmin  :jmax  ,:],
                     rhoE_EX[imin:imax,jmin  :jmax  ,:])
            q_ip1 = (rho_EX [imin:imax,jmin+1:jmax+1,:],
                     rhoU_EX[imin:imax,jmin+1:jmax+1,:],
                     rhoV_EX[imin:imax,jmin+1:jmax+1,:],
                     rhoW_EX[imin:imax,jmin+1:jmax+1,:],
                     rhoE_EX[imin:imax,jmin+1:jmax+1,:])
            
            div_fy = self.Upwind_1st_Flux('y', q_im1, q_i, q_ip1, EOS)

        # Inviscid fluxes - z
        if MUSCL_flux:
            fz = []
        else:
            div_fz = []
            # TODO

        # Finalize divergences and coordinate transform
        if MUSCL_flux:
            div_fx = []
            div_fy = []
            div_fz = []
            for fxi in fx[:-1]: ## FIX - ignoring species
                div_fx.append(self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:] / self.d_xi *
                              (fxi[1:,:,:] - fxi[:-1,:,:]))
            for fyi in fy[:-1]:
                div_fy.append(self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:] / self.d_eta *
                              (fyi[:,1:,:] - fyi[:,:-1,:]))
        
        return div_fx, div_fy, div_fz
    

    # 1st-order upwind flux reconstruction
    # Returns fluxes at i
    #
    def Upwind_1st_Flux( self,
                         ndir,
                         q_im1,
                         q_i,
                         q_ip1,
                         EOS ):
        
        # Left state
        q1_im1 = q_im1[0] # rho
        q2_im1 = q_im1[1] # rho*U
        q3_im1 = q_im1[2] # rho*V
        q4_im1 = q_im1[3] # rho*W
        q5_im1 = q_im1[4] # rho*E

        # Central state
        q1_i = q_i[0]
        q2_i = q_i[1]
        q3_i = q_i[2]
        q4_i = q_i[3]
        q5_i = q_i[4]

        # Right state
        q1_ip1 = q_ip1[0]
        q2_ip1 = q_ip1[1]
        q3_ip1 = q_ip1[2]
        q4_ip1 = q_ip1[3]
        q5_ip1 = q_ip1[4]

        # Species partial densities
        qY_im1 = []
        qY_i   = []
        qY_ip1 = []
        for isc in range(EOS.num_sc):
            qY_im1.append( q_im1[5+isc] )
            qY_i.append(   q_i  [5+isc] )
            qY_ip1.append( q_ip1[5+isc] )

        # Primitive variables
        #   x-velocity
        u_im1 = q2_im1/q1_im1
        u_i   = q2_i  /q1_i
        u_ip1 = q2_ip1/q1_ip1
        #
        #   y-velocity
        v_im1 = q3_im1/q1_im1
        v_i   = q3_i  /q1_i
        v_ip1 = q3_ip1/q1_ip1
        #
        #   z-velocity
        w_im1 = q4_im1/q1_im1
        w_i   = q4_i  /q1_i
        w_ip1 = q4_ip1/q1_ip1
        #
        #   internal energy
        e_im1 = q5_im1/q1_im1 - 0.5*(u_im1**2 + v_im1**2 + w_im1**2)
        e_i   = q5_i  /q1_i   - 0.5*(u_i**2   + v_i**2   + w_i**2)
        e_ip1 = q5_ip1/q1_ip1 - 0.5*(u_ip1**2 + v_ip1**2 + w_ip1**2)
        
        # Secondary variables
        # Temperature
        T_i   = EOS.get_T_internal_energy(e_i)
        #
        # Sound speeds
        c_i   = EOS.get_soundspeed_T(T_i)
        #
        # Pressure
        p_im1 = EOS.get_P_rho_internal_energy(q1_im1, e_im1)
        p_i   = EOS.get_P_rho_internal_energy(q1_i,   e_i)
        p_ip1 = EOS.get_P_rho_internal_energy(q1_ip1, e_ip1)
        
        # Transformed velocity in computational coordinates
        if (ndir=='x'):
            trans_zm1 = 0.0
            trans_z   = 0.0
            trans_zp1 = 0.0
            
            trans_xm1 = self.xi_x_Jac[self.imin_-1:self.imax_-1,self.jmin_:self.jmax_,:]
            trans_ym1 = self.xi_y_Jac[self.imin_-1:self.imax_-1,self.jmin_:self.jmax_,:]
            trans_x   = self.xi_x_Jac[self.imin_  :self.imax_,  self.jmin_:self.jmax_,:]
            trans_y   = self.xi_y_Jac[self.imin_  :self.imax_,  self.jmin_:self.jmax_,:]
            trans_xp1 = self.xi_x_Jac[self.imin_+1:self.imax_+1,self.jmin_:self.jmax_,:]
            trans_yp1 = self.xi_y_Jac[self.imin_+1:self.imax_+1,self.jmin_:self.jmax_,:]

            qq_im1 = trans_xm1 * u_im1 + trans_ym1 * v_im1
            qq_i   = trans_x   * u_i   + trans_y   * v_i  
            qq_ip1 = trans_xp1 * u_ip1 + trans_yp1 * v_ip1

            d_xi_j = self.d_xi
            
        elif (ndir=='y'):
            trans_zm1 = 0.0
            trans_z   = 0.0
            trans_zp1 = 0.0
            
            trans_xm1 = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_-1:self.jmax_-1,:]
            trans_ym1 = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_-1:self.jmax_-1,:]
            trans_x   = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_  :self.jmax_,  :]
            trans_y   = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_  :self.jmax_,  :]
            trans_xp1 = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_+1:self.jmax_+1,:]
            trans_yp1 = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_+1:self.jmax_+1,:]

            qq_im1 = trans_xm1 * u_im1 + trans_ym1 * v_im1
            qq_i   = trans_x   * u_i   + trans_y   * v_i  
            qq_ip1 = trans_xp1 * u_ip1 + trans_yp1 * v_ip1

            d_xi_j = self.d_eta
            
        elif (ndir=='z'):  ## UPDATE for 3D
            trans_xm1 = 0.0
            trans_ym1 = 0.0
            trans_zm1 = 1.0
            trans_x   = 0.0
            trans_y   = 0.0
            trans_z   = 1.0
            trans_xp1 = 0.0
            trans_yp1 = 0.0
            trans_zp1 = 1.0

            qq_im1 = w_im1
            qq_i   = w_i
            qq_ip1 = w_ip1

            d_xi_j = self.d_z
            

        # Upwind limiters - using transformed advective velocity
        c_trans = c_i * torch.sqrt(trans_x**2 + trans_y**2 + trans_z**2)
        if ndir=='x':
            q_avg   = (qq_im1 + qq_i + qq_ip1)/3.0
        else:
            q_avg = qq_i
        wL = q_avg - c_trans
        wR = q_avg + c_trans
        SL = torch.minimum(torch.sign(wL), torch.zeros_like(wL))
        SR = torch.maximum(torch.sign(wR), torch.zeros_like(wR))
        ## q_avg makes post-shock sol smoother

        sum_S = SR - SL
        SL /= sum_S
        SR /= sum_S
        # Non-periodic boundaries
        if (ndir=='y' and not self.periodic_eta):
            if (self.jproc==0):
                SL[:,0,:] = -1.0
                SR[:,0,:] =  0.0
            if (self.jproc==self.npy-1):
                SL[:,-1,:] = 0.0
                SR[:,-1,:] = 1.0
        
        # Fluxes
        # q1 - rho
        F_im1 = q1_im1 * qq_im1
        F_i   = q1_i   * qq_i  
        F_ip1 = q1_ip1 * qq_ip1
        div_f1 = Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j \
            * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:]

        # q2 - rho*U
        # F_im1 = (q2_im1 * qq_im1 + p_im1 * trans_xm1 * EOS.P_fac)
        # F_i   = (q2_i   * qq_i   + p_i   * trans_x   * EOS.P_fac)
        # F_ip1 = (q2_ip1 * qq_ip1 + p_ip1 * trans_xp1 * EOS.P_fac)
        F_im1 = (p_im1 * trans_xm1 * EOS.P_fac)
        F_i   = (p_i   * trans_x   * EOS.P_fac)
        F_ip1 = (p_ip1 * trans_xp1 * EOS.P_fac)

        div_f2 = Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j  \
            * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:]

        # q3 - rho*V
        # F_im1 = (q3_im1 * qq_im1 + p_im1 * trans_ym1 * EOS.P_fac)
        # F_i   = (q3_i   * qq_i   + p_i   * trans_y   * EOS.P_fac)
        # F_ip1 = (q3_ip1 * qq_ip1 + p_ip1 * trans_yp1 * EOS.P_fac)
        F_im1 = (p_im1 * trans_ym1 * EOS.P_fac)
        F_i   = (p_i   * trans_y   * EOS.P_fac)
        F_ip1 = (p_ip1 * trans_yp1 * EOS.P_fac)

        div_f3 = Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j \
            * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:]

        # q4 - rho*W
        F_im1 = (q4_im1 * qq_im1 + p_im1 * trans_zm1 * EOS.P_fac)
        F_i   = (q4_i   * qq_i   + p_i   * trans_z   * EOS.P_fac)
        F_ip1 = (q4_ip1 * qq_ip1 + p_ip1 * trans_zp1 * EOS.P_fac)
        div_f4 = Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j \
            * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:]

        # q5 - rho*E
        # F_im1 = (q5_im1 + p_im1 * EOS.P_fac) * qq_im1
        # F_i   = (q5_i   + p_i   * EOS.P_fac) * qq_i  
        # F_ip1 = (q5_ip1 + p_ip1 * EOS.P_fac) * qq_ip1
        F_im1 = (p_im1 * EOS.P_fac) * qq_im1
        F_i   = (p_i   * EOS.P_fac) * qq_i  
        F_ip1 = (p_ip1 * EOS.P_fac) * qq_ip1

        div_f5 = Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j \
            * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:]

        # rho*Y
        div_fSC = []
        for isc in range(EOS.num_sc):
            F_im1 = qY_im1 * qq_im1
            F_i   = qY_i   * qq_i  
            F_ip1 = qY_ip1 * qq_ip1
            div_fSC.append(Upwind(F_im1, F_i, F_ip1, SL, SR) / d_xi_j \
                           * self.Jac[self.imin_:self.imax_,self.jmin_:self.jmax_,:])

        return div_f1, div_f2, div_f3, div_f4, div_f5, div_fSC
    

    # MUSCL HLLE flux reconstruction
    # Returns fluxes at i+1/2
    #
    def MUSCL_HLLE_Flux( self,
                         ndir,
                         q_im1,
                         q_i,
                         q_ip1,
                         q_ip2,
                         EOS ):
        # Left state
        q1_im1 = q_im1[0] # rho
        q2_im1 = q_im1[1] # rho*U
        q3_im1 = q_im1[2] # rho*V
        q4_im1 = q_im1[3] # rho*W
        q5_im1 = q_im1[4] # rho*E

        # Central state
        q1_i = q_i[0]
        q2_i = q_i[1]
        q3_i = q_i[2]
        q4_i = q_i[3]
        q5_i = q_i[4]

        # Right state - 1
        q1_ip1 = q_ip1[0]
        q2_ip1 = q_ip1[1]
        q3_ip1 = q_ip1[2]
        q4_ip1 = q_ip1[3]
        q5_ip1 = q_ip1[4]

        # Right state - 2
        q1_ip2 = q_ip2[0]
        q2_ip2 = q_ip2[1]
        q3_ip2 = q_ip2[2]
        q4_ip2 = q_ip2[3]
        q5_ip2 = q_ip2[4]

        # Species partial densities
        Y_im1 = []
        Y_i   = []
        Y_ip1 = []
        Y_ip2 = []
        for isc in range(EOS.num_sc):
            Y_im1.append( q_im1[5+isc] )
            Y_i.append(   q_i  [5+isc] )
            Y_ip1.append( q_ip1[5+isc] )
            Y_ip2.append( q_ip2[5+isc] )

        # Primitive variables
        #   x-velocity
        u_im1 = q2_im1/q1_im1
        u_i   = q2_i  /q1_i
        u_ip1 = q2_ip1/q1_ip1
        u_ip2 = q2_ip2/q1_ip2
        #
        #   y-velocity
        v_im1 = q3_im1/q1_im1
        v_i   = q3_i  /q1_i
        v_ip1 = q3_ip1/q1_ip1
        v_ip2 = q3_ip2/q1_ip2
        #
        #   z-velocity
        w_im1 = q4_im1/q1_im1
        w_i   = q4_i  /q1_i
        w_ip1 = q4_ip1/q1_ip1
        w_ip2 = q4_ip2/q1_ip2
        #
        #   internal energy
        e_im1 = q5_im1/q1_im1 - 0.5*(u_im1**2 + v_im1**2 + w_im1**2)
        e_i   = q5_i  /q1_i   - 0.5*(u_i**2   + v_i**2   + w_i**2)
        e_ip1 = q5_ip1/q1_ip1 - 0.5*(u_ip1**2 + v_ip1**2 + w_ip1**2)
        e_ip2 = q5_ip2/q1_ip2 - 0.5*(u_ip2**2 + v_ip2**2 + w_ip2**2)

        # Flattening factors -- FINISH
        xiL = 0.5
        xiR = 0.5

        #print(q1_im1.shape, q1_i.shape, q1_ip1.shape, q1_ip2.shape)

        # MUSCL reconstruction of primitive variables
        if (EOS.num_sc > 0):
            rhoL, rhoR, YL, YR = get_MUSCL_LR(q1_im1, q1_i, q1_ip1, q1_ip2, xiL, xiR,
                                              species=(Y_im1,  Y_i,  Y_ip1,  Y_ip2))
            # Convert partial densities to mass fractions
            for isc in range(EOS.num_sc):
                YL[isc] = YL[isc] / rhoL
                YR[isc] = YR[isc] / rhoR
        else:
            rhoL, rhoR = get_MUSCL_LR(q1_im1, q1_i, q1_ip1, q1_ip2, xiL, xiR)
            YL = None; YR = None

        uL, uR = get_MUSCL_LR(u_im1,  u_i,  u_ip1,  u_ip2, xiL, xiR)
        vL, vR = get_MUSCL_LR(v_im1,  v_i,  v_ip1,  v_ip2, xiL, xiR)
        wL, wR = get_MUSCL_LR(w_im1,  w_i,  w_ip1,  w_ip2, xiL, xiR)
        eL, eR = get_MUSCL_LR(e_im1,  e_i,  e_ip1,  e_ip2, xiL, xiR) # reconstructing e instead of p
        
        # Secondary variables
        # Temperature
        TL = EOS.get_T_internal_energy(eL) # TODO -- general interface to EOS.get_T()
        TR = EOS.get_T_internal_energy(eR)
        #
        # Pressure
        pL = EOS.get_P_rho_internal_energy(rhoL, eL) # TODO -- general itnerface to EOS
        pR = EOS.get_P_rho_internal_energy(rhoR, eR)

        # TEMPERATURE LIMITER -- NEED TO IMPLEMENT? even if reconstrucing e vs p?


        # Total enthalpy
        HL = eL + 0.5*(uL*uL + vL*vL + wL*wL) + pL/rhoL
        HR = eR + 0.5*(uR*uR + vR*vR + wR*wR) + pR/rhoR

        # Specific heat ratios (gamma)
        gL = EOS.get_gamma_TY(TL, YL)
        gR = EOS.get_gamma_TY(TL, YL)

        # HLLE approximate Riemann solver
        # Roe-averaged variables
        sqrt_rhoL = torch.sqrt(rhoL)
        sqrt_rhoR = torch.sqrt(rhoR)
        iden_Roe  = 1.0 / (sqrt_rhoL + sqrt_rhoR)
        rho_Roe = iden_Roe * (sqrt_rhoL + sqrt_rhoR) * torch.sqrt(rhoL * rhoR)
        u_Roe   = iden_Roe * (sqrt_rhoL * uL + sqrt_rhoR * uR)
        v_Roe   = iden_Roe * (sqrt_rhoL * vL + sqrt_rhoR * vR)
        w_Roe   = iden_Roe * (sqrt_rhoL * wL + sqrt_rhoR * wR)
        e_Roe   = iden_Roe * (sqrt_rhoL * eL + sqrt_rhoR * eR)
        #H_Roe   = iden_Roe * (sqrt_rhoL * HL + sqrt_rhoR * HR)
        p_Roe   = iden_Roe * (sqrt_rhoL * pL + sqrt_rhoR * pR) # eliminate for multispecies?
        g_Roe   = iden_Roe * (sqrt_rhoL * gL + sqrt_rhoR * gR) # eliminate for multispecies?

        # Speed of sound
        #cL    = torch.sqrt(gL * pL / rhoL)
        #cR    = torch.sqrt(gR * pR / rhoR)
        #c_Roe = torch.sqrt(g_Roe * p_Roe / rho_Roe)
        
        cL = EOS.get_soundspeed_T(TL)
        cR = EOS.get_soundspeed_T(TR)
        c_Roe = EOS.get_soundspeed_T(EOS.get_T_internal_energy(e_Roe))
        
        # Transformed velocity in computational coordinates
        if (ndir=='x'):
            trans_zL = 0.0
            trans_zR = 0.0
            
            #trans_xm1 = self.xi_x_Jac[self.imin_b-1:self.imax_b-1,self.jmin_:self.jmax_,:]
            #trans_ym1 = self.xi_y_Jac[self.imin_b-1:self.imax_b-1,self.jmin_:self.jmax_,:]
            trans_x   = self.xi_x_Jac[self.imin_b  :self.imax_b,  self.jmin_:self.jmax_,:]
            trans_y   = self.xi_y_Jac[self.imin_b  :self.imax_b,  self.jmin_:self.jmax_,:]
            trans_xp1 = self.xi_x_Jac[self.imin_b+1:self.imax_b+1,self.jmin_:self.jmax_,:]
            trans_yp1 = self.xi_y_Jac[self.imin_b+1:self.imax_b+1,self.jmin_:self.jmax_,:]
            #trans_xp2 = self.xi_x_Jac[self.imin_b+2:self.imax_b+2,self.jmin_:self.jmax_,:]
            #trans_yp2 = self.xi_y_Jac[self.imin_b+2:self.imax_b+2,self.jmin_:self.jmax_,:]
            
            qL = trans_x   * uL + trans_y   * vL
            qR = trans_xp1 * uR + trans_yp1 * vR
            trans_x_Roe = iden_Roe * (sqrt_rhoL * trans_x + sqrt_rhoR * trans_xp1)
            trans_y_Roe = iden_Roe * (sqrt_rhoL * trans_y + sqrt_rhoR * trans_yp1)
            q_Roe = trans_x_Roe * u_Roe + trans_y_Roe * v_Roe

            #q_im1 = trans_xm1 * u_im1 + trans_ym1 * v_im1
            #q_i   = trans_x   * u_i   + trans_y   * v_i  
            #q_ip1 = trans_xp1 * u_ip1 + trans_yp1 * v_ip1
            #q_ip2 = trans_xp2 * u_ip2 + trans_yp2 * v_ip2

            #qL, qR = get_MUSCL_LR(q_im1,  q_i,  q_ip1,  q_ip2, xiL, xiR)
            #q_Roe = iden_Roe * (sqrt_rhoL * qL + sqrt_rhoR * qR)

            # global monotonicity hard-force means all LR values are at i+1/2!
            #trans_x = trans_xp1 = self.xi_x_xm_Jac[self.imin_b:self.imax_b,self.jmin_:self.jmax_,  :]
            #trans_y = trans_yp1 = self.xi_y_xm_Jac[self.imin_b:self.imax_b,self.jmin_:self.jmax_,  :]
            
            inv_Jac = self.inv_Jac_xm[self.imin_b:self.imax_b,self.jmin_:self.jmax_,:]
            #inv_Jac = self.inv_Jac[self.imin_b:self.imax_b,self.jmin_:self.jmax_,:]
            
        elif (ndir=='y'):
            trans_zL = 0.0
            trans_zR = 0.0
            
            #trans_xm1 = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_b-1:self.jmax_b-1,:]
            #trans_ym1 = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_b-1:self.jmax_b-1,:]
            trans_x   = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_b  :self.jmax_b,  :]
            trans_y   = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_b  :self.jmax_b,  :]
            trans_xp1 = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_b+1:self.jmax_b+1,:]
            trans_yp1 = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_b+1:self.jmax_b+1,:]
            #trans_xp2 = self.eta_x_Jac[self.imin_:self.imax_,self.jmin_b+2:self.jmax_b+2,:]
            #trans_yp2 = self.eta_y_Jac[self.imin_:self.imax_,self.jmin_b+2:self.jmax_b+2,:]
            
            qL = trans_x   * uL + trans_y   * vL
            qR = trans_xp1 * uR + trans_yp1 * vR
            trans_x_Roe = iden_Roe * (sqrt_rhoL * trans_x + sqrt_rhoR * trans_xp1)
            trans_y_Roe = iden_Roe * (sqrt_rhoL * trans_y + sqrt_rhoR * trans_yp1)
            q_Roe = trans_x_Roe * u_Roe + trans_y_Roe * v_Roe

            #q_im1 = trans_xm1 * u_im1 + trans_ym1 * v_im1
            #q_i   = trans_x   * u_i   + trans_y   * v_i  
            #q_ip1 = trans_xp1 * u_ip1 + trans_yp1 * v_ip1
            #q_ip2 = trans_xp2 * u_ip2 + trans_yp2 * v_ip2

            #qL, qR = get_MUSCL_LR(q_im1,  q_i,  q_ip1,  q_ip2, xiL, xiR)
            #q_Roe = iden_Roe * (sqrt_rhoL * qL + sqrt_rhoR * qR)

            #trans_x = trans_xp1 = self.eta_x_ym_Jac[self.imin_:self.imax_,self.jmin_b  :self.jmax_b,  :]
            #trans_y = trans_yp1 = self.eta_y_ym_Jac[self.imin_:self.imax_,self.jmin_b  :self.jmax_b,  :]
            
            inv_Jac = self.inv_Jac_ym[self.imin_:self.imax_,self.jmin_b:self.jmax_b,:]
            #inv_Jac = self.inv_Jac[self.imin_:self.imax_,self.jmin_b:self.jmax_b,:]
            
        elif (ndir=='z'):
            trans_xL = 0.0
            trans_yL = 0.0
            trans_xR = 0.0
            trans_yR = 0.0
            trans_zL = 1.0
            trans_zR = 1.0
            qL = wL
            qR = wR
            q_Roe = w_Roe
            

        # Wave speeds
        SL = torch.minimum(qL - cL, q_Roe - c_Roe)
        SR = torch.maximum(qR + cR, q_Roe + c_Roe)
        iden_S = 1.0 / (SR - SL)

        yy = 1
        #print(uL[127,yy,0], vL[127,yy,0], qL[127,yy,0], qR[127,yy,0], cL[127,yy,0])
        #print(trans_x[127,yy,0], trans_y[127,yy,0], inv_Jac[127,yy,0])

        ## ONLY got cylinder case working by setting to sqrt(1/Jac),
        ## using one part in qL,qR and the other (abs) in div_fx. This
        ## DOES NOT work for VGwavy. Why?? (Cylinder still blew up eventually.)
        
        # Fluxes
        # q1 - rho
        fL = rhoL * qL
        fR = rhoR * qR
        fS = (SR*fL - SL*fR + SL*SR*(rhoR - rhoL)*inv_Jac) * iden_S
        f1avg = HLLE(fL, fR, fS, SL, SR)

        # q2 - rho*U
        q2L = rhoL * uL
        q2R = rhoR * uR
        fL  = q2L * qL + pL * trans_x   * EOS.P_fac
        fR  = q2R * qR + pR * trans_xp1 * EOS.P_fac
        fS = (SR*fL - SL*fR + SL*SR*(q2R - q2L)*inv_Jac) * iden_S
        f2avg = HLLE(fL, fR, fS, SL, SR)

        # q3 - rho*V
        q3L = rhoL * vL
        q3R = rhoR * vR
        fL  = q3L * qL + pL * trans_y   * EOS.P_fac
        fR  = q3R * qR + pR * trans_yp1 * EOS.P_fac
        fS = (SR*fL - SL*fR + SL*SR*(q3R - q3L)*inv_Jac) * iden_S
        f3avg = HLLE(fL, fR, fS, SL, SR)

        # q4 - rho*W
        q4L = rhoL * wL
        q4R = rhoR * wR
        fL  = q4L * qL + pL * trans_zL * EOS.P_fac
        fR  = q4R * qR + pR * trans_zR * EOS.P_fac
        fS = (SR*fL - SL*fR + SL*SR*(q4R - q4L)*inv_Jac) * iden_S
        f4avg = HLLE(fL, fR, fS, SL, SR)

        # q5 - rho*E
        q5L = rhoL * HL - pL
        q5R = rhoR * HR - pR
        fL  = (q5L + pL * EOS.P_fac) * qL  # Modified for dimensional/dimensionless P
        fR  = (q5R + pR * EOS.P_fac) * qR
        fS = (SR*fL - SL*fR + SL*SR*(q5R - q5L)*inv_Jac) * iden_S
        f5avg = HLLE(fL, fR, fS, SL, SR)

        # rho*Y
        fSCavg = []
        for isc in range(EOS.num_sc):
            rhok_L = rhoL * YL[isc]
            rhok_R = rhoR * YR[isc]
            fL = rhok_L * qL
            fR = rhok_R * qR
            fS = (SR*fL - SL*fR + SL*SR*(rhok_R - rhok_L)*inv_Jac) * iden_S
            fSCavg.append(HLLE(fL, fR, fS, SL, SR))

        return f1avg, f2avg, f3avg, f4avg, f5avg, fSCavg
    

    def Steger_Warming_Fluxes( self,
                               rho_in,
                               rhoU,
                               rhoV,
                               rhoE,
                               EOS ):

        # Computing fluxes at "cell faces" (modified Steger-Warming)
        #
        #   WARNING: NOT MPI READY
        #   WARNING: ASSUMING 2D
        #
        
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = 0
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = 1

        # Primitives on interior
        # Include +1 point on each side for MPI overlaps and periodic BCs
        rho = rho_in[imin_-1:imax_+1,jmin_-1:jmax_+1,kmin_:kmax_]
        u = rhoU[imin_-1:imax_+1,jmin_-1:jmax_+1,kmin_:kmax_] / rho
        v = rhoV[imin_-1:imax_+1,jmin_-1:jmax_+1,kmin_:kmax_] / rho
        #w = rhoW/rho
        E = rhoE[imin_-1:imax_+1,jmin_-1:jmax_+1,kmin_:kmax_] / rho
        
        p = EOS.get_P_rho_internal_energy(rho, E - 0.5*(u*u + v*v))
        c = EOS.get_soundspeed_rp(rho, p)
        
        gamma = EOS.get_gamma_TY(None, None)  # Update for multispecies

        # Conserved variables
        Q = torch.stack((rho,
                         rhoU[imin_-1:imax_+1,jmin_-1:jmax_+1,:],
                         rhoV[imin_-1:imax_+1,jmin_-1:jmax_+1,:],
                         rhoE[imin_-1:imax_+1,jmin_-1:jmax_+1,:]), dim=0)

        # ================================================================================
        # x-split fluxes

        # Left/right states - perform weighted average for
        # - velocity components
        # - speed of sound
        sigma = 0.1 # maybe set to 0.5
        p_L = p[:-1,1:-1]
        p_R = p[1:,1:-1]
        th = sigma*(p_R - p_L)/torch.minimum(p_L, p_R)
        wt = 1.0 - 0.5/(th*th + 1.0)
        
        u_im_L, u_im_R = interp_im_weighted(u[:,1:-1,...], wt)
        v_im_L, v_im_R = interp_im_weighted(v[:,1:-1,...], wt)
        c_im_L, c_im_R = interp_im_weighted(c[:,1:-1,...], wt)
        
        up_L = ( self.norm_xx * u_im_L + self.norm_xy * v_im_L )
        up_R = ( self.norm_xx * u_im_R + self.norm_xy * v_im_R )

        # Left-split flux
        Fp = Euler_split_flux_Jac_2(u_im_L, v_im_L, c_im_L, up_L, gamma, self.norm_xx, self.norm_xy,
                                    Q[:, :-1, 1:-1], self.Ap, '+', self.save_jac)

        # Right-split flux
        Fm = Euler_split_flux_Jac_2(u_im_R, v_im_R, c_im_R, up_R, gamma, self.norm_xx, self.norm_xy,
                                    Q[:, 1:, 1:-1], self.Am, '-', self.save_jac)

        # Total flux
        F = (Fp + Fm) * self.Sx[None,...]

        # ================================================================================
        # y-split fluxes
        
        # Up/down states - perform weighted average for
        # - velocity components
        # - speed of sound
        p_L = p[1:-1,:-1]
        p_R = p[1:-1,1:]
        th = sigma*(p_R - p_L)/torch.minimum(p_L, p_R)
        wt = 1.0 - 0.5/(th*th + 1.0)
        
        u_jm_L, u_jm_R = interp_jm_weighted(u[1:-1], wt)
        v_jm_L, v_jm_R = interp_jm_weighted(v[1:-1], wt)
        c_jm_L, c_jm_R = interp_jm_weighted(c[1:-1], wt)
        
        vp_L = ( self.norm_yx * u_jm_L + self.norm_yy * v_jm_L )
        vp_R = ( self.norm_yx * u_jm_R + self.norm_yy * v_jm_R )

        # Left-split flux
        Gp = Euler_split_flux_Jac_2(u_jm_L, v_jm_L, c_jm_L, vp_L, gamma, self.norm_yx, self.norm_yy,
                                    Q[:, 1:-1, :-1], self.Bp, '+', self.save_jac)
        
        # Right-split flux
        Gm = Euler_split_flux_Jac_2(u_jm_R, v_jm_R, c_jm_R, vp_R, gamma, self.norm_yx, self.norm_yy,
                                    Q[:, 1:-1, 1:], self.Bm, '-', self.save_jac)
        
        # Total flux
        G = (Gp + Gm) * self.Sy[None,...]

        # ================================================================================
        # Dirichlet boundaries
        
        # X
        if not self.periodic_xi:
            if self.iproc==0:
                F[:,:1] *= 0.0
                self.Ap[:,:,:1] *= 0.0
                self.Am[:,:,:1] *= 0.0
            if self.iproc==self.npx-1:
                F[:,-1:] *= 0.0
                self.Ap[:,:,-1:] *= 0.0
                self.Am[:,:,-1:] *= 0.0

        # Y
        if not self.periodic_eta:
            if self.jproc==0:
                F[:,:,:1,...] *= 0.5
                G[:,:,:1,...] *= 0.0
                self.Bp[:,:,:,:1] *= 0.0
                self.Bm[:,:,:,:1] *= 0.0
            if self.jproc==self.npy-1:
                G[:,:,-1:,...] *= 0.0
                self.Bp[:,:,:,-1:] *= 0.0
                self.Bm[:,:,:,-1:] *= 0.0

        # ================================================================================
        # Flux divergence
        
        div_fx = (F[:,1:,...] - F[:,:-1,...]) / self.d_xi
        div_fy = (G[:,:,1:,...] - G[:,:,:-1,...]) / self.d_eta

        # Divide by the local cell volume
        div_fx *= self.Jac[None,self.imin_:self.imax_,self.jmin_:self.jmax_]
        div_fy *= self.Jac[None,self.imin_:self.imax_,self.jmin_:self.jmax_]
        
        return div_fx, div_fy


    def get_dplr_matrices(self, dt):
        # Assemble LHS matrices needed for DPLR time advancer
        #
        # NOTE: Assumes that self.Ap, Am, Bp, Bm are up-to-date
        # (i.e. RHS has been evaluated at current time level with metrics.save_jac = True)

        dt_ov_V = dt * self.Jac[None,None,self.imin_:self.imax_,self.jmin_:self.jmax_]

        self.Ap *= self.Sx[None,None,...] / self.d_xi
        self.Am *= self.Sx[None,None,...] / self.d_xi
        self.Bp *= self.Sy[None,None,...] / self.d_eta
        self.Bm *= self.Sy[None,None,...] / self.d_eta
        
        eye = torch.eye(4, dtype=self.WP).to(self.device)
        A_hat = ( eye[...,None,None,None] +
                  dt_ov_V * (self.Ap[:,:,1:] - self.Am[:,:,:-1] + self.Bp[:,:,:,1:] - self.Bm[:,:,:,:-1]) )

        B_hat = dt_ov_V * self.Bm[:,:,:,1:]

        C_hat = dt_ov_V * self.Bp[:,:,:,:-1]

        D_hat = dt_ov_V * self.Am[:,:,1:]

        E_hat = dt_ov_V * self.Ap[:,:,:-1]

        #### BOUNDARY CONDITIONS -- handled in Steger-Warming function

        #### Truncate Dirichlet (dQ=0) boundary values
        #### NOTE: Assumes homogeneous Dirichlet boundary values!
        mat_list = torch.stack((A_hat, B_hat, C_hat, D_hat, E_hat), dim=0)
        # X
        if not self.periodic_xi:
            if self.iproc==0:
                mat_list = mat_list[:,:,:,1:]
            if self.iproc==self.npx-1:
                mat_list = mat_list[:,:,:,:-1]
        # Y
        if not self.periodic_eta:
            if self.jproc==0:
                mat_list = mat_list[:,:,:,:,1:]
            if self.jproc==self.npy-1:
                mat_list = mat_list[:,:,:,:,:-1]

        A_hat = mat_list[0]
        B_hat = mat_list[1]
        C_hat = mat_list[2]
        D_hat = mat_list[3]
        E_hat = mat_list[4]

        ## NOTE 10/2: Density is not Dirichlet! Need to modify block
        ## matrices to include dQ_rho on LHS and all other dQs on RHS

        #### NEED VISCOUS JACOBIANS

        return A_hat, B_hat, C_hat, D_hat, E_hat
        
    
    def grad4_node( self,
                    u,
                    compute_extended=False,
                    extended_input=False ):
            
        # 4th derivatives for artificial diffusion
        #   Returns derivatives in the computational plane (does NOT apply grid Jacobian)
        
        # Get indices
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = self.kmin_
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = self.kmax_
        
        if (compute_extended):
            # Compute derivatives on extended interior
            # Input u has full overlaps (nxo_,nyo_,nzo_)
            imin_ = self.imini_; jmin_ = self.jmini_; kmin_ = self.kmini_
            imax_ = self.imaxi_; jmax_ = self.jmaxi_; kmax_ = self.kmaxi_

            # Edge cases: truncate the extended interior for
            # non-periodic boundaries
            if (not self.periodic_xi):
                if (self.iproc==0):          imin_ = self.imin_
                if (self.iproc==self.npx-1): imax_ = self.imax_
            if (not self.periodic_eta):
                if (self.jproc==0):          jmin_ = self.jmin_
                if (self.jproc==self.npy-1): jmax_ = self.jmax_
                
        elif (extended_input):
            # Compute derivatives on true interior
            # Input u has extended interior (nxi_,nyi_,nzi_)
            imin_ = self.noveri; imax_ = -self.noveri
            jmin_ = self.noveri; jmax_ = -self.noveri
            kmin_ = self.noveri; kmax_ = -self.noveri
            if (self.nz==1):
                kmin_ = 0; kmax_ = 1
        
        # Xi
        if (self.nx > 1):
            if (not self.periodic_xi and (self.iproc==0 or self.iproc==self.npx-1)):
                if (self.iproc==0 and self.npx>1):
                    # Left non-periodic boundary
                    u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
                    u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
                    u2   = u[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_]
                    u3   = u[imin_+3:imin_+4,jmin_:jmax_,kmin_:kmax_]
                    u4   = u[imin_+4:imin_+5,jmin_:jmax_,kmin_:kmax_]
                    u5   = u[imin_+5:imin_+6,jmin_:jmax_,kmin_:kmax_]
                    u0m6 = u[imin_  :imax_-3,jmin_:jmax_,kmin_:kmax_]
                    u1m5 = u[imin_+1:imax_-2,jmin_:jmax_,kmin_:kmax_]
                    u2m4 = u[imin_+2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    u3m3 = u[imin_+3:imax_  ,jmin_:jmax_,kmin_:kmax_]
                    u4m2 = u[imin_+4:imax_+1,jmin_:jmax_,kmin_:kmax_]
                    u5m1 = u[imin_+5:imax_+2,jmin_:jmax_,kmin_:kmax_]
                    u6m0 = u[imin_+6:imax_+3,jmin_:jmax_,kmin_:kmax_]
                    u_xi = torch.cat(( (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 0
                                       (u1 - 4.0*u2 + 6.0*u3 - 4.0*u4 + u5),      # 1
                                       (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 2
                                       ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                         (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ) ),
                                     dim=0) * self.d_xi4_i

                elif (self.iproc==self.npx-1 and self.npx>1):
                    # Right non-periodic boundary
                    u0m6 = u[imin_-3:imax_-6,jmin_:jmax_,kmin_:kmax_]
                    u1m5 = u[imin_-2:imax_-5,jmin_:jmax_,kmin_:kmax_]
                    u2m4 = u[imin_-1:imax_-4,jmin_:jmax_,kmin_:kmax_]
                    u3m3 = u[imin_  :imax_-3,jmin_:jmax_,kmin_:kmax_]
                    u4m2 = u[imin_+1:imax_-2,jmin_:jmax_,kmin_:kmax_]
                    u5m1 = u[imin_+2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    u6m0 = u[imin_+3:imax_  ,jmin_:jmax_,kmin_:kmax_]
                    um6  = u[imax_-6:imax_-5,jmin_:jmax_,kmin_:kmax_]
                    um5  = u[imax_-5:imax_-4,jmin_:jmax_,kmin_:kmax_]
                    um4  = u[imax_-4:imax_-3,jmin_:jmax_,kmin_:kmax_]
                    um3  = u[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_]
                    um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]
                    u_xi = torch.cat(( ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                         (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ),
                                       (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1),  # -3
                                       (um6 - 4.0*um5 + 6.0*um4 - 4.0*um3 + um2),  # -2
                                       (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1) ),# -1
                                     dim=0) * self.d_xi4_i

                else:
                    # Non-periodic xi and npx=1
                    u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
                    u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
                    u2   = u[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_]
                    u3   = u[imin_+3:imin_+4,jmin_:jmax_,kmin_:kmax_]
                    u4   = u[imin_+4:imin_+5,jmin_:jmax_,kmin_:kmax_]
                    u5   = u[imin_+5:imin_+6,jmin_:jmax_,kmin_:kmax_]
                    u0m6 = u[imin_  :imax_-6,jmin_:jmax_,kmin_:kmax_]
                    u1m5 = u[imin_+1:imax_-5,jmin_:jmax_,kmin_:kmax_]
                    u2m4 = u[imin_+2:imax_-4,jmin_:jmax_,kmin_:kmax_]
                    u3m3 = u[imin_+3:imax_-3,jmin_:jmax_,kmin_:kmax_]
                    u4m2 = u[imin_+4:imax_-2,jmin_:jmax_,kmin_:kmax_]
                    u5m1 = u[imin_+5:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    u6m0 = u[imin_+6:imax_  ,jmin_:jmax_,kmin_:kmax_]
                    um6  = u[imax_-6:imax_-5,jmin_:jmax_,kmin_:kmax_]
                    um5  = u[imax_-5:imax_-4,jmin_:jmax_,kmin_:kmax_]
                    um4  = u[imax_-4:imax_-3,jmin_:jmax_,kmin_:kmax_]
                    um3  = u[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_]
                    um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
                    um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]
                    u_xi = torch.cat(( (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 0
                                       (u1 - 4.0*u2 + 6.0*u3 - 4.0*u4 + u5),      # 1
                                       (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 2
                                       ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                         (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ),
                                       (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1),  # -3
                                       (um6 - 4.0*um5 + 6.0*um4 - 4.0*um3 + um2),  # -2
                                       (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1) ),# -1
                                     dim=0) * self.d_xi4_i

            else:
                # Interior only and/or periodic-xi
                ui  = u[imin_  :imax_  ,jmin_:jmax_,kmin_:kmax_]

                u1l = u[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
                u2l = u[imin_-2:imax_-2,jmin_:jmax_,kmin_:kmax_]
                u3l = u[imin_-3:imax_-3,jmin_:jmax_,kmin_:kmax_]

                u1r = u[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
                u2r = u[imin_+2:imax_+2,jmin_:jmax_,kmin_:kmax_]
                u3r = u[imin_+3:imax_+3,jmin_:jmax_,kmin_:kmax_]

                u_xi  = ( 2.0*(u2r - 4.0*u1r + 6.0*ui - 4.0*u1l + u2l) - 
                          (u3r - 9.0*u1r + 16.0*ui - 9.0*u1l + u3l)/6.0 ) * self.d_xi4_i
        else:
            u_xi = 0.0

        # Eta
        if (not self.periodic_eta and (self.jproc==0 or self.jproc==self.npy-1)):
            if (self.jproc==0 and self.npy>1):
                # Bottom non-periodic boundary
                u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
                u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
                u2   = u[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_]
                u3   = u[imin_:imax_,jmin_+3:jmin_+4,kmin_:kmax_]
                u4   = u[imin_:imax_,jmin_+4:jmin_+5,kmin_:kmax_]
                u5   = u[imin_:imax_,jmin_+5:jmin_+6,kmin_:kmax_]
                u0m6 = u[imin_:imax_,jmin_  :jmax_-3,kmin_:kmax_]
                u1m5 = u[imin_:imax_,jmin_+1:jmax_-2,kmin_:kmax_]
                u2m4 = u[imin_:imax_,jmin_+2:jmax_-1,kmin_:kmax_]
                u3m3 = u[imin_:imax_,jmin_+3:jmax_  ,kmin_:kmax_]
                u4m2 = u[imin_:imax_,jmin_+4:jmax_+1,kmin_:kmax_]
                u5m1 = u[imin_:imax_,jmin_+5:jmax_+2,kmin_:kmax_]
                u6m0 = u[imin_:imax_,jmin_+6:jmax_+3,kmin_:kmax_]
                u_eta = torch.cat(( (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 0
                                    (u1 - 4.0*u2 + 6.0*u3 - 4.0*u4 + u5),      # 1
                                    (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 2
                                    ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                      (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ) ),
                                  dim=1) * self.d_eta4_i
                
            elif (self.jproc==self.npy-1 and self.npy>1):
                # Top non-periodic boundary
                u0m6 = u[imin_:imax_,jmin_-3:jmax_-6,kmin_:kmax_]
                u1m5 = u[imin_:imax_,jmin_-2:jmax_-5,kmin_:kmax_]
                u2m4 = u[imin_:imax_,jmin_-1:jmax_-4,kmin_:kmax_]
                u3m3 = u[imin_:imax_,jmin_  :jmax_-3,kmin_:kmax_]
                u4m2 = u[imin_:imax_,jmin_+1:jmax_-2,kmin_:kmax_]
                u5m1 = u[imin_:imax_,jmin_+2:jmax_-1,kmin_:kmax_]
                u6m0 = u[imin_:imax_,jmin_+3:jmax_  ,kmin_:kmax_]
                um6  = u[imin_:imax_,jmax_-6:jmax_-5,kmin_:kmax_]
                um5  = u[imin_:imax_,jmax_-5:jmax_-4,kmin_:kmax_]
                um4  = u[imin_:imax_,jmax_-4:jmax_-3,kmin_:kmax_]
                um3  = u[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_]
                um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
                um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
                u_eta = torch.cat(( ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                      (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ),
                                    (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1),  # -3
                                    (um6 - 4.0*um5 + 6.0*um4 - 4.0*um3 + um2),  # -2
                                    (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1) ),# -1
                                  dim=1) * self.d_eta4_i
                
            else:
                # Non-periodic eta and npy=1
                u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
                u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
                u2   = u[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_]
                u3   = u[imin_:imax_,jmin_+3:jmin_+4,kmin_:kmax_]
                u4   = u[imin_:imax_,jmin_+4:jmin_+5,kmin_:kmax_]
                u5   = u[imin_:imax_,jmin_+5:jmin_+6,kmin_:kmax_]
                u0m6 = u[imin_:imax_,jmin_  :jmax_-6,kmin_:kmax_]
                u1m5 = u[imin_:imax_,jmin_+1:jmax_-5,kmin_:kmax_]
                u2m4 = u[imin_:imax_,jmin_+2:jmax_-4,kmin_:kmax_]
                u3m3 = u[imin_:imax_,jmin_+3:jmax_-3,kmin_:kmax_]
                u4m2 = u[imin_:imax_,jmin_+4:jmax_-2,kmin_:kmax_]
                u5m1 = u[imin_:imax_,jmin_+5:jmax_-1,kmin_:kmax_]
                u6m0 = u[imin_:imax_,jmin_+6:jmax_  ,kmin_:kmax_]
                um6  = u[imin_:imax_,jmax_-6:jmax_-5,kmin_:kmax_]
                um5  = u[imin_:imax_,jmax_-5:jmax_-4,kmin_:kmax_]
                um4  = u[imin_:imax_,jmax_-4:jmax_-3,kmin_:kmax_]
                um3  = u[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_]
                um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
                um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
                u_eta = torch.cat(( (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 0
                                    (u1 - 4.0*u2 + 6.0*u3 - 4.0*u4 + u5),      # 1
                                    (u0 - 4.0*u1 + 6.0*u2 - 4.0*u3 + u4),      # 2
                                    ( 2.0*(u1m5 - 4.0*u2m4 + 6.0*u3m3 - 4.0*u4m2 + u5m1)  -  # 3:-3
                                      (u0m6 - 9.0*u2m4 + 16.0*u3m3 - 9.0*u4m2 + u6m0)/6.0 ),
                                    (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1),  # -3
                                    (um6 - 4.0*um5 + 6.0*um4 - 4.0*um3 + um2),  # -2
                                    (um5 - 4.0*um4 + 6.0*um3 - 4.0*um2 + um1) ),# -1
                                  dim=1) * self.d_eta4_i
                
        else:
            # Interior only and/or periodic-eta
            ui  = u[imin_:imax_,jmin_  :jmax_  ,kmin_:kmax_]
            
            u1l = u[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
            u2l = u[imin_:imax_,jmin_-2:jmax_-2,kmin_:kmax_]
            u3l = u[imin_:imax_,jmin_-3:jmax_-3,kmin_:kmax_]
            
            u1r = u[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
            u2r = u[imin_:imax_,jmin_+2:jmax_+2,kmin_:kmax_]
            u3r = u[imin_:imax_,jmin_+3:jmax_+3,kmin_:kmax_]

            u_eta  = ( 2.0*(u2r - 4.0*u1r + 6.0*ui - 4.0*u1l + u2l) - 
                       (u3r - 9.0*u1r + 16.0*ui - 9.0*u1l + u3l)/6.0 ) * self.d_eta4_i

        # Extend overlaps to full extended interior
        if (compute_extended):
            u_xi  = self.expand_overlaps(u_xi)
            u_eta = self.expand_overlaps(u_eta)
            
        # Z: periodic, rectilinear
        if (self.nz > 1):
            ui  = u[imin_:imax_,jmin_:jmax_,kmin_  :kmax_  ]
            
            u1l = u[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1]
            u2l = u[imin_:imax_,jmin_:jmax_,kmin_-2:kmax_-2]
            u3l = u[imin_:imax_,jmin_:jmax_,kmin_-3:kmax_-3]
            
            u1r = u[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1]
            u2r = u[imin_:imax_,jmin_:jmax_,kmin_+2:kmax_+2]
            u3r = u[imin_:imax_,jmin_:jmax_,kmin_+3:kmax_+3]

            u_z  = ( 2.0*(u2r - 4.0*u1r + 6.0*ui - 4.0*u1l + u2l) - 
                     (u3r - 9.0*u1r + 16.0*ui - 9.0*u1l + u3l)/6.0 ) * self.d_z4_i
        else:
            u_z = None

        return u_xi,u_eta,u_z

    def lap(self, u):
        # TURN HAVE TRANSFORMS OFF SO WE GET COMP PLANE DERIVS!
        self.have_transforms = False
        # Move these to init eventually
        u_xi, u_eta = self.grad_node(u, compute_extended=True)[:2]
        _, u_xieta = self.grad_node(u_xi, compute_dx=False, extended_input=True)[:2] # Do this with proper stencil!!!
        self.have_transforms = True
        # TURN HAVE TRANSFORMS BACK ON SO WE DON'T BREAK ANYTHING!
        u_xi = self.ext2int(u_xi)
        u_eta = self.ext2int(u_eta)

        u_xixi, u_etaeta, _ = self.grad2cc_node(u)
        lap = (
                self.xi_lap_coeff[:, :, :] * u_xixi
                + self.eta_lap_coeff[:, :, :] * u_etaeta
                + self.cross_lap_coeff[:, :, :] * u_xieta
                + (self.xi_xx[:, :, :] + self.xi_yy[:, :, :]) * u_xi
                + (self.eta_xx[:, :, :] + self.eta_yy[:, :, :]) * u_eta
        )

        return lap

    def grad2cc_node(self, u):
        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_

        # Xi
        ull = u[imin_-2:imax_-2, jmin_:jmax_, kmin_:kmax_]
        ul = u[imin_-1:imax_-1, jmin_:jmax_, kmin_:kmax_]
        uc = u[imin_  :imax_  , jmin_:jmax_, kmin_:kmax_]
        ur = u[imin_+1:imax_+1, jmin_:jmax_, kmin_:kmax_]
        urr = u[imin_+2:imax_+2, jmin_:jmax_, kmin_:kmax_]
        u_xixi = (-2.5 * uc + (4/3) * (ul + ur) - (1/12) * (ull + urr)) / self.d_xi**2

        # Eta
        if self.jproc == 0 or self.jproc == self.npy-1:
            if self.jproc == 0 and self.npy > 1:
                u0 = u[imin_:imax_, jmin_:jmin_+1, kmin_:kmax_]
                u1 = u[imin_:imax_, jmin_+1:jmin_+2, kmin_:kmax_]
                u2 = u[imin_:imax_, jmin_+2:jmin_+3, kmin_:kmax_]

                u0m4 = u[imin_:imax_, jmin_:jmax_-2, kmin_:kmax_]
                u1m3 = u[imin_:imax_, jmin_+1:jmax_-1, kmin_:kmax_]
                uc = u[imin_:imax_, jmin_+2:jmax_, kmin_:kmax_]
                u3m1 = u[imin_:imax_, jmin_+3:jmax_+1, kmin_:kmax_]
                u4m0 = u[imin_:imax_, jmin_+4:jmax_+2, kmin_:kmax_]

                u_etaeta = torch.cat((
                    (u0 - 2 * u1 + u2) / self.d_eta**2,
                    (u0 - 2 * u1 + u2) / self.d_eta**2,
                    (-2.5 * uc + (4/3) * (u3m1 + u1m3) - (1/12) * (u4m0 + u0m4)) / self.d_eta**2
                ), dim=1)
            elif self.jproc == self.npy - 1 and self.npy > 1:
                # Top non-periodic boundary
                u0m4 = u[imin_:imax_, jmin_-2:jmax_-4, kmin_:kmax_]
                u1m3 = u[imin_:imax_, jmin_-1:jmax_-3, kmin_:kmax_]
                uc = u[imin_:imax_, jmin_:jmax_-2, kmin_:kmax_]
                u3m1 = u[imin_:imax_, jmin_+1:jmax_-1, kmin_:kmax_]
                u4m0 = u[imin_:imax_, jmin_+2:jmax_, kmin_:kmax_]

                um3 = u[imin_:imax_, jmax_-3:jmax_-2, kmin_:kmax_]
                um2 = u[imin_:imax_, jmax_-2:jmax_-1, kmin_:kmax_]
                um1 = u[imin_:imax_, jmax_-1:jmax_, kmin_:kmax_]

                u_etaeta = torch.cat((
                    (-2.5 * uc + (4/3) * (u3m1 + u1m3) - (1/12) * (u4m0 + u0m4)) / self.d_eta ** 2,
                    (um1 - 2 * um2 + um3) / self.d_eta ** 2,
                    (um1 - 2 * um2 + um3) / self.d_eta ** 2), dim=1)
            else:
                # Non-periodic eta and npy=1
                u0 = u[imin_:imax_, jmin_:jmin_+1, kmin_:kmax_]
                u1 = u[imin_:imax_, jmin_+1:jmin_+2, kmin_:kmax_]
                u2 = u[imin_:imax_, jmin_+2:jmin_+3, kmin_:kmax_]

                u0m4 = u[imin_:imax_, jmin_:jmax_-4, kmin_:kmax_]
                u1m3 = u[imin_:imax_, jmin_+1:jmax_-3, kmin_:kmax_]
                uc = u[imin_:imax_, jmin_+2:jmax_-2, kmin_:kmax_]
                u3m1 = u[imin_:imax_, jmin_+3:jmax_-1, kmin_:kmax_]
                u4m0 = u[imin_:imax_, jmin_+4:jmax_, kmin_:kmax_]

                um3 = u[imin_:imax_, jmax_-3:jmax_-2, kmin_:kmax_]
                um2 = u[imin_:imax_, jmax_-2:jmax_-1, kmin_:kmax_]
                um1 = u[imin_:imax_, jmax_-1:jmax_, kmin_:kmax_]

                u_etaeta = torch.cat((
                    (u0 - 2 * u1 + u2) / self.d_eta**2,
                    (u0 - 2 * u1 + u2) / self.d_eta**2,
                    (-2.5 * uc + (4/3) * (u3m1 + u1m3) - (1/12) * (u4m0 + u0m4)) / self.d_eta**2,
                    (um1 - 2 * um2 + um3) / self.d_eta**2,
                    (um1 - 2 * um2 + um3) / self.d_eta**2), dim=1)
        else:
            # Interior scheme
            u0m4 = u[imin_:imax_, jmin_-2:jmax_-2, kmin_:kmax_]
            u1m3 = u[imin_:imax_, jmin_-1:jmax_-1, kmin_:kmax_]
            uc = u[imin_:imax_, jmin_:jmax_, kmin_:kmax_]
            u3m1 = u[imin_:imax_, jmin_+1:jmax_+1, kmin_:kmax_]
            u4m0 = u[imin_:imax_, jmin_+2:jmax_+2, kmin_:kmax_]

            u_etaeta = (-2.5 * uc + (4 / 3) * (u3m1 + u1m3) - (1 / 12) * (u4m0 + u0m4)) / self.d_eta ** 2

        # Z
        u_zz = None
        if self.nz > 1:
            ull = u[imin_:imax_, jmin_:jmax_, kmin_-2:kmax_-2]
            ul = u[imin_:imax_, jmin_:jmax_, kmin_-1:kmax_-1]
            uc = u[imin_:imax_, jmin_:jmax_, kmin_:kmax_]
            ur = u[imin_:imax_, jmin_:jmax_, kmin_+1:kmax_+1]
            urr = u[imin_:imax_, jmin_:jmax_, kmin_+2:kmax_+2]
            u_zz = (-2.5 * uc + (4/3) * (ul + ur) - (1/12) * (ull + urr)) / self.d_z**2

        return u_xixi, u_etaeta, u_zz

