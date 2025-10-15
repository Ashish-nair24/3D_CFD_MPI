"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Operator.py

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

from .Grid import get_deltas_weighted
from .Library import Tridiagonal


# Constants for truncated Gaussian filter (Cook & Cabot JCP 2005)
C0 = 3565.0/10368.0
C1 = 3091.0/12960.0
C2 = 1997.0/25920.0
C3 =  149.0/12960.0
C4 = 107.0/103680.0


# ------------------------------------------------------
# 4th-order artificial diffusivity (Kawai & Lele JCP 2008)
#   Strain-rate based for bulk viscosity
#   See paper for species diffusivities
# ------------------------------------------------------
def art_diff4_S(rho,S,T,e,cs,grid,metrics):
    # Model constants
    C_mu    = 0.002
    C_beta  = 10.0
    C_kappa = 0.01
    C_D     = 0.01
    C_Y     = 100.0

    # Compute derivatives
    S_xi,S_eta = metrics.grad4_node(S) # update with dilatation-based diff
    e_xi,e_eta = metrics.grad4_node(e)

    # Grid transforms
    xi_term   = (grid.xi_x**2  + grid.xi_y**2 )**2
    eta_term  = (grid.eta_x**2 + grid.eta_y**2)**2

    # Viscosity
    filt_term = torch.abs( xi_term[:,:,None]  * S_xi  * grid.delta_xi[:,:,None]**6 +
                           eta_term[:,:,None] * S_eta * grid.delta_eta[:,:,None]**6 )
    mu = C_mu * gauss_filter( rho * filt_term, grid )

    beta = mu * C_beta/C_mu

    # Thermal conductivity
    filt_term = torch.abs( xi_term[:,:,None]  * e_xi  * grid.delta_xi[:,:,None]**5 +
                           eta_term[:,:,None] * e_eta * grid.delta_eta[:,:,None]**5 )
    kappa = C_kappa * gauss_filter( rho*cs/T * filt_term, grid )

    return mu,beta,kappa


## Need to make the following changes to enable adjoint calculation for art_diff:
#  Need to extend:
#     Grid.get_deltas_weighted
#     metrics.grad4_node - DONE
#     grid.delta_xi, grid.delta_eta
#     gauss_filter - just change imin to imini, etc.
#     might need to hardcode noveri in metrics -- need 2 pt ext interior + 4 (3?) pts for filter

    
# ------------------------------------------------------
# 4th-order artificial diffusivity (Kawai, Shankar, Lele JCP 2010)
#    Dilatation-based for bulk viscosity
#    For two-dimensional RHS
# ------------------------------------------------------
def art_diff4_D_2D(rho,drho_dx,drho_dy,S,tmp_grad,
                   div_u,curl_u,T,e,cs,grid,metrics,device):
    # Model constants
    C_mu    = 0.002
    C_beta  = 1.75
    C_kappa = 0.01

    # Computational grid spacing
    d_xi4  = grid.d_xi**4
    d_eta4 = grid.d_eta**4

    # Physical grid spacing
    drho_dx_int = metrics.ext2int( drho_dx )
    drho_dy_int = metrics.ext2int( drho_dy )
    drho_mag = vec_mag_2D(drho_dx_int, drho_dy_int) + 1e-15
    delta_xi_beta2,delta_eta_beta2 = \
        get_deltas_weighted(grid,drho_dx_int/drho_mag,drho_dy_int/drho_mag)

    de_dx,de_dy = metrics.grad_node( e, Neumann=True )[:2]
    de_mag = vec_mag_2D(de_dx, de_dy) + 1e-15
    delta_xi_kappa2,delta_eta_kappa2 = \
        get_deltas_weighted(grid,de_dx/de_mag,de_dy/de_mag)
    
    # Compute 4th derivatives
    S_int     = metrics.ext2int( S )
    div_u_int = metrics.ext2int( div_u )
    tmp_grad.copy( S_int );     S_xi,S_eta     = metrics.grad4_node(tmp_grad.var)[:2]
    tmp_grad.copy( div_u_int ); dil_xi,dil_eta = metrics.grad4_node(tmp_grad.var)[:2]
    e_xi,e_eta = metrics.grad4_node( e )[:2]

    # Shock sensor
    #fsw = 1.0 # LAD-D0
    # LAD-D2
    div_u2 = div_u_int**2
    curl_u_int = metrics.ext2int( curl_u )
    fsw = torch.heaviside(-div_u_int,torch.tensor([0.5], dtype=metrics.WP).to(device)) * div_u2 / \
        ( div_u2 + curl_u_int**2 + 1e-32 )

    # Mu
    filt_term = torch.abs( S_xi  * d_xi4  * grid.delta_xi[:,:,None]**2 +
                           S_eta * d_eta4 * grid.delta_eta[:,:,None]**2 ) * rho
    tmp_grad.copy( filt_term )
    mu = C_mu * gauss_filter( tmp_grad, metrics )

    # Beta
    filt_term = torch.abs( dil_xi  * d_xi4  * delta_xi_beta2 +
                           dil_eta * d_eta4 * delta_eta_beta2 ) * rho * fsw
    tmp_grad.copy( filt_term )
    beta = C_beta * gauss_filter( tmp_grad, metrics )

    # Kappa
    filt_term = torch.abs( e_xi  * d_xi4  * torch.sqrt(delta_xi_kappa2) +
                           e_eta * d_eta4 * torch.sqrt(delta_eta_kappa2) ) * \
                           rho*cs/T
    tmp_grad.copy( filt_term )
    kappa = C_kappa * gauss_filter( tmp_grad, metrics )

    # Synchronize artificial transport coefficients in the overlaps
    # and truncate to extended interiors needed for sigma, q
    tmp_grad.copy( mu    ); mu =    metrics.full2ext( tmp_grad.var )
    tmp_grad.copy( beta  ); beta =  metrics.full2ext( tmp_grad.var )
    tmp_grad.copy( kappa ); kappa = metrics.full2ext( tmp_grad.var )

    return mu,beta,kappa


# ------------------------------------------------------
# 4th-order artificial diffusivity (Kawai, Shankar, Lele JCP 2010) -- NEEDS UPDATE
#    Dilatation-based for bulk viscosity
#    For three-dimensional RHS
# ------------------------------------------------------
def art_diff4_D_3D(rho,drho_dx,drho_dy,drho_dz,S,tmp_grad,
                   div_u,curl_u,T,e,cs,grid,metrics,device):
    # Model constants
    C_mu    = 0.002
    C_beta  = 1.75
    C_kappa = 0.01

    # Computational grid spacing
    d_xi4  = grid.d_xi**4
    d_eta4 = grid.d_eta**4
    d_z4   = grid.d_z**4

    # Physical grid spacing
    drho_dx_int = metrics.ext2int( drho_dx )
    drho_dy_int = metrics.ext2int( drho_dy )
    drho_dz_int = metrics.ext2int( drho_dz )
    drho_mag = vec_mag_3D(drho_dx_int, drho_dy_int, drho_dz_int) + 1e-15
    delta_xi_beta2,delta_eta_beta2,delta_z_beta2 = \
        get_deltas_weighted( grid,
                             drho_dx_int/drho_mag,
                             drho_dy_int/drho_mag,
                             drho_dz_int/drho_mag )

    de_dx,de_dy,de_dz = metrics.grad_node( e, Neumann=True )
    de_mag = vec_mag_3D(de_dx, de_dy, de_dz) + 1e-15
    delta_xi_kappa2,delta_eta_kappa2,delta_z_kappa2 = \
        get_deltas_weighted( grid,
                             de_dx/de_mag,
                             de_dy/de_mag,
                             de_dz/de_mag )
    
    # Compute 4th derivatives
    S_int     = metrics.ext2int( S )
    div_u_int = metrics.ext2int( div_u )
    tmp_grad.copy( S_int );     S_xi,S_eta,S_z       = metrics.grad4_node(tmp_grad.var)
    tmp_grad.copy( div_u_int ); dil_xi,dil_eta,dil_z = metrics.grad4_node(tmp_grad.var)
    e_xi,e_eta,e_z = metrics.grad4_node( e )

    # Shock sensor
    #fsw = 1.0 # LAD-D0
    # LAD-D2
    div_u2 = div_u_int**2
    curl_u_int = metrics.ext2int( curl_u )
    fsw = torch.heaviside(-div_u_int,torch.tensor([0.5], dtype=metrics.WP).to(device)) * div_u2 / \
        ( div_u2 + curl_u_int**2 + 1e-32 )

    # Mu
    filt_term = torch.abs( S_xi  * d_xi4  * grid.delta_xi**2  +
                           S_eta * d_eta4 * grid.delta_eta**2 +
                           S_z   * d_z4   * grid.delta_z**2   ) * rho.interior()
    tmp_grad.copy( filt_term )
    mu = C_mu * gauss_filter( tmp_grad, metrics )

    # Beta
    filt_term = torch.abs( dil_xi  * d_xi4  * delta_xi_beta2  +
                           dil_eta * d_eta4 * delta_eta_beta2 +
                           dil_z   * d_z4   * delta_z_beta2   ) * rho.interior() * fsw
    tmp_grad.copy( filt_term )
    beta = C_beta * gauss_filter( tmp_grad, metrics )

    # Kappa
    filt_term = torch.abs( e_xi  * d_xi4  * torch.sqrt(delta_xi_kappa2)  +
                           e_eta * d_eta4 * torch.sqrt(delta_eta_kappa2) +
                           e_z   * d_z4   * torch.sqrt(delta_z_kappa2)   ) * \
                           rho.interior() * cs/metrics.full2int( T )
    tmp_grad.copy( filt_term )
    kappa = C_kappa * gauss_filter( tmp_grad, metrics )

    # Synchronize artificial transport coefficients in the overlaps
    # and truncate to extended interiors needed for sigma, q
    tmp_grad.copy( mu    ); mu =    metrics.full2ext( tmp_grad.var )
    tmp_grad.copy( beta  ); beta =  metrics.full2ext( tmp_grad.var )
    tmp_grad.copy( kappa ); kappa = metrics.full2ext( tmp_grad.var )

    return mu,beta,kappa


# ------------------------------------------------------
# Truncated Gaussian filter (Cook & Cabot JCP 2005)
# ------------------------------------------------------
def gauss_filter(u,metrics):
    iproc = metrics.iproc; npx = metrics.npx
    jproc = metrics.jproc; npy = metrics.npy
    kproc = metrics.kproc; npz = metrics.npz
    
    imin_ = metrics.imin_; imax_ = metrics.imax_
    jmin_ = metrics.jmin_; jmax_ = metrics.jmax_
    kmin_ = metrics.kmin_; kmax_ = metrics.kmax_
    
    # xi
    if (metrics.periodic_xi or (iproc>0 and iproc<npx-1)):
        # periodic boundaries or interior -- use overlaps
        u1l = u.var[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
        u2l = u.var[imin_-2:imax_-2,jmin_:jmax_,kmin_:kmax_]
        u3l = u.var[imin_-3:imax_-3,jmin_:jmax_,kmin_:kmax_]
        u4l = u.var[imin_-4:imax_-4,jmin_:jmax_,kmin_:kmax_]
        
        u1r = u.var[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
        u2r = u.var[imin_+2:imax_+2,jmin_:jmax_,kmin_:kmax_]
        u3r = u.var[imin_+3:imax_+3,jmin_:jmax_,kmin_:kmax_]
        u4r = u.var[imin_+4:imax_+4,jmin_:jmax_,kmin_:kmax_]
        
    elif (iproc==0 and npx>1):
        # On left boundary and npx>1
        #   mirror points across boundaries
        u1l = torch.cat(( u.var[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_],
                          u.var[imin_  :imax_-1,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u2l = torch.cat(( u.var[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_], u1l[:-1,:,:] ), dim=0)
        u3l = torch.cat(( u.var[imin_+3:imin_+4,jmin_:jmax_,kmin_:kmax_], u2l[:-1,:,:] ), dim=0)
        u4l = torch.cat(( u.var[imin_+4:imin_+5,jmin_:jmax_,kmin_:kmax_], u3l[:-1,:,:] ), dim=0)
        
        u1r = u.var[imin_+1:imax_+1,jmin_:jmax_,kmin_:kmax_]
        u2r = u.var[imin_+2:imax_+2,jmin_:jmax_,kmin_:kmax_]
        u3r = u.var[imin_+3:imax_+3,jmin_:jmax_,kmin_:kmax_]
        u4r = u.var[imin_+4:imax_+4,jmin_:jmax_,kmin_:kmax_]

    elif (iproc==npx-1 and npx>1):
        # On right boundary and npx>1
        u1l = u.var[imin_-1:imax_-1,jmin_:jmax_,kmin_:kmax_]
        u2l = u.var[imin_-2:imax_-2,jmin_:jmax_,kmin_:kmax_]
        u3l = u.var[imin_-3:imax_-3,jmin_:jmax_,kmin_:kmax_]
        u4l = u.var[imin_-4:imax_-4,jmin_:jmax_,kmin_:kmax_]

        u1r = torch.cat(( u.var[imin_+1:imax_  ,jmin_:jmax_,kmin_:kmax_],
                          u.var[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u2r = torch.cat(( u1r[1:,:,:], u.var[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u3r = torch.cat(( u2r[1:,:,:], u.var[imax_-4:imax_-3,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u4r = torch.cat(( u3r[1:,:,:], u.var[imax_-5:imax_-4,jmin_:jmax_,kmin_:kmax_] ), dim=0)

    else:
        # Non-periodic xi and npx=1
        u1l = torch.cat(( u.var[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_],
                          u.var[imin_  :imax_-1,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u2l = torch.cat(( u.var[imin_+2:imin_+3,jmin_:jmax_,kmin_:kmax_], u1l[:-1,:,:] ), dim=0)
        u3l = torch.cat(( u.var[imin_+3:imin_+4,jmin_:jmax_,kmin_:kmax_], u2l[:-1,:,:] ), dim=0)
        u4l = torch.cat(( u.var[imin_+4:imin_+5,jmin_:jmax_,kmin_:kmax_], u3l[:-1,:,:] ), dim=0)

        u1r = torch.cat(( u.var[imin_+1:imax_  ,jmin_:jmax_,kmin_:kmax_],
                          u.var[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u2r = torch.cat(( u1r[1:,:,:], u.var[imax_-3:imax_-2,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u3r = torch.cat(( u2r[1:,:,:], u.var[imax_-4:imax_-3,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        u4r = torch.cat(( u3r[1:,:,:], u.var[imax_-5:imax_-4,jmin_:jmax_,kmin_:kmax_] ), dim=0)
        
    uf = ( C0*u.interior() +
           C1*(u1l + u1r) +
           C2*(u2l + u2r) +
           C3*(u3l + u3r) +
           C4*(u4l + u4r) )

    # Sync overlaps
    u.copy( uf )

    # eta
    if (metrics.periodic_eta or (jproc>0 and jproc<npy-1)):
        # periodic or interior -- use overlaps
        u1l = u.var[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
        u2l = u.var[imin_:imax_,jmin_-2:jmax_-2,kmin_:kmax_]
        u3l = u.var[imin_:imax_,jmin_-3:jmax_-3,kmin_:kmax_]
        u4l = u.var[imin_:imax_,jmin_-4:jmax_-4,kmin_:kmax_]
        
        u1r = u.var[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
        u2r = u.var[imin_:imax_,jmin_+2:jmax_+2,kmin_:kmax_]
        u3r = u.var[imin_:imax_,jmin_+3:jmax_+3,kmin_:kmax_]
        u4r = u.var[imin_:imax_,jmin_+4:jmax_+4,kmin_:kmax_]
        
    elif (jproc==0 and npy>1):
        # On bottom boundary and npy>1
        #   mirror points across boundaries
        u1l = torch.cat(( u.var[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_],
                          u.var[imin_:imax_,jmin_  :jmax_-1,kmin_:kmax_] ), dim=1)
        u2l = torch.cat(( u.var[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_], u1l[:,:-1,:] ), dim=1)
        u3l = torch.cat(( u.var[imin_:imax_,jmin_+3:jmin_+4,kmin_:kmax_], u2l[:,:-1,:] ), dim=1)
        u4l = torch.cat(( u.var[imin_:imax_,jmin_+4:jmin_+5,kmin_:kmax_], u3l[:,:-1,:] ), dim=1)
        
        u1r = u.var[imin_:imax_,jmin_+1:jmax_+1,kmin_:kmax_]
        u2r = u.var[imin_:imax_,jmin_+2:jmax_+2,kmin_:kmax_]
        u3r = u.var[imin_:imax_,jmin_+3:jmax_+3,kmin_:kmax_]
        u4r = u.var[imin_:imax_,jmin_+4:jmax_+4,kmin_:kmax_]

    elif (jproc==npy-1 and npy>1):
        # On top boundary and npy>1
        u1l = u.var[imin_:imax_,jmin_-1:jmax_-1,kmin_:kmax_]
        u2l = u.var[imin_:imax_,jmin_-2:jmax_-2,kmin_:kmax_]
        u3l = u.var[imin_:imax_,jmin_-3:jmax_-3,kmin_:kmax_]
        u4l = u.var[imin_:imax_,jmin_-4:jmax_-4,kmin_:kmax_]

        u1r = torch.cat(( u.var[imin_:imax_,jmin_+1:jmax_  ,kmin_:kmax_],
                          u.var[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_] ), dim=1)
        u2r = torch.cat(( u1r[:,1:,:], u.var[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_] ), dim=1)
        u3r = torch.cat(( u2r[:,1:,:], u.var[imin_:imax_,jmax_-4:jmax_-3,kmin_:kmax_] ), dim=1)
        u4r = torch.cat(( u3r[:,1:,:], u.var[imin_:imax_,jmax_-5:jmax_-4,kmin_:kmax_] ), dim=1)

    else:
        # Non-periodic eta and npy=1
        u1l = torch.cat(( u.var[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_],
                          u.var[imin_:imax_,jmin_  :jmax_-1,kmin_:kmax_] ), dim=1)
        u2l = torch.cat(( u.var[imin_:imax_,jmin_+2:jmin_+3,kmin_:kmax_], u1l[:,:-1,:] ), dim=1)
        u3l = torch.cat(( u.var[imin_:imax_,jmin_+3:jmin_+4,kmin_:kmax_], u2l[:,:-1,:] ), dim=1)
        u4l = torch.cat(( u.var[imin_:imax_,jmin_+4:jmin_+5,kmin_:kmax_], u3l[:,:-1,:] ), dim=1)

        u1r = torch.cat(( u.var[imin_:imax_,jmin_+1:jmax_  ,kmin_:kmax_],
                          u.var[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_] ), dim=1)
        u2r = torch.cat(( u1r[:,1:,:], u.var[imin_:imax_,jmax_-3:jmax_-2,kmin_:kmax_] ), dim=1)
        u3r = torch.cat(( u2r[:,1:,:], u.var[imin_:imax_,jmax_-4:jmax_-3,kmin_:kmax_] ), dim=1)
        u4r = torch.cat(( u3r[:,1:,:], u.var[imin_:imax_,jmax_-5:jmax_-4,kmin_:kmax_] ), dim=1)

    uf = ( C0*u.interior() +
           C1*(u1l + u1r) +
           C2*(u2l + u2r) +
           C3*(u3l + u3r) +
           C4*(u4l + u4r) )

    # Z (periodic)
    if (metrics.nz > 1):
        # Sync overlaps
        u.copy( uf )
        u1l = u.var[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1]
        u2l = u.var[imin_:imax_,jmin_:jmax_,kmin_-2:kmax_-2]
        u3l = u.var[imin_:imax_,jmin_:jmax_,kmin_-3:kmax_-3]
        u4l = u.var[imin_:imax_,jmin_:jmax_,kmin_-4:kmax_-4]
        
        u1r = u.var[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1]
        u2r = u.var[imin_:imax_,jmin_:jmax_,kmin_+2:kmax_+2]
        u3r = u.var[imin_:imax_,jmin_:jmax_,kmin_+3:kmax_+3]
        u4r = u.var[imin_:imax_,jmin_:jmax_,kmin_+4:kmax_+4]

        u_filt = ( C0*u.interior() +
                   C1*(u1l + u1r) +
                   C2*(u2l + u2r) +
                   C3*(u3l + u3r) +
                   C4*(u4l + u4r) )
    else:
        u_filt = uf
        
    return u_filt


# ---------------------------------------------------------------------
# 6th-order (explicit) and 8th-order (compact) low-pass spatial filters
#    (Lele JCP 1992)
# ---------------------------------------------------------------------
class Lowpass_filter_6:
    def __init__(self, grid, decomp, implicit=False):
        self.WP = decomp.WP

        self.u_device = decomp.device
        self.solve_device = decomp.device

        # Save values for apply
        self.periodic_xi  = grid.periodic_xi
        self.periodic_eta = grid.periodic_eta
        self.BC_eta_top = grid.BC_eta_top
        self.BC_eta_bot = grid.BC_eta_bot
        self.ndim = grid.ndim

        # Parallel decomposition
        self.iproc = decomp.iproc; self.npx = decomp.npx
        self.jproc = decomp.jproc; self.npy = decomp.npy
        self.kproc = decomp.kproc; self.npz = decomp.npz

        self.imin_ = decomp.imin_; self.imax_ = decomp.imax_+1
        self.jmin_ = decomp.jmin_; self.jmax_ = decomp.jmax_+1
        self.kmin_ = decomp.kmin_; self.kmax_ = decomp.kmax_+1

        # Explicit or implicit
        self.implicit = implicit
        if (self.implicit):
            self.alpha = 0.495
        else:
            self.alpha = 0.0

        # Constants - F6
        self.a0_f6 = 11.0/16.0 + 5.0/8.0*self.alpha
        self.a1_f6 = 15.0/32.0 + 17.0/16.0*self.alpha
        self.a2_f6 = -3.0/16.0 + 3.0/8.0*self.alpha
        self.a3_f6 =  1.0/32.0 - self.alpha/16.0

        # Constants - F4
        self.a0_f4 =  0.625 + 0.75*self.alpha
        self.a1_f4 =  0.5   + self.alpha
        self.a2_f4 = -0.125 + 0.25*self.alpha

        # Constants - F2
        self.a0_f2 = 0.5 + self.alpha
        self.a1_f2 = 0.5 + self.alpha

        # Constants - F1 boundary scheme (non-Dirichlet)
        self.a0_f1_N = 0.5 + self.alpha * 0.5
        self.a1_f1_N = 0.5 + self.alpha * 0.5

        # Constants - F1 boundary scheme (Dirichlet)        
        self.a0_f1_D = 1.0 
        self.a1_f1_D = 0.0
        
        # Linear system for implicit filter
        #  NOTES:
        #    1. Can use higher-order boundary schemes (have points)
        #    2. Works for non-periodic, periodic, and mixed!
        
        if (self.implicit):
            self.decomp = decomp
            
            # x-direction
            self.a_x_N = torch.ones( decomp.nx_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.a_x_D = torch.ones( decomp.nx_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.b_x   = torch.ones( decomp.nx_, dtype=self.WP ).to(self.solve_device)
            self.c_x_N = torch.ones( decomp.nx_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.c_x_D = torch.ones( decomp.nx_, dtype=self.WP ).to(self.solve_device) * self.alpha

            # Modify stencils for Dirichlet boundaries 
            if (self.npx > 1):
                self.c_x_D[0]  = 0.0
                self.a_x_D[-1] = 0.0 # parallel TDMA takes a_y and a_x arrays different than serial
            else:
                # Serial run
                self.a_x_D[-2] = 0.0  # a_x[-1],c_x[-1] does not have any influence on serial tridiag solver
                self.c_x_D[0]  = 0.0

            # Factorize if nonperiodic and serial - disabled in favor of solve_serial_nonfactored
            #if (decomp.npx==1 and (not self.periodic_xi)):
            #    Tridiagonal.factor( self.a_x_N, self.b_x, self.c_x_N )

            # y-direction
            self.a_y_N = torch.ones( decomp.ny_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.a_y_D = torch.ones( decomp.ny_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.b_y   = torch.ones( decomp.ny_, dtype=self.WP ).to(self.solve_device)
            self.c_y_N = torch.ones( decomp.ny_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.c_y_D = torch.ones( decomp.ny_, dtype=self.WP ).to(self.solve_device) * self.alpha

            # Modify stencils for Dirichlet boundaries
            if (self.npy > 1):
                self.c_y_D[0]  = 0.0
                self.a_y_D[-1] = 0.0
            else:
                # Serial run
                self.a_y_D[-2] = 0.0 
                self.c_y_D[0]  = 0.0 

            # Factorize if nonperiodic and serial
            #if (decomp.npy==1 and (not self.periodic_eta)):
            #    Tridiagonal.factor( self.a_y, self.b_y, self.c_y )

            # z-direction (periodic)
            self.a_z = torch.ones( decomp.nz_, dtype=self.WP ).to(self.solve_device) * self.alpha
            self.b_z = torch.ones( decomp.nz_, dtype=self.WP ).to(self.solve_device)
            self.c_z = torch.ones( decomp.nz_, dtype=self.WP ).to(self.solve_device) * self.alpha

        else:
            self.a_x_N = None
            self.c_x_N = None
            self.a_y_N = None
            self.c_y_N = None
            self.a_x_D = None
            self.c_x_D = None
            self.a_y_D = None
            self.c_y_D = None
        
        return


    def is_Dirichlet_L(self):
        if (self.iproc==0 and (not self.periodic_xi)):
            return True
        else:
            return False


    def is_Dirichlet_R(self):
        if (self.iproc==self.npx-1 and (not self.periodic_xi)):
            return True
        else:
            return False

    
    def is_Dirichlet_B(self, name):
        if (self.jproc==0 and
            ((self.BC_eta_bot=='wall' and (name=='rhoU' or name=='rhoV' or name=='rhoW' or name=='rhoE' or name=='rho')) or
             (self.BC_eta_bot=='farfield'))):
            return True
        else:
            return False


    def is_Dirichlet_T(self,name):
        if (self.jproc==self.npy-1):
            ## JFM 6/30/23 -- hack to prevent oscillations upstream oscillations when filtering delq
            return False
        if (self.jproc==self.npy-1 and
            ((self.BC_eta_top=='wall' and (name=='rhoU' or name=='rhoV' or name=='rhoW' or name=='rhoE' or name=='rho')) or
             (self.BC_eta_top=='farfield'))):
            return True
        else:
            return False

        
    def expand_LR(self, u, uf):
        if self.is_Dirichlet_L():
            uf = torch.cat((u[:1,:,:], uf),  dim=0)
        if self.is_Dirichlet_R():
            uf = torch.cat((uf, u[-1:,:,:]), dim=0)
        return uf

    
    def expand_TB(self, u, uf, var):
        if self.is_Dirichlet_B(var):
            uf = torch.cat((u[:,:1,:], uf),  dim=1)
        if self.is_Dirichlet_T(var):
            uf = torch.cat((uf, u[:,-1:,:]), dim=1)
        return uf
    
    
    def apply(self, Q, names=None):

        ## JFM 6/30/23: If names==None, assume that Q is not a PCL State object

        if ((self.BC_eta_bot=='wall' or self.BC_eta_top=='wall') and self.implicit):
            # Implicit filter:  Need to solve Dirichlet and Neumann variables separately
            Dirichlet_vars = ['rhoU','rhoV','rhoW','rhoE', 'rho']
            Neumann_vars   = []
            for var in Q.names:
                if var not in Dirichlet_vars:
                    Neumann_vars.append(var)
                
            # ---------------------------- xi ----------------------------
            # Dirichlet variables
            uf_list = []
            for var in Dirichlet_vars:
                uf_list.append( self.filter_exp_x(Q[var].var, var) )

            # Convert list to Pytorch tensor
            uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)
            
            a_x_imp = self.a_x_N
            c_x_imp = self.c_x_N
            if self.is_Dirichlet_L(): c_x_imp = self.c_x_D
            if self.is_Dirichlet_R(): a_x_imp = self.a_x_D
            
            # Solve the tridiagonal system
            Tridiagonal.solve( self.decomp, 'x', a_x_imp, self.b_x, c_x_imp, uf_P, self.solve_device )

            # Expand truncated interiors and sync overlaps
            for ivar,var in enumerate(Dirichlet_vars):
                Q[var].copy(self.expand_TB(Q[var].interior(), uf_P[:,:,:,ivar].to(self.u_device), var))

                
            # Neumann variables
            uf_list = []
            for var in Neumann_vars:
                uf_list.append( self.filter_exp_x(Q[var].var, var) )

            if uf_list:
                # Convert list to Pytorch tensor
                uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)

                # Solve the tridiagonal system
                Tridiagonal.solve( self.decomp, 'x', a_x_imp, self.b_x, c_x_imp, uf_P, self.solve_device )

                # Expand truncated interiors and sync overlaps
                for ivar,var in enumerate(Neumann_vars):
                    Q[var].copy(self.expand_TB(Q[var].interior(), uf_P[:,:,:,ivar].to(self.u_device), var))

                
            # ---------------------------- eta ----------------------------
            # Dirichlet variables
            uf_list = []
            for var in Dirichlet_vars:
                uf_list.append( self.filter_exp_y(Q[var].var, var) )

            # Convert list to Pytorch tensor
            uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)
            
            a_y_imp = self.a_y_N
            c_y_imp = self.c_y_N
            if self.is_Dirichlet_B('rhoU'): c_y_imp = self.c_y_D
            if self.is_Dirichlet_T('rhoU'): a_y_imp = self.a_y_D
            
            # Solve the tridiagonal system
            uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous()
            Tridiagonal.solve( self.decomp, 'y', a_y_imp, self.b_y, c_y_imp, uf_P, self.solve_device )
            uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous().to(self.u_device)

            # Expand truncated interiors and sync overlaps
            for ivar,var in enumerate(Dirichlet_vars):
                Q[var].copy(self.expand_LR(Q[var].interior(), uf_P[:,:,:,ivar]))

                
            # Neumann variables
            uf_list = []
            for var in Neumann_vars:
                uf_list.append( self.filter_exp_y(Q[var].var, var) )

            if uf_list:
                # Convert list to Pytorch tensor
                uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)

                a_y_imp = self.a_y_N
                c_y_imp = self.c_y_N
                if self.is_Dirichlet_B('rho'): c_y_imp = self.c_y_D
                if self.is_Dirichlet_T('rho'): a_y_imp = self.a_y_D

                # Solve the tridiagonal system
                uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous()
                Tridiagonal.solve( self.decomp, 'y', a_y_imp, self.b_y, c_y_imp, uf_P, self.solve_device )
                uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous().to(self.u_device)

                # Expand truncated interiors and sync overlaps
                for ivar,var in enumerate(Neumann_vars):
                    Q[var].copy(self.expand_LR(Q[var].interior(), uf_P[:,:,:,ivar]))

        else:
            # All variables are either Dirichlet or Neumann, or we are not applying the implicit filter
            # ---------------------------- xi ----------------------------
            uf_list = []
            if names is None:
                for var in Q.names:
                    uf_list.append( self.filter_exp_x(Q[var].var, var) )
            else:
                for ivar,var in enumerate(names):
                    uf_list.append( self.filter_exp_x(Q[ivar], var) )
                    
            # Solve the tridiagonal system
            if self.implicit:
                # Convert list to Pytorch tensor
                uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)
                
                a_x_imp = self.a_x_N
                c_x_imp = self.c_x_N
                if self.is_Dirichlet_L(): c_x_imp = self.c_x_D
                if self.is_Dirichlet_R(): a_x_imp = self.a_x_D

                # Solve the tridiagonal system
                Tridiagonal.solve( self.decomp, 'x', a_x_imp, self.b_x, c_x_imp, uf_P, self.solve_device )

                # Expand truncated interiors and sync overlaps
                for ivar,var in enumerate(Q.names):
                    Q[var].copy(self.expand_TB(Q[var].interior(), uf_P[:,:,:,ivar].to(self.u_device), var))
                    
            else:
                # Expand truncated interiors and sync overlaps
                if names is None:
                    for ivar,var in enumerate(Q.names):
                        Q[var].copy(self.expand_TB(Q[var].interior(), uf_list[ivar], var))
                else:
                    for ivar,var in enumerate(names):
                        Q[ivar,
                          self.imin_:self.imax_,
                          self.jmin_:self.jmax_,
                          self.kmin_:self.kmax_] = self.expand_TB(Q[ivar,
                                                                    self.imin_:self.imax_,
                                                                    self.jmin_:self.jmax_,
                                                                    self.kmin_:self.kmax_], uf_list[ivar], var)

               
            # ---------------------------- eta ----------------------------
            uf_list = []
            if names is None:
                for var in Q.names:
                    uf_list.append( self.filter_exp_y(Q[var].var, var) )
            else:
                for ivar,var in enumerate(names):
                    uf_list.append( self.filter_exp_y(Q[ivar], var) )

            # Solve the tridiagonal system
            if self.implicit:
                # Convert list to Pytorch tensor
                uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)
                
                a_y_imp = self.a_y_N
                c_y_imp = self.c_y_N
                if self.is_Dirichlet_B('rhoU'): c_y_imp = self.c_y_D
                if self.is_Dirichlet_T('rhoU'): a_y_imp = self.a_y_D
            
                # Solve the tridiagonal system
                uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous()
                Tridiagonal.solve( self.decomp, 'y', a_y_imp, self.b_y, c_y_imp, uf_P, self.solve_device )
                uf_P = torch.swapaxes( uf_P, 0, 1 ).contiguous().to(self.u_device)

                # Expand truncated interiors and sync overlaps
                for ivar,var in enumerate(Q.names):
                    Q[var].copy(self.expand_LR(Q[var].interior(), uf_P[:,:,:,ivar]))

            else:
                # Expand truncated interiors and sync overlaps
                if names is None:
                    for ivar,var in enumerate(Q.names):
                        Q[var].copy(self.expand_LR(Q[var].interior(), uf_list[ivar]))
                else:
                    for ivar,var in enumerate(names):
                        Q[ivar,
                          self.imin_:self.imax_,
                          self.jmin_:self.jmax_,
                          self.kmin_:self.kmax_] = self.expand_LR(Q[ivar,
                                                                    self.imin_:self.imax_,
                                                                    self.jmin_:self.jmax_,
                                                                    self.kmin_:self.kmax_], uf_list[ivar])

            ## JFM 6/30/23: Need to fix 'if names is None' for:
            #  1. implicit solve
            #  2. z-direction

            
            
        # ---------------------------- z ----------------------------
        if (self.ndim==3):
            uf_list = []
            for var in Q.names:
                uf_list.append( self.filter_exp_z(Q[var].var) )

            # Solve the tridiagonal system
            if self.implicit:
                # Convert list to Pytorch tensor
                uf_P = torch.stack(uf_list, dim=3).to(self.solve_device)

                # Solve the tridiagonal system
                uf_P = torch.swapaxes( uf_P, 0, 2 ).contiguous()
                Tridiagonal.solve( self.decomp, 'z', self.a_z, self.b_z, self.c_z, uf_P, self.solve_device )
                uf_P = torch.swapaxes( uf_P, 0, 2 ).contiguous().to(self.u_device)

                # Sync overlaps
                for ivar,var in enumerate(Q.names):
                    Q[var].copy(uf_P[:,:,:,ivar])

            else:
                # Sync overlaps
                for ivar,var in enumerate(Q.names):
                    Q[var].copy(uf_list[ivar])

        return

    
    def filter_exp_x(self, u, name=None):
        iproc = self.iproc; npx = self.npx
        jproc = self.jproc; npy = self.npy
        kproc = self.kproc; npz = self.npz

        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_

        # Set interior offsets to prevent tangential filtering along Dirichlet boundaries
        # Bottom offset and weights
        jj1 = 0
        if self.is_Dirichlet_B(name):
            jj1 = 1

        # Top offset and weights
        jj2 = 0
        if self.is_Dirichlet_T(name):
            jj2 = 1

        # Left/right offsets and weights
        a0_f1_ii1 = self.a0_f1_N
        a1_f1_ii1 = self.a1_f1_N
        a0_f1_ii2 = self.a0_f1_N
        a1_f1_ii2 = self.a1_f1_N
        if self.is_Dirichlet_L():
            a0_f1_ii1 = self.a0_f1_D
            a1_f1_ii1 = self.a1_f1_D
        if self.is_Dirichlet_R():
            a0_f1_ii2 = self.a0_f1_D
            a1_f1_ii2 = self.a1_f1_D
            
        # xi
        if (self.periodic_xi or (iproc>0 and iproc<npx-1)):
            # Periodic xi boundaries or x-interior
            ui  = u[imin_  :imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            
            u1l = u[imin_-1:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2l = u[imin_-2:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3l = u[imin_-3:imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            u1r = u[imin_+1:imax_+1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2r = u[imin_+2:imax_+2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3r = u[imin_+3:imax_+3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            
            # F6 scheme
            uf = self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                       self.a2_f6*(u2l + u2r) +
                                       self.a3_f6*(u3l + u3r) )
        
        elif (iproc==0 and npx>1):
            # Left non-periodic boundary and npx>1
            # For left boundary
            u0  = u[imin_  :imin_+1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1  = u[imin_+1:imin_+2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2  = u[imin_+2:imin_+3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3  = u[imin_+3:imin_+4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u4  = u[imin_+4:imin_+5,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # For interior
            u3l = u[imin_  :imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2l = u[imin_+1:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1l = u[imin_+2:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            ui  = u[imin_+3:imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1r = u[imin_+4:imax_+1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2r = u[imin_+5:imax_+2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3r = u[imin_+6:imax_+3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # F1 (0), F2 (1), F4 (2), F6 (interior)
            uf = torch.cat(( ( a0_f1_ii1*u0 + a1_f1_ii1*u1 ),
                             ( self.a0_f2*u1 + 0.5 * self.a1_f2*(u0 + u2) ),
                             ( self.a0_f4*u2 + 0.5*( self.a1_f4*(u1 + u3) + self.a2_f4*(u0 + u4) )),
                             ( self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                                     self.a2_f6*(u2l + u2r) +
                                                     self.a3_f6*(u3l + u3r) )) ), dim=0)

        elif (iproc==npx-1 and npx>1):
            # Right non-periodic boundary and npx>1
            # For interior
            u3l = u[imin_-3:imax_-6,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2l = u[imin_-2:imax_-5,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1l = u[imin_-1:imax_-4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            ui  = u[imin_  :imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1r = u[imin_+1:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2r = u[imin_+2:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3r = u[imin_+3:imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # For right boundary
            um0 = u[imax_-1:imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um1 = u[imax_-2:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um2 = u[imax_-3:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um3 = u[imax_-4:imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um4 = u[imax_-5:imax_-4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # F6 (interior), F4 (m2), F2 (m1), F1 (m0)
            uf = torch.cat(( ( self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                                     self.a2_f6*(u2l + u2r) +
                                                     self.a3_f6*(u3l + u3r) )),
                             ( self.a0_f4*um2 + 0.5*( self.a1_f4*(um1 + um3) + self.a2_f4*(um0 + um4) )),
                             ( self.a0_f2*um1 + 0.5 * self.a1_f2*(um0 + um2) ),
                             ( a0_f1_ii2*um0 + a1_f1_ii2*um1 ) ), dim=0)

        else:
            # Non-periodic xi and npx=1
            # For left boundary
            u0  = u[imin_  :imin_+1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1  = u[imin_+1:imin_+2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2  = u[imin_+2:imin_+3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3  = u[imin_+3:imin_+4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u4  = u[imin_+4:imin_+5,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            
            # For interior
            u3l = u[imin_  :imax_-6,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2l = u[imin_+1:imax_-5,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1l = u[imin_+2:imax_-4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            ui  = u[imin_+3:imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u1r = u[imin_+4:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u2r = u[imin_+5:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            u3r = u[imin_+6:imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # For right boundary
            um0 = u[imax_-1:imax_  ,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um1 = u[imax_-2:imax_-1,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um2 = u[imax_-3:imax_-2,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um3 = u[imax_-4:imax_-3,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]
            um4 = u[imax_-5:imax_-4,jmin_+jj1:jmax_-jj2,kmin_:kmax_,...]

            # F1 (0), F2 (1), F4 (2), F6 (interior), F4 (m2), F2 (m1), F1 (m0)
            uf = torch.cat(( ( a0_f1_ii1*u0 + a1_f1_ii1*u1 ),
                             ( self.a0_f2*u1 +  0.5 * self.a1_f2*(u0 + u2) ),
                             ( self.a0_f4*u2 +  0.5*( self.a1_f4*(u1 + u3) + self.a2_f4*(u0 + u4) )),
                             ( self.a0_f6*ui +  0.5*( self.a1_f6*(u1l + u1r) +
                                                      self.a2_f6*(u2l + u2r) +
                                                      self.a3_f6*(u3l + u3r) )),
                             ( self.a0_f4*um2 + 0.5*( self.a1_f4*(um1 + um3) + self.a2_f4*(um0 + um4) )),
                             ( self.a0_f2*um1 + 0.5 * self.a1_f2*(um0 + um2) ),
                             ( a0_f1_ii2*um0 + a1_f1_ii2*um1 ) ), dim=0)

        return uf
    
    
    def filter_exp_y(self, u, name=None):
        iproc = self.iproc; npx = self.npx
        jproc = self.jproc; npy = self.npy
        kproc = self.kproc; npz = self.npz

        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_

        # Set interior offsets to prevent tangential filtering along Dirichlet boundaries
        # Bottom offset and weights
        a0_f1_jj1 = self.a0_f1_N
        a1_f1_jj1 = self.a1_f1_N
        if self.is_Dirichlet_B(name):
            a0_f1_jj1 = self.a0_f1_D
            a1_f1_jj1 = self.a1_f1_D

        # Top offset and weights
        a0_f1_jj2 = self.a0_f1_N
        a1_f1_jj2 = self.a1_f1_N
        if self.is_Dirichlet_T(name):
            a0_f1_jj2 = self.a0_f1_D
            a1_f1_jj2 = self.a1_f1_D

        # Left/right offsets and weights
        ii1 = 0
        ii2 = 0
        if self.is_Dirichlet_L():
            ii1 = 1
        if self.is_Dirichlet_R():
            ii2 = 1
            
        # eta
        if (self.periodic_eta or (jproc>0 and jproc<npy-1)):
            # periodic or interior -- use overlaps
            ui  = u[imin_+ii1:imax_-ii2,jmin_  :jmax_  ,kmin_:kmax_,...]
            
            u1l = u[imin_+ii1:imax_-ii2,jmin_-1:jmax_-1,kmin_:kmax_,...]
            u2l = u[imin_+ii1:imax_-ii2,jmin_-2:jmax_-2,kmin_:kmax_,...]
            u3l = u[imin_+ii1:imax_-ii2,jmin_-3:jmax_-3,kmin_:kmax_,...]

            u1r = u[imin_+ii1:imax_-ii2,jmin_+1:jmax_+1,kmin_:kmax_,...]
            u2r = u[imin_+ii1:imax_-ii2,jmin_+2:jmax_+2,kmin_:kmax_,...]
            u3r = u[imin_+ii1:imax_-ii2,jmin_+3:jmax_+3,kmin_:kmax_,...]

            # F6 scheme
            uf = self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                       self.a2_f6*(u2l + u2r) +
                                       self.a3_f6*(u3l + u3r) )

        elif (jproc==0 and npy>1):
            # Bottom non-periodic boundary and npy>1
            # For bottom boundary
            u0  = u[imin_+ii1:imax_-ii2,jmin_  :jmin_+1,kmin_:kmax_,...]
            u1  = u[imin_+ii1:imax_-ii2,jmin_+1:jmin_+2,kmin_:kmax_,...]
            u2  = u[imin_+ii1:imax_-ii2,jmin_+2:jmin_+3,kmin_:kmax_,...]
            u3  = u[imin_+ii1:imax_-ii2,jmin_+3:jmin_+4,kmin_:kmax_,...]
            u4  = u[imin_+ii1:imax_-ii2,jmin_+4:jmin_+5,kmin_:kmax_,...]

            # For interior
            u3l = u[imin_+ii1:imax_-ii2,jmin_  :jmax_-3,kmin_:kmax_,...]
            u2l = u[imin_+ii1:imax_-ii2,jmin_+1:jmax_-2,kmin_:kmax_,...]
            u1l = u[imin_+ii1:imax_-ii2,jmin_+2:jmax_-1,kmin_:kmax_,...]
            ui  = u[imin_+ii1:imax_-ii2,jmin_+3:jmax_  ,kmin_:kmax_,...]
            u1r = u[imin_+ii1:imax_-ii2,jmin_+4:jmax_+1,kmin_:kmax_,...]
            u2r = u[imin_+ii1:imax_-ii2,jmin_+5:jmax_+2,kmin_:kmax_,...]
            u3r = u[imin_+ii1:imax_-ii2,jmin_+6:jmax_+3,kmin_:kmax_,...]

            # F1 (0), F2 (1), F4 (2), F6 (interior)
            uf = torch.cat(( ( a0_f1_jj1*u0 + a1_f1_jj1*u1 ),
                             ( self.a0_f2*u1 + 0.5 * self.a1_f2*(u0 + u2) ),
                             ( self.a0_f4*u2 + 0.5*( self.a1_f4*(u1 + u3) + self.a2_f4*(u0 + u4) )),
                             ( self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                                     self.a2_f6*(u2l + u2r) +
                                                     self.a3_f6*(u3l + u3r) )) ), dim=1)
        
        elif (jproc==npy-1 and npy>1):
            # Top non-periodic boundary and npy>1
            # For interior
            u3l = u[imin_+ii1:imax_-ii2,jmin_-3:jmax_-6,kmin_:kmax_,...]
            u2l = u[imin_+ii1:imax_-ii2,jmin_-2:jmax_-5,kmin_:kmax_,...]
            u1l = u[imin_+ii1:imax_-ii2,jmin_-1:jmax_-4,kmin_:kmax_,...]
            ui  = u[imin_+ii1:imax_-ii2,jmin_  :jmax_-3,kmin_:kmax_,...]
            u1r = u[imin_+ii1:imax_-ii2,jmin_+1:jmax_-2,kmin_:kmax_,...]
            u2r = u[imin_+ii1:imax_-ii2,jmin_+2:jmax_-1,kmin_:kmax_,...]
            u3r = u[imin_+ii1:imax_-ii2,jmin_+3:jmax_  ,kmin_:kmax_,...]

            # For top boundary
            um0 = u[imin_+ii1:imax_-ii2,jmax_-1:jmax_  ,kmin_:kmax_,...]
            um1 = u[imin_+ii1:imax_-ii2,jmax_-2:jmax_-1,kmin_:kmax_,...]
            um2 = u[imin_+ii1:imax_-ii2,jmax_-3:jmax_-2,kmin_:kmax_,...]
            um3 = u[imin_+ii1:imax_-ii2,jmax_-4:jmax_-3,kmin_:kmax_,...]
            um4 = u[imin_+ii1:imax_-ii2,jmax_-5:jmax_-4,kmin_:kmax_,...]

            # F6 (interior), F4 (m2), F2 (m1), F1 (m0)
            uf = torch.cat(( ( self.a0_f6*ui + 0.5*( self.a1_f6*(u1l + u1r) +
                                                     self.a2_f6*(u2l + u2r) +
                                                     self.a3_f6*(u3l + u3r) )),
                             ( self.a0_f4*um2 + 0.5*( self.a1_f4*(um1 + um3) + self.a2_f4*(um0 + um4) )),
                             ( self.a0_f2*um1 + 0.5 * self.a1_f2*(um0 + um2) ),
                             ( a0_f1_jj2*um0 + a1_f1_jj2*um1 ) ), dim=1)
        
        else:
            # Non-periodic eta and npy=1
            u0  = u[imin_+ii1:imax_-ii2,jmin_  :jmin_+1,kmin_:kmax_,...]
            u1  = u[imin_+ii1:imax_-ii2,jmin_+1:jmin_+2,kmin_:kmax_,...]
            u2  = u[imin_+ii1:imax_-ii2,jmin_+2:jmin_+3,kmin_:kmax_,...]
            u3  = u[imin_+ii1:imax_-ii2,jmin_+3:jmin_+4,kmin_:kmax_,...]
            u4  = u[imin_+ii1:imax_-ii2,jmin_+4:jmin_+5,kmin_:kmax_,...]

            # For interior
            u3l = u[imin_+ii1:imax_-ii2,jmin_  :jmax_-6,kmin_:kmax_,...]
            u2l = u[imin_+ii1:imax_-ii2,jmin_+1:jmax_-5,kmin_:kmax_,...]
            u1l = u[imin_+ii1:imax_-ii2,jmin_+2:jmax_-4,kmin_:kmax_,...]
            ui  = u[imin_+ii1:imax_-ii2,jmin_+3:jmax_-3,kmin_:kmax_,...]
            u1r = u[imin_+ii1:imax_-ii2,jmin_+4:jmax_-2,kmin_:kmax_,...]
            u2r = u[imin_+ii1:imax_-ii2,jmin_+5:jmax_-1,kmin_:kmax_,...]
            u3r = u[imin_+ii1:imax_-ii2,jmin_+6:jmax_  ,kmin_:kmax_,...]

            # For top boundary
            um0 = u[imin_+ii1:imax_-ii2,jmax_-1:jmax_  ,kmin_:kmax_,...]
            um1 = u[imin_+ii1:imax_-ii2,jmax_-2:jmax_-1,kmin_:kmax_,...]
            um2 = u[imin_+ii1:imax_-ii2,jmax_-3:jmax_-2,kmin_:kmax_,...]
            um3 = u[imin_+ii1:imax_-ii2,jmax_-4:jmax_-3,kmin_:kmax_,...]
            um4 = u[imin_+ii1:imax_-ii2,jmax_-5:jmax_-4,kmin_:kmax_,...]

            # F1 (0), F2 (1), F4 (2), F6 (interior), F4 (m2), F2 (m1), F1 (m0)
            uf = torch.cat(( ( a0_f1_jj1*u0 + a1_f1_jj1*u1 ),
                             ( self.a0_f2*u1 +  0.5 * self.a1_f2*(u0 + u2) ),
                             ( self.a0_f4*u2 +  0.5*( self.a1_f4*(u1 + u3) + self.a2_f4*(u0 + u4) )),
                             ( self.a0_f6*ui +  0.5*( self.a1_f6*(u1l + u1r) +
                                                      self.a2_f6*(u2l + u2r) +
                                                      self.a3_f6*(u3l + u3r) )),
                             ( self.a0_f4*um2 + 0.5*( self.a1_f4*(um1 + um3) + self.a2_f4*(um0 + um4) )),
                             ( self.a0_f2*um1 + 0.5 * self.a1_f2*(um0 + um2) ),
                             ( a0_f1_jj2*um0 + a1_f1_jj2*um1 ) ), dim=1)

        return uf

    
    def filter_exp_z(self, u):
        imin_ = self.imin_; imax_ = self.imax_
        jmin_ = self.jmin_; jmax_ = self.jmax_
        kmin_ = self.kmin_; kmax_ = self.kmax_
        
        # Z (periodic)
        u1l = u[imin_:imax_,jmin_:jmax_,kmin_-1:kmax_-1,...]
        u2l = u[imin_:imax_,jmin_:jmax_,kmin_-2:kmax_-2,...]
        u3l = u[imin_:imax_,jmin_:jmax_,kmin_-3:kmax_-3,...]

        u1r = u[imin_:imax_,jmin_:jmax_,kmin_+1:kmax_+1,...]
        u2r = u[imin_:imax_,jmin_:jmax_,kmin_+2:kmax_+2,...]
        u3r = u[imin_:imax_,jmin_:jmax_,kmin_+3:kmax_+3,...]

        # F6 scheme
        uf = self.a0_f6*u[imin_:imax_,jmin_:jmax_,kmin_:kmax_,...] + 0.5*( self.a1_f6*(u1l + u1r) +
                                                                           self.a2_f6*(u2l + u2r) +
                                                                           self.a3_f6*(u3l + u3r) )

        return uf


# ------------------------------------------------------
# 2D strain-rate tensor magnitude
# ------------------------------------------------------
def strainrate_mag_2D(u11,u12,u21,u22):
    S11 = u11
    S12 = 0.5*(u12 + u21)
    S22 = u22
    
    # u_mag = sqrt(u_ij : u_ij)
    return torch.sqrt( 2.0*( S11*S11 + 2.0*S12*S12 + S22*S22 ) )


# ------------------------------------------------------
# 3D strain-rate tensor magnitude
# ------------------------------------------------------
def strainrate_mag_3D(u11,u12,u13,u21,u22,u23,u31,u32,u33):
    S11 = u11
    S12 = 0.5*(u12 + u21)
    S13 = 0.5*(u13 + u31)
    S22 = u22
    S23 = 0.5*(u23 + u32)
    S33 = u33
    
    # u_mag = sqrt(u_ij : u_ij)
    return torch.sqrt( 2.0* (S11*S11 + 2.0*S12*S12 + 2.0*S13*S13 +
                             S22*S22 + 2.0*S23*S23 +
                             S33*S33) )


# ------------------------------------------------------
# 2D vector magnitude
# ------------------------------------------------------
def vec_mag_2D(u1,u2):
    return torch.sqrt( u1*u1 + u2*u2 )


# ------------------------------------------------------
# 3D vector magnitude
# ------------------------------------------------------
def vec_mag_3D(u1,u2,u3):
    return torch.sqrt( u1*u1 + u2*u2 + u3*u3 )
