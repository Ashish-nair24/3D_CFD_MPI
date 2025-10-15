"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Grid.py

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


import numpy as np
import torch
from mpi4py import MPI

from . import Data


#=======================================================
# GENERAL GRID FUNCTIONS
#=======================================================


# ------------------------------------------------------
# Finalize the global grid dimensions
#       Called once upon invoking PyFlowCL.run()
#       Needed before initializing the parallel decomposition
# ------------------------------------------------------
def enforce_periodic(cfg):
    grid = cfg.grid
    
    # Modified grids to account for periodic BCs
    if (grid.periodic_xi and grid.Nx1 > 1):
        grid.xi_grid_mod = grid.xi_grid[:-1]
    else:
        grid.xi_grid_mod = grid.xi_grid
        # 
    if grid.periodic_eta:
        grid.eta_grid_mod = grid.eta_grid[:-1]
    else:
        grid.eta_grid_mod = grid.eta_grid

    # Save global sizes
    grid.Nx1 = len(grid.xi_grid_mod)
    grid.Nx2 = len(grid.eta_grid_mod)

    # Initialize the z-grid
    if (grid.Nx3 > 1):
        grid.z_grid = torch.linspace(-0.5*grid.Lx3,0.5*grid.Lx3,grid.Nx3).to(cfg.device)
        grid.d_z    = grid.Lx3/float(grid.Nx3-1)
        grid.z_grid_mod = grid.z_grid[:-1]
        grid.ndim = 3
        grid.Nx3  = len(grid.z_grid_mod)
    else:
        grid.Nx3  = 1
        if (grid.Nx1 > 1):
            grid.ndim = 2
        else:
            grid.ndim = 1

    return grid


# ------------------------------------------------------
# Initialize grid transforms
#       Called once upon invoking PyFlowCL.run()
# ------------------------------------------------------
def initialize_transforms(cfg, grid, decomp, metrics):

    # Truncate the grids to this MPI task's domain interior
    #       All subsequent grid arrays are local only
    grid.xi_grid_mod  = grid.xi_grid_mod [decomp.imin_loc:decomp.imax_loc+1]
    grid.eta_grid_mod = grid.eta_grid_mod[decomp.jmin_loc:decomp.jmax_loc+1]
    if (grid.ndim==3):
        grid.z_grid_mod = grid.z_grid_mod[decomp.kmin_loc:decomp.kmax_loc+1]
       
    # Meshgrids
    grid.Xi, grid.Eta = torch.meshgrid( grid.xi_grid_mod, grid.eta_grid_mod, indexing='ij' )
    if hasattr(grid, "get_xy"):
        grid.X, grid.Y = grid.get_xy( grid.Xi, grid.Eta )
    else:
        if (decomp.rank==0): print(" --> Reading grid from file")
        grid.X, grid.Y = grid.read_grid(decomp, cfg)

    # Initialize temporary PCL_Var objects for meshgrid communication
    X_PCL = Data.PCL_Var(decomp, 'X', force_2D=True)
    Y_PCL = Data.PCL_Var(decomp, 'Y', force_2D=True)
    
    # Grid transforms at nodes
    if hasattr(grid, 'get_transform'):
        # Our specific grid object has analytic transforms -- use them
        grid.x_xi ,grid.y_xi, grid.x_eta, grid.y_eta, Jac = grid.get_transform( grid.Xi, grid.Eta )
        
    else:
        # Grid object does not have analytic transforms -- compute using finite differences
        if (decomp.rank==0): print(' --> Using numerical grid transforms')

        X_PCL.copy(grid.X[:,:,None])
        Y_PCL.copy(grid.Y[:,:,None])

        # Compute 4th-order grid transforms using PCL metrics
        x_xi, x_eta, _ = metrics.grad_node(X_PCL.var, compute_dz=False, force_2D=True)
        y_xi, y_eta, _ = metrics.grad_node(Y_PCL.var, compute_dz=False, force_2D=True)

        grid.x_xi = x_xi[:,:,0]; grid.x_eta = x_eta[:,:,0]
        grid.y_xi = y_xi[:,:,0]; grid.y_eta = y_eta[:,:,0]

        Jac = grid.x_xi * grid.y_eta - grid.x_eta * grid.y_xi

    # Jacobian determinant of the forward transformation (x --> xi)
    # This is 1/Jac
    grid.inv_Jac = 1.0/Jac
    
    grid.xi_x  =  grid.y_eta * grid.inv_Jac
    grid.eta_x = -grid.y_xi  * grid.inv_Jac
    grid.xi_y  = -grid.x_eta * grid.inv_Jac
    grid.eta_y =  grid.x_xi  * grid.inv_Jac

    # Second-order forward  transforms
    # x
    X_PCL.copy(grid.x_xi[...,None]); Y_PCL.copy(grid.x_eta[...,None])
    x_xi2, _, _  = metrics.grad_node(X_PCL.var, compute_dy=False, compute_dz=False, force_2D=True)
    _, x_eta2, _ = metrics.grad_node(Y_PCL.var, compute_dx=False, compute_dz=False, force_2D=True)
    # y
    X_PCL.copy(grid.y_xi[...,None]); Y_PCL.copy(grid.y_eta[...,None])
    y_xi2, _, _  = metrics.grad_node(X_PCL.var, compute_dy=False, compute_dz=False, force_2D=True)
    _, y_eta2, _ = metrics.grad_node(Y_PCL.var, compute_dx=False, compute_dz=False, force_2D=True)

    inv_Jac2 = 1.0/(x_xi2 * y_eta2 - x_eta2 * y_xi2)

    # Second-order inverse transforms
    grid.xi_xx  =  y_eta2 * inv_Jac2
    grid.xi_yy  = -x_eta2 * inv_Jac2
    grid.eta_xx = -y_xi2  * inv_Jac2
    grid.eta_yy =  x_xi2  * inv_Jac2

    # Grid transforms at midpoints
    grid.have_midpoint_transforms = False
    if hasattr(grid, 'get_transform'):
        # Use analytic grid transforms for midpoints
        xi_m  = grid.xi_grid_mod  + 0.5*grid.d_xi
        eta_m = grid.eta_grid_mod + 0.5*grid.d_eta

        Xi_m, Eta_m = torch.meshgrid(xi_m, eta_m, indexing='ij')

        # x-midpoints
        x_xi_xm, y_xi_xm, x_eta_xm, y_eta_xm, Jac_xm = grid.get_transform(Xi_m, grid.Eta)
        inv_Jac_xm = 1.0/Jac_xm
        
        # y-midpoints
        x_xi_ym, y_xi_ym, x_eta_ym, y_eta_ym, Jac_ym = grid.get_transform(grid.Xi, Eta_m)
        inv_Jac_ym = 1.0/Jac_ym

        # Inverse tranforms
        # x-midpoints
        grid.xi_x_xm  =  y_eta_xm * inv_Jac_xm
        grid.eta_x_xm = -y_xi_xm  * inv_Jac_xm
        grid.xi_y_xm  = -x_eta_xm * inv_Jac_xm
        grid.eta_y_xm =  x_xi_xm  * inv_Jac_xm
        # y-midpoints
        grid.xi_x_ym  =  y_eta_ym * inv_Jac_ym
        grid.eta_x_ym = -y_xi_ym  * inv_Jac_ym
        grid.xi_y_ym  = -x_eta_ym * inv_Jac_ym
        grid.eta_y_ym =  x_xi_ym  * inv_Jac_ym
        
        grid.have_midpoint_transforms = True
        
    else:
        # Use numerical grid transforms for midpoints
        # x-midpoints
        X_xm = 0.5*(grid.X[1:,:,None] + grid.X[:-1,:,None])
        Y_xm = 0.5*(grid.Y[1:,:,None] + grid.Y[:-1,:,None])
        X_xm = torch.cat((X_xm, 2.0*grid.X[-1:,:,None] - X_xm[-1:]), dim=0)  ## CHK for MPI (extrapolating into neighboring CPU decomp?)
        Y_xm = torch.cat((Y_xm, 2.0*grid.Y[-1:,:,None] - Y_xm[-1:]), dim=0)
        X_PCL.copy(X_xm)
        Y_PCL.copy(Y_xm)

        x_xi_xm, x_eta_xm, _ = metrics.grad_node(X_PCL.var, compute_dz=False, force_2D=True)
        y_xi_xm, y_eta_xm, _ = metrics.grad_node(Y_PCL.var, compute_dz=False, force_2D=True)
        inv_Jac_xm = 1.0/(x_xi_xm * y_eta_xm - x_eta_xm * y_xi_xm)
        
        grid.xi_x_xm  = ( y_eta_xm * inv_Jac_xm)[:,:,0]
        grid.eta_x_xm = (-y_xi_xm  * inv_Jac_xm)[:,:,0]
        grid.xi_y_xm  = (-x_eta_xm * inv_Jac_xm)[:,:,0]
        grid.eta_y_xm = ( x_xi_xm  * inv_Jac_xm)[:,:,0]
        
        # y-midpoints
        X_ym = 0.5*(grid.X[:,1:,None] + grid.X[:,:-1,None])
        Y_ym = 0.5*(grid.Y[:,1:,None] + grid.Y[:,:-1,None])
        X_ym = torch.cat((X_ym, 2.0*grid.X[:,-1:,None] - X_ym[:,-1:]), dim=1)  ## CHK for MPI
        Y_ym = torch.cat((Y_ym, 2.0*grid.Y[:,-1:,None] - Y_ym[:,-1:]), dim=1)
        X_PCL.copy(X_ym)
        Y_PCL.copy(Y_ym)

        x_xi_ym, x_eta_ym, _ = metrics.grad_node(X_PCL.var, compute_dz=False, force_2D=True)
        y_xi_ym, y_eta_ym, _ = metrics.grad_node(Y_PCL.var, compute_dz=False, force_2D=True)
        inv_Jac_ym = 1.0/(x_xi_ym * y_eta_ym - x_eta_ym * y_xi_ym)
        
        grid.xi_x_ym  = ( y_eta_ym * inv_Jac_ym)[:,:,0]
        grid.eta_x_ym = (-y_xi_ym  * inv_Jac_ym)[:,:,0]
        grid.xi_y_ym  = (-x_eta_ym * inv_Jac_ym)[:,:,0]
        grid.eta_y_ym = ( x_xi_ym  * inv_Jac_ym)[:,:,0]
         
        grid.have_midpoint_transforms = True

    # Jacobian determinant of the inverse transformation (xi --> x)
    grid.inv_trans_Jac = torch.abs(grid.xi_x * grid.eta_y - grid.xi_y * grid.eta_x)

    # Jacobian-normalized inverse transforms (needed for predict_dt)
    grid.xi_x_Jac  = grid.xi_x  / grid.inv_trans_Jac
    grid.xi_y_Jac  = grid.xi_y  / grid.inv_trans_Jac
    grid.eta_x_Jac = grid.eta_x / grid.inv_trans_Jac
    grid.eta_y_Jac = grid.eta_y / grid.inv_trans_Jac

    # Broadcast meshgrids for 3D
    if (grid.ndim==3):
        grid.X = grid.X[:,:,None].expand(decomp.nx_, decomp.ny_, decomp.nz_)
        grid.Y = grid.Y[:,:,None].expand(decomp.nx_, decomp.ny_, decomp.nz_)
        grid.Z = grid.z_grid_mod[None,None,:].expand(decomp.nx_, decomp.ny_, decomp.nz_)
    else:
        grid.Z = None

    # Physical-space directional grid spacing
    #       Needed for artificial diffusivity
    if (grid.ndim > 1):
        grid.delta_xi,grid.delta_eta,grid.delta_z = get_deltas(grid.ndim,grid.X,grid.Y,grid.Z)

    # dx,dy,dz needed for CFL calculation
    # x
    if (grid.Nx1 > 1):
        grid.Dx = abs(grid.x_xi*grid.d_xi) + abs(grid.x_eta*grid.d_eta)
    else:
        grid.Dx = torch.ones_like(grid.x_xi)
    # y
    grid.Dy = abs(grid.y_xi*grid.d_xi) + abs(grid.y_eta*grid.d_eta)
    # z
    if (grid.Nx3 > 1):
        grid.Dz = grid.z_grid_mod[1] - grid.z_grid_mod[0]
    else:
        grid.Dz = 1.0

    # Save grid to CPU for I/O
    grid.X_cpu = grid.X.cpu()
    grid.Y_cpu = grid.Y.cpu()
    if (grid.ndim==3):
        grid.Z_cpu = grid.Z.cpu()

    return
    

# ------------------------------------------------------
# Physical-space directional grid spacing -- NEEDS BOUNDARY MODIFICATIONS FOR MPI
#       Needed for artificial diffusivity
# ------------------------------------------------------
def get_deltas(ndim,X,Y,Z=None):
    if (ndim==2):
        return get_deltas_2D(X,Y)
    elif (ndim==3):
        return get_deltas_3D(X,Y,Z)
    else:
        raise Exception('Grid.get_deltas: not yet implemented for 1D')

    
def get_deltas_2D(X,Y):
    delta_xi  = torch.sqrt(torch.cat(( (X[1:2,:] - X[:1,:])**2 + (Y[1:2,:] - Y[:1,:])**2,
                                       (0.5*(X[2:,:] - X[:-2,:]))**2 + (0.5*(Y[2:,:] - Y[:-2,:]))**2,
                                       (X[-1:,:] - X[-2:-1,:])**2 + (Y[-1:,:] - Y[-2:-1,:])**2 ),
                                     dim=0))
    delta_eta = torch.sqrt(torch.cat(( (X[:,1:2] - X[:,:1])**2 + (Y[:,1:2] - Y[:,:1])**2,
                                       (0.5*(X[:,2:] - X[:,:-2]))**2 + (0.5*(Y[:,2:] - Y[:,:-2]))**2,
                                       (X[:,-1:] - X[:,-2:-1])**2 + (Y[:,-1:] - Y[:,-2:-1])**2 ),
                                     dim=1))
    return delta_xi,delta_eta,None

    
def get_deltas_3D(X,Y,Z):
    delta_xi  = torch.sqrt(torch.cat(( (X[1:2,:,:] - X[:1,:,:])**2 + (Y[1:2,:,:] - Y[:1,:,:])**2 + (Z[1:2,:,:] - Z[:1,:,:])**2,
                                       (0.5*(X[2:,:,:] - X[:-2,:,:]))**2 + (0.5*(Y[2:,:,:] - Y[:-2,:,:]))**2 + (0.5*(Z[2:,:,:] - Z[:-2,:,:]))**2,
                                       (X[-1:,:,:] - X[-2:-1,:,:])**2 + (Y[-1:,:,:] - Y[-2:-1,:,:])**2 + (Z[-1:,:,:] - Z[-2:-1,:,:])**2 ),
                                     dim=0))
    delta_eta = torch.sqrt(torch.cat(( (X[:,1:2,:] - X[:,:1,:])**2 + (Y[:,1:2,:] - Y[:,:1,:])**2 + (Z[:,1:2,:] - Z[:,:1,:])**2,
                                       (0.5*(X[:,2:,:] - X[:,:-2,:]))**2 + (0.5*(Y[:,2:,:] - Y[:,:-2,:]))**2 + (0.5*(Z[:,2:,:] - Z[:,:-2,:]))**2,
                                       (X[:,-1:,:] - X[:,-2:-1,:])**2 + (Y[:,-1:,:] - Y[:,-2:-1,:])**2 + (Z[:,-1:,:] - Z[:,-2:-1,:])**2 ),
                                     dim=1))
    delta_z   = torch.sqrt(torch.cat(( (X[:,:,1:2] - X[:,:,:1])**2 + (Y[:,:,1:2] - Y[:,:,:1])**2 + (Z[:,:,1:2] - Z[:,:,:1])**2,
                                       (0.5*(X[:,:,2:] - X[:,:,:-2]))**2 + (0.5*(Y[:,:,2:] - Y[:,:,:-2]))**2 + (0.5*(Z[:,:,2:] - Z[:,:,:-2]))**2,
                                       (X[:,:,-1:] - X[:,:,-2:-1])**2 + (Y[:,:,-1:] - Y[:,:,-2:-1])**2 + (Z[:,:,-1:] - Z[:,:,-2:-1])**2 ),
                                     dim=2))
    return delta_xi,delta_eta,delta_z


# ------------------------------------------------------
# Weighted physical-space directional grid spacing
#       Returns squared magnitudes to reduce operation count -- NEEDS BOUNDARY MODIFICATIONS FOR MPI
#       Can save delta values to further reduce op counts
# ------------------------------------------------------
def get_deltas_weighted(grid,w_x,w_y,w_z=None):
    if (grid.ndim==2):
        return get_deltas_weighted_2D(grid.X,grid.Y,w_x,w_y)
    elif (grid.ndim==3):
        return get_deltas_weighted_3D(grid.X,grid.Y,grid.Z,w_x,w_y,w_z)
    else:
        raise Exception('Grid.get_deltas_weighted: not yet implemented for 1D')

    
def get_deltas_weighted_2D(X,Y,w_x,w_y):
    delta_xi2  = torch.cat(( ((X[1:2,:,None] - X[:1,:,None])*w_x[:1,:,:])**2 +
                ((Y[1:2,:,None] - Y[:1,:,None])*w_y[:1,:,:])**2,
                (0.5*(X[2:,:,None] - X[:-2,:,None])*w_x[1:-1,:,:])**2 +
                (0.5*(Y[2:,:,None] - Y[:-2,:,None])*w_y[1:-1,:,:])**2,
                ((X[-1:,:,None] - X[-2:-1,:,None])*w_x[-1:,:,:])**2 +
                ((Y[-1:,:,None] - Y[-2:-1,:,None])*w_y[-1:,:,:])**2 ),
                dim=0)
    delta_eta2 = torch.cat(( ((X[:,1:2,None] - X[:,:1,None])*w_x[:,:1,:])**2 +
                ((Y[:,1:2,None] - Y[:,:1,None])*w_y[:,:1,:])**2,
                (0.5*(X[:,2:,None] - X[:,:-2,None])*w_x[:,1:-1,:])**2 +
                (0.5*(Y[:,2:,None] - Y[:,:-2,None])*w_y[:,1:-1,:])**2,
                ((X[:,-1:,None] - X[:,-2:-1,None])*w_x[:,-1:,:])**2 +
                ((Y[:,-1:,None] - Y[:,-2:-1,None])*w_y[:,-1:,:])**2 ),
                dim=1)
    return delta_xi2,delta_eta2

    
def get_deltas_weighted_3D(X,Y,Z,w_x,w_y,w_z):
    delta_xi2  = torch.cat(( ((X[1:2,:,:] - X[:1,:,:])*w_x[:1,:,:])**2 +
                ((Y[1:2,:,:] - Y[:1,:,:])*w_y[:1,:,:])**2 +
                ((Z[1:2,:,:] - Z[:1,:,:])*w_z[:1,:,:])**2,
                (0.5*(X[2:,:,:] - X[:-2,:,:])*w_x[1:-1,:,:])**2 +
                (0.5*(Y[2:,:,:] - Y[:-2,:,:])*w_y[1:-1,:,:])**2 +
                (0.5*(Z[2:,:,:] - Z[:-2,:,:])*w_z[1:-1,:,:])**2,
                ((X[-1:,:,:] - X[-2:-1,:,:])*w_x[-1:,:,:])**2 +
                ((Y[-1:,:,:] - Y[-2:-1,:,:])*w_y[-1:,:,:])**2 +
                ((Z[-1:,:,:] - Z[-2:-1,:,:])*w_z[-1:,:,:])**2 ),
                dim=0)
    delta_eta2 = torch.cat(( ((X[:,1:2,:] - X[:,:1,:])*w_x[:,:1,:])**2 +
                ((Y[:,1:2,:] - Y[:,:1,:])*w_y[:,:1,:])**2 +
                ((Z[:,1:2,:] - Z[:,:1,:])*w_z[:,:1,:])**2,
                (0.5*(X[:,2:,:] - X[:,:-2,:])*w_x[:,1:-1,:])**2 +
                (0.5*(Y[:,2:,:] - Y[:,:-2,:])*w_y[:,1:-1,:])**2 +
                (0.5*(Z[:,2:,:] - Z[:,:-2,:])*w_z[:,1:-1,:])**2,
                ((X[:,-1:,:] - X[:,-2:-1,:])*w_x[:,-1:,:])**2 +
                ((Y[:,-1:,:] - Y[:,-2:-1,:])*w_y[:,-1:,:])**2 +
                ((Z[:,-1:,:] - Z[:,-2:-1,:])*w_z[:,-1:,:])**2 ),
                dim=1)
    delta_z2   = torch.cat(( ((X[:,:,1:2] - X[:,:,:1])*w_x[:,:,:1])**2 +
                ((Y[:,:,1:2] - Y[:,:,:1])*w_y[:,:,:1])**2 +
                ((Z[:,:,1:2] - Z[:,:,:1])*w_z[:,:,:1])**2,
                (0.5*(X[:,:,2:] - X[:,:,:-2])*w_x[:,:,1:-1])**2 +
                (0.5*(Y[:,:,2:] - Y[:,:,:-2])*w_y[:,:,1:-1])**2 +
                (0.5*(Z[:,:,2:] - Z[:,:,:-2])*w_z[:,:,1:-1])**2,
                ((X[:,:,-1:] - X[:,:,-2:-1])*w_x[:,:,-1:])**2 +
                ((Y[:,:,-1:] - Y[:,:,-2:-1])*w_y[:,:,-1:])**2 +
                ((Z[:,:,-1:] - Z[:,:,-2:-1])*w_z[:,:,-1:])**2 ),
                dim=2)
    return delta_xi2,delta_eta2,delta_z2

# ------------------------------------------------------
# Set up absorbing layer activation functions
# ------------------------------------------------------
def get_absorbing_layers(grid,cfg,decomp):
    one = torch.ones((1,), dtype=decomp.WP).to(cfg.device)
    
    if (grid.BC_eta_bot=='farfield' and decomp.jproc==0):
        # -y
        if (grid.ndim==3):
            dist = torch.minimum( grid.Y[:,:,0]/cfg.BC_thickness, one )
        else:
            dist = torch.minimum( grid.Y/cfg.BC_thickness, one )
        grid.sigma_BC_bot = torch.zeros((decomp.nx_,decomp.ny_), dtype=decomp.WP).to(cfg.device)
        grid.sigma_BC_bot[:,:] = cfg.BC_strength * (1.0 - dist)**cfg.BC_order
        
    if (grid.BC_eta_top=='farfield' and decomp.jproc==decomp.npy-1):
        # +y
        if (grid.ndim==3):
            dist = torch.minimum( (grid.Lx2 - grid.Y[:,:,0])/cfg.BC_thickness, one )
        else:
            dist = torch.minimum( (grid.Lx2 - grid.Y)/cfg.BC_thickness, one )
        grid.sigma_BC_top = torch.zeros((decomp.nx_,decomp.ny_), dtype=decomp.WP).to(cfg.device)
        grid.sigma_BC_top[:,:] = cfg.BC_strength * (1.0 - dist)**cfg.BC_order
        
    
    
       
    if (not grid.periodic_xi and decomp.iproc==0):
        #-x
        if (grid.ndim==3):
            dist = torch.minimum( grid.X[:,:,0]/cfg.BC_thickness, one )
        else:
            dist = torch.minimum( grid.X/cfg.BC_thickness, one )
    
        if ((cfg.IC_opt!= "planar_jet_spatial_lam") and (cfg.IC_opt!= "planar_jet_spatial_turb") and (cfg.IC_opt!= "planar_jet_spatial_RANS")):
            grid.sigma_BC_left  = torch.zeros((decomp.nx_,decomp.ny_), dtype=decomp.WP).to(cfg.device)
            grid.sigma_BC_left[:,:] = cfg.BC_strength * (1.0 - dist)**cfg.BC_order
        else:
            grid.sigma_BC_left = None 

    
    if (not grid.periodic_xi and decomp.iproc==decomp.npx-1):
        # +x
        if hasattr(cfg, 'BC_thickness_right'):
            if (grid.ndim==3):
                dist = torch.minimum( (grid.Lx1 - grid.X[:,:,0])/cfg.BC_thickness_right, one )
            else:
                dist = torch.minimum( (grid.Lx1 - grid.X)/cfg.BC_thickness_right, one )
    
        else:
            if (grid.ndim==3):
                dist = torch.minimum( (grid.Lx1 - grid.X[:,:,0])/cfg.BC_thickness, one )
            else:
                dist = torch.minimum( (grid.Lx1 - grid.X)/cfg.BC_thickness, one )       
        grid.sigma_BC_right = torch.zeros((decomp.nx_,decomp.ny_), dtype=decomp.WP).to(cfg.device)
        grid.sigma_BC_right[:,:] = cfg.BC_strength * (1.0 - dist)**cfg.BC_order
    
    return


    
#=======================================================
# CASE-SPECIFIC GRID GENERATORS
#=======================================================


# ------------------------------------------------------
# Basic uniform grid
#       x: periodic or non-periodic (farfield)
#       y_min: farfield, wall, periodic
#       y_max: farfield, wall, periodic
# ------------------------------------------------------
class uniform:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,periodic_xi,
                 BC_eta_top,BC_eta_bot):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,Lx1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi
        y = eta
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = 1.0 * torch.ones_like(eta)

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac

    
# ------------------------------------------------------
# Basic uniform grid -- d\xi = d\eta = 1
#       x: periodic or non-periodic (farfield)
#       y_min: farfield, wall, periodic
#       y_max: farfield, wall, periodic
# ------------------------------------------------------
class uniform_unit_spacing:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,periodic_xi,
                 BC_eta_top,BC_eta_bot):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

        # Uniform computational grid
        self.xi_grid  = torch.linspace(0,Nx1,Nx1+1).to(device)
        self.eta_grid = torch.linspace(0,Nx2,Nx2+1).to(device)

        self.d_xi  = 1.0
        self.d_eta = 1.0

        self.dx = self.Lx1 / (self.Nx1)
        self.dy = self.Lx2 / (self.Nx2)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi  * self.dx
        y = eta * self.dy
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)  * self.dx
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = 1.0 * torch.ones_like(eta) * self.dy

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac

# ------------------------------------------------------
# Uniform grid with wavy perturbation
# ------------------------------------------------------
class uniform_sine:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,periodic_xi,
                 BC_eta_top,BC_eta_bot):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Wave parameters
        self.wave_start = 2.0
        self.wave_end   = 4.0
        self.wave_amp   = 0.125
        self.wave_freq  = 0.5*np.pi

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False

        # Uniform computational grid
        #   xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,Lx1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi
        ones = torch.ones_like(x)
        y = eta + ( torch.heaviside(xi-self.wave_start, 0*ones) *
                    torch.heaviside(self.wave_end-xi, 0*ones) *
                    self.wave_amp*torch.sin(self.wave_freq * xi) )
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        ones = torch.ones_like(xi)
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi) + ( torch.heaviside(xi-self.wave_start, 0*ones) *
                                              torch.heaviside(self.wave_end-xi, 0*ones) *
                                              self.wave_amp*self.wave_freq*torch.cos(self.wave_freq*xi) )
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = 1.0 * torch.ones_like(eta)

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac


# ------------------------------------------------------
# 2D/3D Channel-flow grid with tanh stretching near walls
#       x: periodic
#       y_min: wall
#       y_max: wall
#       z: periodic
# ------------------------------------------------------
class channel:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,sy=1.0):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.sy  = sy  # y-stretching parameter

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'wall'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,Lx1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi

        ytilde = 2.0 * eta/self.Lx2 - 1.0
        y = 0.5 * self.Lx2 * (torch.tanh( self.sy * ytilde )/np.tanh( self.sy ))
        
        return x,y

    def coth(self,z):
        return (np.exp(2.0*z) + 1.0)/(np.exp(2.0*z) - 1.0)

    def sech(self,z):
        return 2.0/(torch.exp(z) + torch.exp(-z))

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = self.sy * self.coth(self.sy) * (self.sech(self.sy*(2.0*eta/self.Lx2 - 1.0)))**2

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac


# ------------------------------------------------------
# 1D-y Channel-flow grid with tanh stretching near walls
#       y_min: wall
#       y_max: wall
# ------------------------------------------------------
class channel_1Dy:
    def __init__(self,device,Nx2,Lx2,sy=1.0):

        # Save values needed for member functions
        self.Lx1 = 0.0
        self.Lx2 = Lx2
        self.Lx3 = 0.0
        self.Nx1 = 1
        self.Nx2 = Nx2
        self.Nx3 = 1
        self.sy  = sy  # y-stretching parameter

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'wall'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        self.xi_grid  = torch.DoubleTensor((0.0,)).to(device)
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)   # eta = y

        self.d_xi  = 0.0
        self.d_eta = Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi

        ytilde = 2.0 * eta/self.Lx2 - 1.0
        y = 0.5 * self.Lx2 * (torch.tanh( self.sy * ytilde )/np.tanh( self.sy ))
        
        return x,y

    def coth(self,z):
        return (np.exp(2.0*z) + 1.0)/(np.exp(2.0*z) - 1.0)

    def sech(self,z):
        return 2.0/(torch.exp(z) + torch.exp(-z))

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = self.sy * self.coth(self.sy) * (self.sech(self.sy*(2.0*eta/self.Lx2 - 1.0)))**2

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac


# ------------------------------------------------------
# Sinh() refinement in y-direction
#       x: periodic
#       y_min: farfield, wall, periodic
#       y_max: farfield, wall, periodic
# ------------------------------------------------------
class sinh_y:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,sy,
                 BC_eta_top,BC_eta_bot):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Polynomial stretching order
        self.sy = sy

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace( 0.0,1.0,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(-0.5,0.5,Nx2).to(device)   # eta = y

        self.d_xi  = 1.0/float(Nx1-1)
        self.d_eta = 1.0/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = self.Lx1 * xi
        
        # sinh stretching in y
        y = 0.5*self.Lx2*torch.sinh( 2.0*self.sy*eta )/np.sinh(self.sy)
        
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = self.Lx1 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = self.Lx2 * self.sy * (torch.exp(-2.0*self.sy*eta) + torch.exp(2.0*self.sy*eta)) / \
            (np.exp(self.sy) - np.exp(-self.sy))

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac    

    
# ------------------------------------------------------
#stretching in y direction using tanh function according to link below
# Ref https://www.cfd-online.com/Wiki/Structured_mesh_generation
#delta is the stretching variable and can be adjusted
# ------------------------------------------------------
class tanh:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,periodic_xi,delta,
                 BC_eta_top,BC_eta_bot):

        self.delta=delta
         # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

       
        

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,Lx1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Nx2,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = self.eta_grid[1:2] - self.eta_grid[0:1]

        return

    def get_xy(self,xi,eta):
       
        
        
        num=torch.tanh(self.delta*(eta/self.Nx2-0.5))
        den=torch.tanh(torch.tensor(self.delta*0.5))
        
       
        x =  xi  #keeping it uniform
        y = (1+(num/den))*0.5*self.Lx2 
        
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        x_eta  = 0.0 * torch.ones_like(eta)
       
        y_xi  = 0.0 * torch.ones_like(eta)
        y_eta =(self.delta*0.5/self.Nx2)*(1-torch.tanh(self.delta*(eta/self.Nx2-0.5))*torch.tanh(self.delta*(eta/self.Nx2-0.5))) \
            *(1/torch.tanh(torch.tensor(self.delta*0.5)))
       
        
        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac

# ------------------------------------------------------
# Used for dense grid at then centre e.g Jets. 
# 
# ------------------------------------------------------
class sinh:
    def __init__(self,device,Nx1,Nx2,Nx3,
                 Lx1,Lx2,Lx3,periodic_xi,delta_x,delta_y,
                 BC_eta_top,BC_eta_bot):

        self.delta_x = delta_x  # higher delta gives finer grid at the centre
        self.delta_y = delta_y 
         # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

        # Uniform computational grid
        self.xi_grid  = torch.linspace(0,Nx1,Nx1).to(device)   #
        self.eta_grid = torch.linspace(0,Nx2,Nx2).to(device)   # input in sinh function is this

        self.d_xi  = self.xi_grid[1:2]-self.xi_grid[0:1]
        self.d_eta = self.eta_grid[1:2]-self.eta_grid[0:1]       

        return

    def get_xy(self,xi,eta):
        num_y = torch.sinh(self.delta_y*(eta/self.Nx2 - 0.5))
        den_y = torch.sinh(torch.tensor(self.delta_y * 0.5))

        num_x = torch.sinh(self.delta_x*(xi/self.Nx1)) #delta_x = 0.0001 for uniform mesh
        den_x = torch.sinh(torch.tensor(self.delta_x)) #half sinh in x, exponentially increasing mesh
        
        x =  (((num_x/den_x))) * self.Lx1 
        y = (1 + (num_y/den_y)) * self.Lx2 * 0.5 
        
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = (self.delta_x * self.Lx1 /(self.Nx1*torch.sinh(torch.tensor(self.delta_x)))) * \
            (torch.cosh(self.delta_x*(xi/self.Nx1)) )
        x_eta  = 0.0 * torch.ones_like(eta)
        y_xi  = 0.0 * torch.ones_like(eta)
        y_eta = (self.delta_y * self.Lx2 * 0.5/(self.Nx2*torch.sinh(torch.tensor(self.delta_y * 0.5)))) * \
            (torch.cosh(self.delta_y*(-eta/self.Nx2 + 0.5)) )
        # Have tested this on laminar planar jets
        
        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac

# ------------------------------------------------------
# VG Wavy grid
#       x: periodic or non-periodic (farfield)
#       y_min: farfield, wall, periodic
#       y_max: farfield, wall, periodic
# ------------------------------------------------------
class VGwavy:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,periodic_xi,
                 BC_eta_top,BC_eta_bot):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3

        # Boundary conditions
        self.periodic_xi = periodic_xi
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        self.periodic_eta = False
        if (BC_eta_top=='periodic' and BC_eta_bot=='periodic'):
            self.periodic_eta = True

        # VGwavy parameters
        self.dx0 = Lx1/float(Nx1-1)
        self.dy0 = Lx2/float(Nx2-1)
        self.Ax  = 0.4/self.dx0
        self.Ay  = 0.6/self.dy0
        self.nx  = 2
        self.ny  = 4
        #print(self.dx0, self.dy0, self.dx0*self.dy0)
        #print(1/self.dx0, 1/self.dy0, 1.0/(self.dx0*self.dy0))

        # Uniform computational grid
        self.xi_grid  = torch.linspace(0,Nx1,Nx1+1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Nx2,Nx2+1).to(device)   # eta = y 

        self.d_xi  = 1.0
        self.d_eta = 1.0

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = self.dx0*( (xi) + self.Ax*torch.sin(self.nx*np.pi*(eta)*self.dy0/self.Lx2) )
        y = self.dy0*((eta) + self.Ay*torch.sin(self.ny*np.pi*( xi)*self.dx0/self.Lx1) )
        return x,y

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = torch.ones_like(xi)  * self.dx0
        y_eta = torch.ones_like(eta) * self.dy0

        x_eta = self.dx0*self.Ax*self.nx*np.pi*self.dy0/self.Lx2 * \
            torch.cos(self.nx*np.pi*(eta)*self.dy0/self.Lx2)
        y_xi  = self.dy0*self.Ay*self.ny*np.pi*self.dx0/self.Lx1 * \
            torch.cos(self.ny*np.pi*( xi)*self.dx0/self.Lx1)
        
        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac


# ------------------------------------------------------
# Channel-flow grid with tanh stretching near walls
#       x: periodic
#       y_min: wall
#       y_max: wall
# ------------------------------------------------------
class channel:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,sy=1.0):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.sy  = sy  # y-stretching parameter

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'wall'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,Lx1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        x = xi

        ytilde = 2.0 * eta/self.Lx2 - 1.0
        y = 0.5 * self.Lx2 * (torch.tanh( self.sy * ytilde )/np.tanh( self.sy ))
        
        return x,y

    def coth(self,z):
        return (np.exp(2.0*z) + 1.0)/(np.exp(2.0*z) - 1.0)

    def sech(self,z):
        return 2.0/(torch.exp(z) + torch.exp(-z))

    # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = self.sy * self.coth(self.sy) * (self.sech(self.sy*(2.0*eta/self.Lx2 - 1.0)))**2

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac


# ------------------------------------------------------
# Single-ramp oblique shock geometry
#       +/- x: farfield
#       y_min: wall
#       y_max: wall, farfield
# ------------------------------------------------------
class single_ramp:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,delta,
                 stretched=False,sx=1.0,sy=1.0):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.delta = delta
        self.stretched = stretched
        self.sx = sx
        self.sy = sy

        # Boundary conditions
        self.periodic_xi = False
        self.BC_eta_top = 'wall' #'farfield' #
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        self.xi_grid  = torch.linspace(-Lx1/2,Lx1/2,Nx1).to(device)
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    def get_xy(self,xi,eta):
        if (self.stretched):
            x,y = self.get_xy_stretched(xi,eta,self.sx,self.sy)
        else:
            x,y = self.get_xy_uniform(xi,eta)
        return x,y

    # Definition of curvilinear x,y coordinates
    #       Uniform in x and y
    def get_xy_uniform(self,xi,eta):
        alpha = 1.0
        x = xi
        y = eta + ( alpha * (torch.exp( torch.min(torch.zeros_like(xi),xi) ) - 1.0) +
                    torch.max(torch.zeros_like(xi),xi) + 1.0 )*np.tan(np.radians(self.delta)) \
                    * (1.0 - eta/self.Lx2)
        return x,y

    # Definition of curvilinear x,y coordinates
    #       Stretched in y
    def get_xy_stretched(self,xi,eta,sx,sy):
        alpha = 1.0
        ramp_end_x = 15

        # sinh stretching in x
        x = 0.5*(self.Lx1*torch.sinh( sx*(2.0*xi/self.Lx1) )/np.sinh(sx) ) + 10
        
        # y
        if (self.BC_eta_top=='wall'):
            # Tanh stretching at top and bottom
            y = ( 0.5*(self.Lx2*torch.tanh( sy*(2.0*eta/self.Lx2 - 1.0) )/np.tanh(sy) + self.Lx2) +
                  ( alpha * (torch.exp( torch.min(torch.zeros_like(x),x) ) - 1.0) + 1.0 +
                    torch.max(torch.zeros_like(x),x) * torch.heaviside(ramp_end_x-x,torch.zeros_like(x))
                    + ramp_end_x * torch.heaviside(x-ramp_end_x,torch.zeros_like(x))
                    - alpha * (torch.exp( torch.min(torch.zeros_like(x),x-ramp_end_x) ) - 1.0) + 1.0
                  )*np.tan(np.radians(self.delta)) * (1.0 - eta/self.Lx2))
        else:
            # Tanh stretching at bottom only
            y = ( self.Lx2*torch.tanh( sy*(eta/self.Lx2 - 1.0) )/np.tanh(sy) + self.Lx2 +
                  ( alpha * (torch.exp( torch.min(torch.zeros_like(x),x) ) - 1.0) + 1.0 +
                    torch.max(torch.zeros_like(x),x) * torch.heaviside(ramp_end_x-x,torch.zeros_like(x))
                    + ramp_end_x * torch.heaviside(x-ramp_end_x,torch.zeros_like(x))
                    - alpha * (torch.exp( torch.min(torch.zeros_like(x),x-ramp_end_x) ) - 1.0) + 1.0
                  )*np.tan(np.radians(self.delta)) * (1.0 - eta/self.Lx2))
                  
        return x,y


# ------------------------------------------------------
# Wedge geometry
#       +/- x: farfield
#       y_min: wall
#       y_max: farfield
# ------------------------------------------------------
class wedge:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,delta,
                 sx=1.0,sy=1.0):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.delta = delta
        self.sx = sx
        self.sy = sy

        # Boundary conditions
        self.periodic_xi = False
        self.BC_eta_top = 'farfield'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        #self.xi_grid  = torch.linspace(-Lx1/2,Lx1/2,Nx1+1).to(device)
        self.xi_grid  = torch.linspace(0,Lx1/2,Nx1+1).to(device)
        self.eta_grid = torch.linspace(0,Lx2,Nx2).to(device)

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)

        return

    def get_xy(self,xi,eta):
        #x = torch.tanh(self.sx*xi)/np.tanh(self.sx)*xi - eta

        # Surface
        zeros = torch.zeros_like(xi)
        tan_delta = np.tan(np.radians(self.delta))
        Rc      = 0.5
        ell = Rc/tan_delta
        beta = 90 - self.delta
        beta_hat = np.radians(beta)*(xi/ell)
        x = ( torch.heaviside(ell - xi, zeros) * Rc*(1.0 - torch.cos(beta_hat)) +
              torch.heaviside(xi - ell, zeros) * (-(Rc/np.sin(np.radians(self.delta)) - Rc) +
                                                  xi * np.cos(np.radians(self.delta))) ) - eta
        
        print(x[:,0])
        #xp = torch.heaviside(ell - xi, torch.zeros_like(xi))
        #print(xp[:,0]*beta_hat[:,0])

        y = xi * np.tan(np.radians(self.delta))

        return x,y
        


# ------------------------------------------------------
# Cylinder "O"-type grid
#       Periodic boundary conditions in \xi
#       No-slip wall at -\eta
#       Farfield boundary at +\eta
# ------------------------------------------------------
class cylinder:
    def __init__(self,device,Nx1,Nx2,Nx3,R_min,R_max,sx,Uniform,Lx3=None,
                 WP=torch.float64):

        # Save values needed for member functions
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.R_min = R_min
        self.R_max = R_max
        self.sx = sx
        self.Uniform = Uniform
        self.Lx3 = Lx3
        self.WP = WP

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'farfield'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,2*np.pi,Nx1+1, dtype=self.WP).to(device)       # xi = theta
        self.d_xi  = 2*np.pi/float(Nx1)
        
        if Uniform:
            self.eta_grid = torch.linspace(R_min,R_max,Nx2, dtype=self.WP).to(device)       # eta = R
            self.d_eta = (R_max - R_min)/float(Nx2-1)
        else:
            self.eta_grid = torch.linspace(0,1,Nx2, dtype=self.WP).to(device)       # eta = R
            self.d_eta = 1.0/float(Nx2-1)
            self.h = R_max - R_min
        return


    def get_absorbing_layers(self,cfg,decomp):
        # Set up absorbing layer activation function
        #  Only at the "top" of eta (outer radius of physical domain)
        if True: # (decomp.jproc==decomp.npy-1):
            one = torch.ones((1,), dtype=self.WP).to(cfg.device)
            # +y
            if (self.ndim==3):
                rad = torch.sqrt( self.X[:,:,0]**2 + self.Y[:,:,0]**2 )
            else:
                rad = torch.sqrt( self.X**2 + self.Y**2 )

            # Find maximum radius
            rad_max_local = torch.max(rad).cpu()
            rad_max = torch.zeros((1,), dtype=self.WP)
            decomp.comm.Allreduce(rad_max_local, rad_max, op=MPI.MAX)
            rad_max = rad_max.to(cfg.device)
            
            # Cutoff
            dist = torch.minimum( rad_max - rad, one*cfg.BC_thickness )

            # Set sigma
            self.sigma_BC_top = torch.zeros((decomp.nx_,decomp.ny_), dtype=self.WP).to(cfg.device)
            self.sigma_BC_top[:,:] = cfg.BC_strength * (1.0 - dist/cfg.BC_thickness)**cfg.BC_order
        return

    def get_radius(self,eta):
        if self.Uniform:
            return eta
        else:
            # Tanh grid
            #return ( self.h * torch.tanh(self.sx * (eta-1.0)) / np.tanh(self.sx) +
            #         self.R_min + self.h )

            # Exp grid
            return ( self.h*( torch.exp(self.sx*eta) - 1.0 )/np.exp(self.sx) + self.R_min )

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):            
        r = self.get_radius(eta)
        x = r * torch.cos(xi)
        y = r * torch.sin(xi)
        return x,y

    def coth(self,z):
        return (np.exp(2.0*z) + 1.0)/(np.exp(2.0*z) - 1.0)

    def sech(self,z):
        return 2.0/(torch.exp(z) + torch.exp(-z))

    # Grid transform interface
    def get_transform(self,xi,eta):
        if (self.Uniform):
            return self.get_transform_uniform(xi,eta)
        else:
            return self.get_transform_stretched(xi,eta)

    #       Analytic derivatives of x,y with respect to xi,eta
    def get_transform_uniform(self,xi,eta):
        x_xi  = -eta * torch.sin(xi)
        y_xi  =  eta * torch.cos(xi)
        x_eta = torch.cos(xi)
        y_eta = torch.sin(xi)

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac

    def get_transform_stretched(self,xi,eta):
        r = self.get_radius(eta)
        x_xi  = -r * torch.sin(xi)
        y_xi  =  r * torch.cos(xi)

        # Tanh grid
        #x_eta = self.h*self.sx * self.coth(self.sx) * (self.sech(self.sx*(eta - 1.0)))**2 * torch.cos(xi)
        #y_eta = self.h*self.sx * self.coth(self.sx) * (self.sech(self.sx*(eta - 1.0)))**2 * torch.sin(xi)

        # Exp grid
        x_eta = self.h * self.sx * torch.exp( self.sx*(eta - 1.0) ) * torch.cos(xi)
        y_eta = self.h * self.sx * torch.exp( self.sx*(eta - 1.0) ) * torch.sin(xi)

        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac



# ------------------------------------------------------
# Simple parametric airfoil model -- "O"-type grid
#       From D. Ziemkiewicz, AIAA J. 55 (2017)
#       Periodic in \xi
#       No-slip wall at -\eta
#       Farfield boundary at +\eta
# ------------------------------------------------------
class airfoil:
    def __init__(self,device,Nx1,Nx2,Nx3,R_min,R_max,
                 B,T,P,C,E,R):

        # Save values needed for member functions
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.R_min = R_min
        self.R_max = R_max

        # Shape parameters
        self.B = B      # Base shape coefficient
        self.T = T      # Thickness/chord
        self.P = P      # Taper exponent
        self.C = C      # Camber/chord
        self.E = E      # Camber exponent
        self.R = R      # Reflex parameter

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'farfield' 
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid
        #       xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0.001,2*np.pi,Nx1+1).to(device)   # xi = theta
        self.eta_grid = torch.linspace(R_min,R_max,Nx2).to(device)       # eta = R

        self.d_xi  = 2*np.pi/float(Nx1)
        self.d_eta = (R_max - R_min)/float(Nx2-1)

        # Tanh grid spacing near center
        self.h = R_max - R_min
        
        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):

        # Airfoil surface
        X = ( 0.5 + 0.5*torch.abs(torch.cos(xi))**self.B / torch.cos(xi) )
        Y = ( 0.5*self.T*torch.abs(torch.sin(xi))**self.B / torch.sin(xi) * (1.0 - X**self.P) +
              self.C * torch.sin(X**self.E * np.pi) +
              self.R * torch.sin(2.0*X*np.pi) )

        # Surface normal vectors: (f,g) = (X,Y)
        secTerm = torch.abs(torch.cos(xi))**self.B / torch.cos(xi)
        Xp = (0.5 - 0.5*self.B)*torch.tan(xi) * secTerm
        Yp = ( np.pi*self.C*self.E*torch.cos(np.pi*(secTerm + 0.5)**self.E)*(0.5*secTerm + 0.5)**(self.E-1) *
               (0.5*torch.tan(xi)*secTerm - 0.5*self.B*torch.abs(torch.cos(xi))**(self.B-2) * torch.sin(xi) ) +
               0.5*self.B*self.T*torch.cos(xi)*torch.abs(torch.sin(xi))**(self.B-2) * 
               ( 1.0 - (0.5*secTerm + 0.5)**self.P ) -
               0.5*self.T*torch.abs(torch.sin(xi))**self.B/(torch.tan(xi)*torch.sin(xi)) *
               ( 1.0 - (0.5*secTerm + 0.5)**self.P ) -
               0.5*self.P*self.T*torch.abs(torch.sin(xi))**self.B/torch.sin(xi) *
               ( 0.5*secTerm + 0.5 )**(self.P-1) *
               ( 0.5*torch.tan(xi)*secTerm - 0.5*self.B*torch.sin(xi)*torch.abs(torch.cos(xi))**(self.B-2) ) +
               2.0*np.pi*self.R*torch.cos(2.0*np.pi*(0.5*secTerm + 0.5)) *
               ( 0.5*torch.tan(xi)*secTerm - 0.5*self.B*torch.sin(xi)*torch.abs(torch.cos(xi))**(self.B-2) ) )

        denom = torch.sqrt(Xp**2 + Yp**2)
        u1 = -Yp / denom
        u2 =  Xp / denom
        #theta = torch.atan( u2/u1 )

        x = X - eta * u1 #torch.cos(theta)
        y = Y - eta * u2 #torch.sin(theta)

        return x,y

#--------------------------------------------
# 2D Cone geometry 
# Uses algebraic grid generation method from R. KUMAR, 
# 'Elliptic Grid Generation for NACA0012 Airfoil', 2015
#
# Farfield at -\xi
# Farfield at +\xi
# No-slip wall at -\eta  
# Farfield at +\eta 
#--------------------------------------------
class cone_2D:
    def __init__(self,device,Nx1,Nx2,Nx3, BC_wall, BC_left, BC_top, BC_right,
                 xi_left = 0.0, xi_right = 1.0, eta_low = 0.0, eta_up = 1.0,
                 dist = 0.4, mu = 2.5,nu = 2.5):
        #set member parameters
        self.Nx1 = Nx1;
        self.Nx2 = Nx2;
        self.Nx3 = Nx3;        
        
        # To construct subdomains
        self.xi_left    = xi_left
        self.xi_right   = xi_right
        self.eta_low    = eta_low
        self.eta_up     = eta_up    
    
        #Ellipse: shape / grid parameters
        self.mu = mu; #scaling parameter in eta/tangential-direction
        self.nu = nu; #scaling parameter in xi/radial-direction
        
        self.dist = dist * self.get_R(torch.tensor([eta_up])).item(); #distance between lower and upper arc
        #parameters specifying the shape of the ellipse.
        #In our case a > b necessary to get a cone shape
        #characteristic length = radius of curvature. Defined as 1 / k
        # where curvature at leading edge = a / b^2
        self.a = 0.85075286; 
        self.b = self.a / 10;

        #Defines how large the elliptic part of the boundary is
        #Currently, only domains with symmetry around pi work,
        #as the symmetry is used in the tanh-scaling.
        self.d =  6 / 4 * np.pi; 
        self.c = 1 / 4 * np.pi; 
        #Ratio of angle domain at which the upper cut is set
        self.xi_cut = 3 / 8;
                
        #computational variables
        self.cutUp = [self.a * np.cos(self.c + self.d * self.xi_cut), self.b*  np.sin(self.c + self.d * self.xi_cut)];
        self.dCutUp = [- self.a * self.d * np.sin(self.c + self.d * self.xi_cut), 
                 self.b * self.d * np.cos(self.c + self.d * self.xi_cut)];
        self.cutLow = [ self.a * np.cos(self.c + self.d * (1 - self.xi_cut)),  self.b * np.sin(self.c + self.d * (1 - self.xi_cut))];
        self.dCutLow = [- self.a * self.d * np.sin(self.c + self.d * (1 - self.xi_cut)), 
                 self.b * self.d * np.cos(self.c + self.d * (1 - self.xi_cut))];
                
        #Bisection iteration limit
        self.bisIter = 10000;

        #Boundary conditions
        self.periodic_xi = False;
        self.BC_xi_left = BC_left;
        self.BC_xi_right = BC_right;
        self.BC_eta_top = BC_top;
        self.BC_eta_bot = BC_wall;
        self.periodic_eta = False;
        
        # Uniform computational grid


        self.xi_grid  = torch.linspace(xi_left,xi_right,Nx1 ).to(device);
        self.eta_grid = torch.linspace(0,1,Nx2 ).to(device);
        self.d_xi  = (xi_right - xi_left)/float(Nx1 - 1);
        self.d_eta = 1/float(Nx2 - 1);

        return

    # -----------
    # Projection of point (x,y) onto ellipse with parameters R * self.a , R * self.b
    # -----------
    
    def projToEllipse(self,x,y):
        
        if (self.ndim ==3):
            raise Exception('3-d is not yet implemented')
        else: 
            r = torch.sqrt(x**2 + y**2);
            if (not r.all()):
                raise Exception('Angle is not defined for points of zero radius');
            
            # Projection onto upper and lower linear wall to calculate theta
            thetaUp = (self.dCutUp[0] * (x - self.cutUp[0])+ self.dCutUp[1] * (y - self.cutUp[1])
                + self.xi_cut * (self.dCutUp[0]**2 + self.dCutUp[1]**2)) / (self.dCutUp[0]**2 + self.dCutUp[1]**2);
            thetaLow = (self.dCutLow[0] * (x - self.cutLow[0])+ self.dCutLow[1] * (y - self.cutLow[1])
                + (1 - self.xi_cut) * (self.dCutLow[0]**2 + self.dCutLow[1]**2)) / (self.dCutLow[0]**2 + self.dCutLow[1]**2);
            
            #Calculate reference point on elliptic part of the boundary.
            #For the calculation see David Eberly, Distance from a point
            #to an Ellipse, an Ellipsoid or a Hyperellipsoid

            #projecting point to first quadrant
            signY = torch.sign(y);
            absY = torch.mul(signY, y);
            signX = torch.sign(x);
            absX = torch.mul(signX, x);
            rA = self.a;
            rB = self.b;

            #Bisection method to find projection onto ellipse
            ratio = rA**2 / rB**2;
            s0 = -1 + absY/ rB;
            s1 = -1 + torch.where( (ratio**2 * (absX / rA)**2 >= (absY / rB)**2),
                    torch.abs(ratio * absX / rA) * torch.sqrt(1 + ( (absY / rB) / (ratio * absX /rA))**2),
                    torch.abs(absY / rB) * torch.sqrt(1 + ((ratio * absX / rA) / (absY /rB) )**2) );
            s = torch.zeros_like(s0, dtype = torch.float64);
            g = torch.zeros_like(s, dtype = torch.float64);
            
            for i in range(0, int(self.bisIter)):
                s = (s0 + s1)/2;
                if (torch.isclose(torch.minimum(s- s0, s1 - s), torch.zeros_like(s),rtol = 1e-13, atol = 1e-13 ).all()):
                    break;

                g = ((ratio * absX / rA) / (s + ratio))**2 + ( (absY / rB) / (s + 1))**2 - 1.0;
                if (torch.isclose(g, torch.zeros_like(g), rtol = 1e-13, atol = 1e-13 ).all() ):
                    break;
                s0 = torch.where( (g > 0), s, s0);
                s1 = torch.where( (g < 0), s, s1);
            t = rB**2 * s;
            x0p = rA**2 * absX /(t + rA**2);
            y0p = rB**2 * absY /(t + rB**2);
            x0 = torch.mul(signX, x0p);
            y0 = torch.mul(signY, y0p);
   
        return x0,y0

    # Inverse grid transform
    def get_xieta(self,x,y):
        
        rA = self.a
        rB = self.b

        # Projection onto upper and lower linear wall to calculate theta
        thetaUp  = (self.dCutUp[0] * (x - self.cutUp[0])+ self.dCutUp[1] * (y - self.cutUp[1])
                 + self.xi_cut * (self.dCutUp[0]**2 + self.dCutUp[1]**2)) / (self.dCutUp[0]**2 + self.dCutUp[1]**2);
        thetaLow = (self.dCutLow[0] * (x - self.cutLow[0])+ self.dCutLow[1] * (y - self.cutLow[1])
                 + (1 - self.xi_cut) * (self.dCutLow[0]**2 + self.dCutLow[1]**2)) / (self.dCutLow[0]**2 + self.dCutLow[1]**2);
        
        [x0,y0] = self.projToEllipse(x,y)

        theta_ellip = torch.where( (torch.isclose(y0 , torch.zeros_like(y0)) ), 0.5, 
                torch.where((y0 > 0), (torch.arccos(x0 / rA ) - self.c) /self.d, (2 * np.pi - torch.arccos(x0 / rA ) - self.c) /self.d) );
        phi = torch.where( torch.logical_and((thetaUp < self.xi_cut), (y > 0) ), thetaUp,
              torch.where( torch.logical_and( (thetaLow > 1 - self.xi_cut), (y < 0)), thetaLow, theta_ellip) )
        # Recover xi
        
        xi_fl = self.inv_tanh_stretching(phi)
        xi_ind = torch.round( (self.Nx1 - 1) * xi_fl)
        # Recover eta
        [innerX, innerY]   = self.get_innerArc(phi);
        [nInnerX, nInnerY] = self.get_normals_theta(phi)
        
        R = torch.where( (torch.isclose(y, 0.0 * y) ), (x - innerX) / (nInnerX * self.dist) , ( y - innerY) / (nInnerY * self.dist))
        
        eta_fl = 1 - torch.asin( (1 - R) * np.sin(self.nu * np.pi / 2)) / (self.nu * np.pi / 2)
        eta_ind = torch.round( (self.Nx2 - 1) * eta_fl)

        return xi_fl, eta_fl, xi_ind, eta_ind, R

    def get_absorbing_layers(self, cfg, decomp):
        one = torch.ones((1,), dtype = torch.float64).to(cfg.device);
        if (self.ndim ==3):
            raise Exception('3-d is not yet implemented')
        else: 
            r = torch.sqrt(self.X**2 + self.Y**2);
            if (not r.all()):
                raise Exception('Angle is not defined for points of zero radius');
            thetaUp = (self.dCutUp[0] * (self.X - self.cutUp[0])+ self.dCutUp[1] * (self.Y - self.cutUp[1])
                + self.xi_cut * (self.dCutUp[0]**2 + self.dCutUp[1]**2)) / (self.dCutUp[0]**2 + self.dCutUp[1]**2);
            thetaLow = (self.dCutLow[0] * (self.X - self.cutLow[0])+ self.dCutLow[1] * (self.Y - self.cutLow[1])
                + (1 - self.xi_cut) * (self.dCutLow[0]**2 + self.dCutLow[1]**2)) / (self.dCutLow[0]**2 + self.dCutLow[1]**2);
            
            if (self.BC_eta_top == 'farfield'and decomp.jproc == decomp.npy-1):                  
                
                xi,_,_,_,_ = self.get_xieta(self.X, self.Y)

                

                phi = self.tanh_stretching(xi)
                [x0,y0] = self.get_innerArc(phi)
                [n0X , n0Y] = self.get_normals_theta(phi);
                distEllip=       torch.sqrt( (x0 + self.dist *n0X -self.X)**2 + (y0 + self.dist * n0Y - self.Y)**2);
                                
                #distance to linear parts of cone
                [xUp, yUp] = self.get_innerArc(thetaUp);
                [nXUp, nYUp] = self.get_normals_theta(thetaUp);
                distUp = torch.sqrt( (xUp + self.dist * nXUp - self.X)**2 + (yUp + self.dist * nYUp - self.Y)**2);
                [xLow, yLow] = self.get_innerArc(thetaLow); 
                [nXLow, nYLow] = self.get_normals_theta(thetaLow);
                distLow = torch.sqrt( (xLow + self.dist * nXLow - self.X)**2 + (yLow + self.dist * nYLow - self.Y)**2);

                dist = torch.minimum(torch.where( torch.logical_and((thetaUp < self.xi_cut), (self.Y > 0) ) , distUp, 
                    torch.where( torch.logical_and((thetaLow > 1 - self.xi_cut), (self.Y < 0) ) , distLow, distEllip ) ) / cfg.BC_thickness, one);
                self.sigma_BC_top = torch.zeros((decomp.nx_,decomp.ny_), dtype = torch.float64).to(cfg.device)
                self.sigma_BC_top[:,:] =  cfg.BC_strength * (1.0 - dist) **cfg.BC_order;

            #Note: Distance calculation only if bdry is completely contained in linear part of the cone
            #Choosing linear part too small or bdry thickness too large breaks this
            if (not self.periodic_xi and decomp.iproc==0):
                [xUp, yUp] = self.get_innerArc(torch.tensor([self.xi_left]).to(cfg.device));
                [nXUp, nYUp] = self.get_normals_theta(torch.tensor([self.xi_left]).to(cfg.device));
                rUp = nXUp * (self.X - xUp) + nYUp *(self.Y - yUp)
                distUpp = torch.minimum( torch.sqrt( (xUp + rUp * nXUp - self.X)**2 + (yUp + rUp * nYUp - self.Y)**2) / cfg.BC_thickness, one);         
                
                distUp = torch.where( torch.logical_and((thetaUp < self.xi_cut), (self.Y > 0) ), distUpp, one); 
                self.sigma_BC_left = torch.zeros((decomp.nx_,decomp.ny_), dtype = torch.float64).to(cfg.device)
                self.sigma_BC_left[:,:] = cfg.BC_strength * (1.0 - distUp) **cfg.BC_order 
            if (not self.periodic_xi and decomp.iproc==decomp.npx-1):
                [xLow, yLow] = self.get_innerArc(torch.tensor([self.xi_right]).to(cfg.device));
                [nXLow, nYLow] = self.get_normals_theta(torch.tensor([self.xi_right]).to(cfg.device));
                rLow = nXLow * (self.X - xLow) + nYLow *(self.Y - yLow)
                distLowp = torch.minimum(torch.sqrt( (xLow +rLow * nXLow - self.X)**2 + (yLow +rLow*nYLow - self.Y)**2) / cfg.BC_thickness, one);
                
                distLow = torch.where( torch.logical_and((thetaLow > 1 - self.xi_cut), (self.Y < 0) ), distLowp, one);
                self.sigma_BC_right = torch.zeros((decomp.nx_,decomp.ny_), dtype = torch.float64).to(cfg.device)
                self.sigma_BC_right[:,:] = cfg.BC_strength * (1.0 - distLow) **cfg.BC_order
        return
    
    def tanh_stretching(self,xi):    
        x = 2 * (xi - 1.0 / 2.0)
        g = torch.where( (torch.isclose(torch.zeros_like(x), x) ),0 , torch.where( ( x > 0), 1 - torch.tanh(self.mu * (1 - x))/np.tanh(self.mu), 
                                                                                   -(1 - torch.tanh(self.mu * (1 + x))/np.tanh(self.mu)) ) );          
        theta = g / 2.0 + 1.0 / 2.0
        return theta
    def inv_tanh_stretching(self,theta):
        g = 2.0 * (theta - 1.0 / 2.0)
        x = torch.where( (torch.isclose(torch.zeros_like(g), g) ), 0, torch.where( (g >= 0), 1 - torch.atanh(np.tanh(self.mu) * (1 - g)) /
                                                                                   self.mu, torch.atanh(np.tanh(self.mu)* (1 + g)) / self.mu - 1) )
        xi = x / 2.0 + 1.0 / 2.0
        return xi

    def get_stretching_der(self, xi,eta):
        theta = 2.0 * (xi - 1.0 / 2.0)
        tanh_arg = torch.where( (theta>= 0), 1 - theta, 1 + theta)
        dxi_scaled_dxi   = - (1 - torch.tanh(self.mu * tanh_arg)**2) * self.mu / np.tanh(self.mu)

        deta_scaled_deta = self.dist * (1 - torch.tanh(self.mu * (1 -eta))**2 ) * self.mu / np.tanh(self.mu)
        Jac = dxi_scaled_dxi * deta_scaled_deta

        dxi_dxi_scaled   = deta_scaled_deta / Jac
        deta_deta_scaled = dxi_scaled_dxi   / Jac
        return dxi_dxi_scaled, deta_deta_scaled


    
    def get_innerArc(self,theta):   
        x = torch.where( (theta < self.xi_cut),  self.dCutUp[0] * (theta - self.xi_cut) + self.cutUp[0] ,
            torch.where( (theta > 1- self.xi_cut),  self.dCutLow[0] * (theta - 1 + self.xi_cut) + self.cutLow[0],
            self.a * torch.cos(self.c + theta * self.d) ) );
        y = torch.where( (theta < self.xi_cut),  self.dCutUp[1] * (theta - self.xi_cut) + self.cutUp[1],
              torch.where( (theta > 1 - self.xi_cut),  self.dCutLow[1] * (theta - 1 +self.xi_cut) + self.cutLow[1],
              self.b * torch.sin(self.c + theta * self.d) ) );
        return x,y

    def get_normals_theta(self,theta):
        #Normals: (dy , -dx), due to direction of theta
        #The factor theta_xi is neglected, it gets normalized
        nXp = torch.where( (theta < self.xi_cut), self.dCutUp[1],
              torch.where( (theta > 1 - self.xi_cut), self.dCutLow[1],
              self.b * self.d *torch.cos(self.c + self.d * theta) ) ) ;
        nYp = torch.where( (theta < self.xi_cut),- self.dCutUp[0],
              torch.where( (theta > 1 -self.xi_cut),-self.dCutLow[0],
              self.a * self.d *torch.sin(self.c + self.d * theta) ) );
        norm = torch.sqrt(nXp**2 + nYp**2);
        nX = nXp / norm;
        nY = nYp / norm;
        
        return nX,nY

    def get_normals(self, xi):
        #Normals: (dy , -dx), due to direction of xi
        nXp = torch.where( (xi < self.xi_cut), self.dCutUp[1],
              torch.where( (xi > 1 - self.xi_cut), self.dCutLow[1],
              self.b * self.d *torch.cos(self.c + self.d * xi) ) ) ;
        nYp = torch.where( (xi < self.xi_cut),- self.dCutUp[0],
              torch.where( (xi > 1 -self.xi_cut),-self.dCutLow[0],
              self.a * self.d *torch.sin(self.c + self.d * xi) ) );
        norm = torch.sqrt(nXp**2 + nYp**2);
        nX = nXp / norm;
        nY = nYp / norm;
        
        return nX,nY

    
    def get_R(self,eta):
        return 1 - torch.tanh(self.nu * (1 - eta)) / np.tanh(self.nu);
    
    def get_xy(self,xi,eta):

        #Stretched grid
        theta = self.tanh_stretching(xi);
        [xi0, yi0] = self.get_innerArc(theta);
        
        #Normals: (dy , -dx), due to direction of theta
        [nX, nY] = self.get_normals_theta(theta);
        
        R = self.get_R(eta)
        
        #Stretched grid
         
        #Inner grid points: Constructed using normals_theta
        x = xi0 + self.dist * R * nX 
        y = yi0 + self.dist * R * nY 
        
        return x,y
    

# ------------------------------------------------------
# Flow over a flat plate with rectilinear mesh no stretching
#   x_min: symmetric   
#   x_max: supersonic outflow
#   y_min: wall
#   y_max: supersonic outflow
# ------------------------------------------------------
class flat_plate:
    def __init__(self,device,Nx1,Nx2,Nx3,Lx1,Lx2,Lx3,wallLs=1.0,stretching=False,sx=1.0, sy=1.0, xLeftOpt=0.5, xRightOpt=1.0, yBotOpt=0.0, yTopOpt=0.35):

        # Save values needed for member functions
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.wallLs  = wallLs  
        
        # Stretching
        self.stretching = stretching
        self.sx = sx
        self.sy = sy

        # Boundary conditions
        self.periodic_xi = False
        self.BC_eta_top = 'supersonic'
        self.BC_eta_bot = 'wall'
        self.BC_xi_left  = 'symmetric'
        self.BC_xi_right = 'supersonic'
        self.periodic_eta = False

        # Uniform computational grid
        #   xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,1,Nx1).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,1,Nx2).to(device)   # eta = y

        self.d_xi  = Lx1/float(Nx1-1)
        self.d_eta = Lx2/float(Nx2-1)
        
        
        # Computing grid point at which wall should start 
        self.wallPoint = int((self.wallLs/self.Lx1) * self.Nx1)
        print(self.wallPoint)
        #print(self.lol)
        
        # Computing grid points at which to cut-off for optimization domain
        x_grid, y_grid = self.get_xy(self.xi_grid, self.eta_grid)
        
        # x-left 
        boolArr = (x_grid <= xLeftOpt)
        self.xIndOptLeft = torch.sum(boolArr)
        
        # x-right 
        boolArr = (x_grid >= xRightOpt)
        self.xIndOptRight = - torch.sum(boolArr)
        
        # y-bottom 
        boolArr = (y_grid <= yBotOpt)
        self.yIndOptBot   = torch.sum(boolArr)
        
        # y-top
        boolArr = (y_grid >= yTopOpt)
        self.yIndOptTop   = -torch.sum(boolArr)
    
        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        
        
        if not self.stretching:
            x = self.Lx1 * xi
            y = self.Lx2 * eta
            
        else:
            xi_tl = xi - xi[self.wallPoint]
            xi_neg = xi_tl[:self.wallPoint] / xi[self.wallPoint]
            xi_pos = xi_tl[self.wallPoint:] / (1 - xi[self.wallPoint])
            
            xi_neg = - (1 - torch.tanh(self.sx * (1 + xi_neg) ) / np.tanh(self.sx) )
            xi_pos = 1 - torch.tanh(self.sx * (1 - xi_pos) ) / np.tanh(self.sx) 
            
            xi_neg = xi_neg * xi[self.wallPoint]
            xi_pos = xi_pos * (1 - xi[self.wallPoint])
    
            x = self.Lx1 * (torch.cat((xi_neg, xi_pos), dim = 0) + xi[self.wallPoint])
            y = self.Lx2 * (1 - torch.tanh(self.sy * (1 - eta) )  / np.tanh(self.sy ) ) 

        
        return x,y

    # # Analytic derivatives of x,y with respect to xi,eta
    # def get_transform(self,xi,eta):
    #     x_xi  = self.Lx1 * torch.ones_like(xi)
    #     y_xi  = 0.0 * torch.ones_like(xi)
    #     x_eta = 0.0 * torch.ones_like(eta)
    #     y_eta = self.Lx2 * torch.ones_like(eta)
    #     # Grid Jacobian
    #     Jac = x_xi * y_eta - x_eta * y_xi

    #     return x_xi,y_xi,x_eta,y_eta,Jac



# ------------------------------------------------------
# Flow over a flat plate with rectilinear mesh no stretching
#   x_min: dirichlet (farfield with zero strenght)   
#   x_max: supersonic outflow
#   y_min: wall
#   y_max: farfield
# ------------------------------------------------------
class flat_plate_2:
    def __init__(self,device,Nx1,Nx2,Nx3,BC_wall, BC_left, BC_top, BC_right,
                 cut_ind, mu = 2.0, sy=2.0, xi_right = 1.0, scaling = True):

        # Save values needed for member functions
        self.Lx1 = 1.0
        self.Lx2 = 0.5
        self.Lx3 = 0.0
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.sy  = sy  # y-stretching parameter
        self.mu = mu
        self.cut_ind = cut_ind# Parameter for change from Neumann to wall BC
        self.scaling = scaling

        # Boundary conditions
        self.periodic_xi = False
        self.BC_eta_top = BC_top
        self.BC_eta_bot = BC_wall
        self.BC_xi_left  = BC_left
        self.BC_xi_right = BC_right
        self.periodic_eta = False

        # Uniform computational grid
        #   xi-grid is periodic, so Nx1+1 point is redundant
        self.xi_grid  = torch.linspace(0,xi_right,Nx1 ).to(device)   # xi = x
        self.eta_grid = torch.linspace(0,1.0,Nx2 ).to(device)   # eta = y

        self.d_xi  = self.Lx1 * xi_right/float(Nx1-1)
        self.d_eta = self.Lx2/float(Nx2-1)

        return

    # Definition of curvilinear x,y coordinates
    def get_xy(self,xi,eta):
        if self.scaling == False:
            x = self.Lx1 * xi
            y = self.Lx2 * eta
        else:
            xi_tl = xi - xi[self.cut_ind]
            xi_neg = xi_tl[:self.cut_ind] / xi[self.cut_ind]
            xi_pos = xi_tl[self.cut_ind:] / (1 - xi[self.cut_ind])

            xi_neg = - (1 - torch.tanh(self.mu * (1 + xi_neg) ) / np.tanh(self.mu) )
            xi_pos = 1 - torch.tanh(self.mu * (1 - xi_pos) ) / np.tanh(self.mu) 

            xi_neg = xi_neg * xi[self.cut_ind]
            xi_pos = xi_pos * (1 - xi[self.cut_ind])

            x = torch.cat((xi_neg, xi_pos), dim = 0) + xi[self.cut_ind]
            y = self.Lx2 * (1 - torch.tanh(self.sy * (1 - eta) )  / np.tanh(self.sy ) ) 
        
        return x,y

    def get_normals(self,xi):
        ones = torch.ones_like(xi)
        zeros = torch.zeros_like(xi)
        return zeros, ones


    """ # Analytic derivatives of x,y with respect to xi,eta
    def get_transform(self,xi,eta):
        x_xi  = 1.0 * torch.ones_like(xi)
        y_xi  = 0.0 * torch.ones_like(xi)
        x_eta = 0.0 * torch.ones_like(eta)
        y_eta = 1.0 * torch.ones_like(eta)
        # Grid Jacobian
        Jac = x_xi * y_eta - x_eta * y_xi

        return x_xi,y_xi,x_eta,y_eta,Jac """


class airfoil_dlsgs:
    def __init__(self, device, Nx1, Nx2, Nx3, grid_name, L, span=None):
        # Save values needed for member functions
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3 if span is not None else 1
        self.dfName_grid = grid_name
        self.L = L
        self.Lx3 = span

        # Boundary conditions
        self.periodic_xi = True
        self.BC_eta_top = 'farfield'
        self.BC_eta_bot = 'wall'
        self.periodic_eta = False

        # Uniform computational grid - xi_grid is periodic, so the Nx1 + 1 point
        # is redundant.
        self.xi_grid = torch.linspace(0, 1, Nx1 + 1).to(device)
        self.d_xi = self.xi_grid[1] - self.xi_grid[0]
        self.eta_grid = torch.linspace(0, 1, Nx2).to(device)
        self.d_eta = self.eta_grid[1] - self.eta_grid[0]
        return

    def read_grid(self, decomp):
        X, Y = Data.read_grid(self.dfName_grid, decomp, 2)
        return X, Y

    def get_absorbing_layers(self, cfg, decomp):
        # Set up absorbing layer activation function
        #  Only at the "top" of eta (outer radius of physical domain)
        if (decomp.jproc == decomp.npy - 1):
            delta = cfg.BC_thickness if hasattr(cfg, "BC_thickness") else 1.0

            one = torch.ones((1,), dtype=torch.float64).to(cfg.device)
            # +y
            if (self.ndim == 3):
                dist = self.L - torch.sqrt(
                    (self.X[:, :, 0] - 0.5)**2 + self.Y[:, :, 0]**2)
            else:
                dist = self.L - torch.sqrt((self.X - 0.5)**2 + self.Y**2)

            # Cutoff
            dist = torch.minimum(dist / delta, one)

            # Set sigma
            self.sigma_BC_top = torch.zeros(
                (decomp.nx_, decomp.ny_),
                dtype=torch.float64).to(cfg.device)
            self.sigma_BC_top[:, :] = (
                cfg.BC_strength * (1.0 - dist) ** cfg.BC_order)
        return


# ------------------------------------------------------
# Grid read in from HDF5 file
# ------------------------------------------------------
class readFromFile:
    def __init__(self,
                 device,
                 dfName_grid, 
                 Nx1, Nx2, Nx3, 
                 Lx1, Lx2, Lx3,
                 ndim, 
                 periodic_xi, 
                 periodic_eta,
                 BC_eta_top,
                 BC_eta_bot,
                 ):        
        self.dfName_grid = dfName_grid
        
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        
        self.Lx1 = Lx1
        self.Lx2 = Lx2
        self.Lx3 = Lx3
        
        self.ndim = ndim
        
        self.periodic_xi = periodic_xi
        self.periodic_eta = periodic_eta
        
        self.BC_eta_top = BC_eta_top
        self.BC_eta_bot = BC_eta_bot
        
        # Read-in period grids are already the correct size, but 
        # `enforce_periodic` removes the end of xi_grid/eta_grid. So, they are
        # made 1 cell longer to compensate for what will be removed.
        self.xi_grid  = torch.linspace(0, Lx1,
                                       Nx1+self.periodic_xi).to(device)
        self.eta_grid = torch.linspace(0, Lx2,
                                       Nx2+self.periodic_eta).to(device)
        
        self.d_xi = self.xi_grid[1] - self.xi_grid[0]
        self.d_eta = self.eta_grid[1] - self.eta_grid[0]

        
    def read_grid(self, decomp, cfg):
        return Data.read_grid(self.dfName_grid, cfg, decomp, self.ndim)
        
