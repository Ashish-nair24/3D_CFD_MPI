"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Boundary.py
@author Jonathan F. MacArt

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

# ---------------------------------------------------------------
# Set target solutions for absorbing layers
#   If "names" is supplied, then this will only update those vars
# ---------------------------------------------------------------
def update_farfield_targets(Q, param, grid, cfg, names=None):

    # Default: Update targets for all conserved variables
    if names is None:
        names = Q.names
        
    # Bottom - eta
    if (grid.BC_eta_bot=='farfield' and param.jproc==0):
        for var in names:
            param.Q_BC_bot[Q.names.index(var),:] = Q[var].interior()[:,0,0]
        if cfg.IC_opt == 'airfoil_dlsgs':
            AoA = torch.tensor(grid.AoA) if hasattr(grid, 'AoA') else torch.tensor(0.0)
            param.Q_BC_top[1, :] = (Q['rho'].interior()[:,-1,0] * 
                                    torch.cos(AoA * torch.pi / 180.0) * EOS.U0)
            param.Q_BC_top[2, :] = (Q['rho'].interior()[:,-1,0] *
                                    torch.sin(AoA * torch.pi / 180.0) * EOS.U0)

    # Top - eta
    if (grid.BC_eta_top=='farfield' and param.jproc==param.npy-1):
        for var in names:
            param.Q_BC_top[Q.names.index(var),:] = Q[var].interior()[:,-1,0]
        if cfg.IC_opt == 'airfoil_dlsgs':
            AoA = torch.tensor(grid.AoA) if hasattr(grid, 'AoA') else torch.tensor(0.0)
            param.Q_BC_top[1, :] = (Q['rho'].interior()[:,-1,0] *
                                    torch.cos(AoA * torch.pi / 180.0) * EOS.U0)
            param.Q_BC_top[2, :] = (Q['rho'].interior()[:,-1,0] *
                                    torch.sin(AoA * torch.pi / 180.0) * EOS.U0)
        
    if (not grid.periodic_xi):
        # Left - xi
        if (param.iproc==0):
            for var in names:
                param.Q_BC_left[Q.names.index(var),:] = Q[var].interior()[0,:,0]
            
            if  (cfg.IC_opt=="planar_jet_spatial_turb" or
                 cfg.IC_opt=="planar_jet_spatial_turb_coflow" or
                 cfg.IC_opt=="planar_jet_spatial_lam" or
                 cfg.IC_opt=="planar_jet_spatial_RANS" or
                 cfg.IC_opt=="planar_jet_spatial_RANS_coflow"):
                
                param.Q_BC_left[1,:] = Q["rhoU"].interior()[0,:,0]/Q["rho"].interior()[0,:,0]
                param.Q_BC_left[2,:] = Q["rhoV"].interior()[0,:,0]/Q["rho"].interior()[0,:,0]
                param.Q_BC_left[3,:] = Q["rhoW"].interior()[0,:,0]/Q["rho"].interior()[0,:,0]
                
                if (cfg.IC_opt=="planar_jet_spatial_RANS" or
                    cfg.IC_opt=="planar_jet_spatial_RANS_coflow"):
                    param.Q_BC_left[5,:] = Q["rhok"].interior()[0,:,0]/Q["rho"].interior()[0,:,0]
                    param.Q_BC_left[6,:] = Q["rhoeps"].interior()[0,:,0]/Q["rho"].interior()[0,:,0]

        # Right - xi
        if (param.iproc==param.npx-1):
            for var in names:
                param.Q_BC_right[Q.names.index(var),:] = Q[var].interior()[-1,:,0]

