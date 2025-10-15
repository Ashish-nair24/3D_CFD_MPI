"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file PyFlowCL.py
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


import numpy as np
import torch
import torch.optim as optim
import time
import inspect
import os
import copy
#import scipy.sparse.linalg as spla
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


from mpi4py import MPI

from . import Grid, Metrics, Data, Boundary, Bodyforce, Thermochemistry, Model
from . import Operator as op
from . import Initial_Conditions as IC
from .Library import Parallel
from .Monitor import Monitor, Statistics
from .RHS import RHS
from .Adjoint import Adjoint_RHS
from .Utilities import Conversion


# ----------------------------------------------------
# Lightweight class for running parameters
# ----------------------------------------------------
class Param:
    def __init__(self, cfg, decomp, EOS, WP, WP_np):
        self.WP = WP
        self.WP_np = WP_np
        
        # Grid size (entire domain)
        self.nx = decomp.nx
        self.ny = decomp.ny
        self.nz = decomp.nz
        
        self.nx_ = decomp.nx_
        self.ny_ = decomp.ny_
        self.nz_ = decomp.nz_
        
        self.nxo_ = decomp.nxo_
        self.nyo_ = decomp.nyo_

        # Parallel info
        self.device = cfg.device
        self.rank   = decomp.rank
        self.iproc  = decomp.iproc; self.npx = decomp.npx
        self.jproc  = decomp.jproc; self.npy = decomp.npy
        self.kproc  = decomp.kproc; self.npz = decomp.npz

        # Local interior indices
        self.imin_ = decomp.imin_; self.imax_ = decomp.imax_+1
        self.jmin_ = decomp.jmin_; self.jmax_ = decomp.jmax_+1
        self.kmin_ = decomp.kmin_; self.kmax_ = decomp.kmax_+1

        # Time step info
        self.dt     = cfg.dt
        self.max_dt = cfg.dt
        if hasattr(cfg,'max_CFL'): self.max_CFL = cfg.max_CFL
        else: self.max_CFL = 0.8 

        # Options needed for RHS
        self.artDiss = False if not hasattr(cfg, 'artDiss') else cfg.artDiss
        self.lapStab = False if not hasattr(cfg, 'lapStab') else cfg.lapStab
        
        # Adiabatic BC (only for rectininear meshes)
        if hasattr(cfg,'Adiabatic'):
            self.adiabatic = cfg.Adiabatic
        else:
            self.adiabatic = False

        # Advection scheme options
        self.upwind = False
        if hasattr(cfg,'advection_scheme'):
            self.advection_scheme = cfg.advection_scheme
            # advection_scheme options:
            #   'central_4th'
            #   'upwind_1st'
            #   'upwind_MUSCL'
            #   'upwind_StegerWarming'
            if 'upwind' in self.advection_scheme:
                self.upwind = True
        else:
            self.advection_scheme = 'central_4th'

        # Number of variables
        self.nvar = 5 + len(EOS.sc_names)

        if hasattr(cfg,'RANS') and cfg.RANS:
            self.nvar += 7
            self.RANS = True
        else:
            self.RANS = False

        # Targets for absorbing boundary conditions
        self.Q_BC_bot   = torch.empty((self.nvar,decomp.nx_), dtype=self.WP).to(cfg.device)
        self.Q_BC_top   = torch.empty((self.nvar,decomp.nx_), dtype=self.WP).to(cfg.device)
        self.Q_BC_left  = torch.empty((self.nvar,decomp.ny_), dtype=self.WP).to(cfg.device)
        self.Q_BC_right = torch.empty((self.nvar,decomp.ny_), dtype=self.WP).to(cfg.device)
        
        # NN model settings
        if hasattr(cfg,'Use_Model'):
            self.Use_Model = cfg.Use_Model

        else:
            self.Use_Model = False

        # Model training options
        if hasattr(cfg,'Train'):
            self.Train = cfg.Train
            self.Nsteps_Optim = cfg.Nsteps_Optim
            self.model_type = cfg.model_type
        else:
            self.Train = False

        if hasattr(cfg,'debug'):
            self.debug = cfg.debug
            self.plot_dir = cfg.outDir
        else:
            self.debug = False

            
def checkPhysical(EOS, step, Nx, params):

    gamma = params.gamma
    Ma    = EOS.Ma

    rho   = step[0,5:-5,5:-5,:]
    rhoU  = step[1,5:-5,5:-5,:]
    rhoV  = step[2,5:-5,5:-5,:]
    rhoE  = step[3,5:-5,5:-5,:]

    u     = rhoU / rho
    v     = rhoV / rho
    e     = rhoE/rho - 0.5*(u**2 + v**2)
    p     = (gamma-1.0) * rho * e * Ma**2    
    T     = gamma * e * Ma**2

    # Checking if all density values are positve
    rhoisTrue = torch.all(rho > 0)
    pisTrue   = torch.all(T > 0)

    if (rhoisTrue and pisTrue):
        return True
    else:
        return False

#@profile    
def bicgstabHB(jvp, F, maxiters=8000, initGuess=None):
    
    # Algo 7.7 from Yousuf Saad
    
    # Step 1. Initial guess
    if not (initGuess is None):
        x = torch.DoubleTensor(initGuess)
    else:
        x = torch.zeros_like(F)
    r = F - jvp(x)
    rStar = r
    
    # Step 2.
    p = r
    
    # Step 3.
    for i in range(maxiters):
        
        # Step 4.
        Ap    = jvp(p)
        alpha = torch.dot(r,rStar) / torch.dot(Ap, rStar)
        
        # Step 5.
        s     = r - alpha * Ap
        
        # Step 6.
        As    = jvp(s)
        omega = torch.dot(As,s) / torch.dot(As, As)
        
        # Step 7.
        x     = x + alpha * p + omega * s
        
        # Step 8.
        rNew     = s - omega * As
        
        # Step 9.
        beta  =  (torch.dot(rNew, rStar) / torch.dot(r, rStar)) * (alpha / omega)
        
        # Step 10.
        p     = rNew + beta * (p - omega * Ap)
        r     = rNew
        
        res_norm = torch.linalg.norm(r)
        if res_norm <= 1e-10:
            print('Converged')
            break
        #print(res_norm)
        if (i%500==0):
            print(res_norm)
    print('Fin res : {}'.format(torch.norm(jvp(x)-F)))
    return x                                              



def gmresHB(jvp, F, maxiters=1000, initGuess=None, tol=1e-10):
    
    # Initialize the guess
    if initGuess is not None:
        x = torch.DoubleTensor(initGuess)  # Ensure x is a column vector
    else:
        x = torch.zeros_like(F)  # Ensure x is a column vector
    
    r = F - jvp(x).squeeze()  # Assuming jvp returns a column vector that needs to be squeezed
    r_norm = torch.linalg.norm(r)
    Q = r.unsqueeze(1) / r_norm  # Start Q as a two-dimensional tensor with one column
    H = torch.zeros(maxiters+1, maxiters, dtype=torch.double)
    cs = torch.zeros(maxiters, dtype=torch.double)
    sn = torch.zeros(maxiters, dtype=torch.double)
    
    def apply_givens_rotation(s, cs, sn, i):
        for k in range(i):
            temp = cs[k] * s[k] + sn[k] * s[k + 1]
            s[k + 1] = -sn[k] * s[k] + cs[k] * s[k + 1]
            s[k] = temp
        return s

    def givens_rotation(v1, v2):
        t = torch.sqrt(v1**2 + v2**2)
        cs = v1 / t
        sn = v2 / t
        return cs, sn
    
    beta = torch.linalg.norm(r)
    s = torch.zeros(maxiters+1, dtype=torch.double)
    s[0] = beta
    
    for i in range(maxiters):
        # Arnoldi process
        w = jvp(Q[:, i:i+1]).squeeze()  # Use slicing to keep dimensions correct
        for k in range(i + 1):
            H[k, i] = torch.dot(Q[:, k], w)
            w = w - H[k, i] * Q[:, k]
        
        H[i + 1, i] = torch.linalg.norm(w)
        if H[i + 1, i] != 0 and i != maxiters - 1:
            Q = torch.cat([Q, w.unsqueeze(1) / H[i + 1, i]], axis=1)  # Ensure w is a column vector
        
        # Apply Givens rotation
        s = apply_givens_rotation(s, cs, sn, i)
        
        # Compute Givens rotation
        cs[i], sn[i] = givens_rotation(H[i, i], H[i+1, i])
        
        # Eliminate H[i+1, i]
        H[i, i] = cs[i] * H[i, i] + sn[i] * H[i + 1, i]
        H[i + 1, i] = 0.0
        s[i + 1] = -sn[i] * s[i]
        s[i] = cs[i] * s[i]
        
        # Check convergence
        res_norm = abs(s[i + 1])
        if res_norm < tol:
            print('Converged at iteration:', i)
            break

        if (i % 500 == 0):
            print('Residual norm:', res_norm)
    
        # Compute solution
        y = torch.linalg.solve(H[:i+1, :i+1], s[:i+1])
        x += Q[:, :i+1] @ y
    print('Fin res : {}'.format(torch.norm(jvp(x)-F)))
    return x.squeeze()  # Return x as a 1D tensor



# ----------------------------------------------------
# Main PyFlowCL routine
# ----------------------------------------------------
def run(cfg, perturb=None):

    # Set working precision
    WP = torch.float64
    WP_np = np.float64
    if hasattr(cfg, 'WP'):
        WP = cfg.WP
    if hasattr(cfg, 'WP_np'):
        WP_np = cfg.WP_np

    # Enforce grid periodicity -- needed for decomp
    grid = Grid.enforce_periodic(cfg)

    # Initialize parallel environment and domain decomposition
    comms  = Parallel.Comms()
    decomp = Parallel.Decomp(cfg,grid,WP,WP_np)
    
    # Initialize metrics
    #   Used to compute numerical grid transforms, if needed
    metrics = Metrics.central_4th_periodicRectZ(grid, decomp)

    # Initialize curvilinear grid transforms
    Grid.initialize_transforms(cfg, grid, decomp, metrics)

    # Save the grid transforms in the metrics object
    metrics.set_transforms(grid, decomp)

    # Initialize thermochemical equations of state
    if hasattr(cfg, 'EOS_Name'):
        EOS = None
        for name, obj in inspect.getmembers(Thermochemistry, inspect.isclass):
            if name == cfg.EOS_Name:
                EOS = obj(cfg)
        if EOS is None:
            raise Exception('PyFlowCL.py: EOS Name not found in Thermochemistry.py')
    else:
        # Default EOS is dimensionless calorically perfect gas
        EOS = Thermochemistry.Perfect_Gas_Nondim(cfg)
        if (comms.rank==0): print('Defaulting to dimensionless CPG EOS')

    # Extract parameters from the input config
    param = Param(cfg, decomp, EOS, WP, WP_np)

    # Initial CFL/DN estimates
    dx_min = min(torch.amin(grid.Dx), torch.amin(grid.Dy)).cpu().numpy()
    CFL = max(EOS.U0, EOS.base_cs) * param.dt / dx_min
    DN  = EOS.base_mu * param.dt / dx_min**2
    param.CFL = CFL
    if (param.rank==0): print('Initial CFL = {:7.3e}, DN = {:7.3e}'.format(CFL,DN))

    # Initial Mach and Re
    if (EOS.dimensional and param.rank==0):
        Ma0 = EOS.U0 / EOS.base_cs
        Re0 = EOS.U0 * EOS.rho0 * cfg.L0 / EOS.mu
        print('Ma = {:7.3e}, Re = {:7.3e}'.format(Ma0, Re0))
        EOS.Ma = Ma0
        EOS.Re = Re0

    # --------------------------------------------------------------
    # Spectral low-pass filter
    use_filter = False
    if (cfg.Nsteps_filter is not None):
        use_filter = True
        
        # Default filter type is implicit.
        #   Explicit filter has larger dissipation.
        if hasattr(cfg,'explicit_filter'): implicit = not cfg.explicit_filter
        else: implicit = True
        
        lowpass_filter = op.Lowpass_filter_6(grid, decomp, implicit)

        
    # --------------------------------------------------------------
    # Initial condition
    
    # Initialize state data
    if param.RANS:
        aux_names = ['rhok','rhoeps']
    else:
        aux_names = []

    t = 0; Nstart = 0
    if (cfg.dfName_read is not None):
        # Restart file contains primitives
        names = ['rho','U','V','W','e'] + EOS.sc_names_prim + aux_names
        Q = Data.State(names, decomp)

        # Load restart file
        Nstart, t, dt_tmp = Data.read_data(cfg.dfName_read, cfg, decomp, Q)
        t_start = t
        #if (dt_tmp > 1e-16): param.dt = dt_tmp
        if (decomp.rank==0):
            print(' --> Restarting from {} at it={}, t={:9.4e}'.format(cfg.dfName_read,Nstart,t))

        # Convert primitives to conserved
        for name in names:
            if (name=='e'):
                # Convert internal energy to total energy
                Q['rhoE'] = Q.pop('e')
                Q['rhoE'].mul( Q['rho'].interior() )
                Q['rhoE'].add( 0.5*( Q['rhoU'].interior()**2 +
                                     Q['rhoV'].interior()**2 +
                                     Q['rhoW'].interior()**2 ) / Q['rho'].interior() )
            elif (name!='rho'):
                Q['rho'+name] = Q.pop(name)
                Q['rho'+name].mul( Q['rho'].interior() )
                
        names = ['rho','rhoU','rhoV','rhoW','rhoE'] + EOS.sc_names + aux_names
        Q.names = names

    else:
        names = ['rho','rhoU','rhoV','rhoW','rhoE'] + EOS.sc_names + aux_names
        Q = Data.State(names, decomp)
            
        # Get ICs
        IC.get_IC(cfg.IC_opt, grid, Q, param, EOS)
        
        if ( not grid.periodic_eta
             and (param.jproc==0 or param.jproc==param.npy-1)
             and cfg.IC_opt!="channel"
             and cfg.IC_opt!="flat_plate"
             and cfg.IC_opt!="uniform_isothermal_wall"):
            # Enforce Dirichlet no-slip walls
            #   Could use law of the wall...
            Q['rhoE'].sub_( 0.5*( Q['rhoU'].interior()**2 +
                                  Q['rhoV'].interior()**2 +
                                  Q['rhoW'].interior()**2 )/Q['rho'].interior() )
            thk = 0.1
            if (grid.BC_eta_bot=='wall' and param.jproc==0):
                Q['rhoU'].mul_( torch.tanh(grid.Eta[:,:,None]/thk).to(decomp.device) )  # rho*U
                Q['rhoV'].mul_( torch.tanh(grid.Eta[:,:,None]/thk).to(decomp.device) )  # rho*v
                Q['rhoW'].mul_( torch.tanh(grid.Eta[:,:,None]/thk).to(decomp.device) )  # rho*W
            if (grid.BC_eta_top=='wall' and param.jproc==param.npy-1):
                Q['rhoU'].mul_( torch.tanh((cfg.Lx2 - grid.Eta[:,:,None])/thk).to(decomp.device) )
                Q['rhoV'].mul_( torch.tanh((cfg.Lx2 - grid.Eta[:,:,None])/thk).to(decomp.device) )
                Q['rhoW'].mul_( torch.tanh((cfg.Lx2 - grid.Eta[:,:,None])/thk).to(decomp.device) )
            # Adjust rhoE
            Q['rhoE'].add_( 0.5*( Q['rhoU'].interior()**2 +
                                  Q['rhoV'].interior()**2 +
                                  Q['rhoW'].interior()**2 )/Q['rho'].interior() )
        Q['rhoE'].update_border()

    # Option to perturb the IC for adjoint verification
    if (perturb is not None):
        Q['rhoU'].interior()[param.nx//2,param.ny//2,param.nz//2] += perturb

    # Temperature for isothermal walls
    if grid.BC_eta_bot == 'wall' and decomp.jproc == 0:
        if cfg.IC_opt == 'airfoil_dlsgs' and hasattr(cfg, 'Tw'):
            Tw = torch.tensor(cfg.Tw)
            Q['rhoE'].interior()[:, 0, :] = Q['rho'].interior()[:, 0, :] * EOS.get_internal_energy_TY(Tw)
    Q['rhoE'].update_border()

    # --------------------------------------------------------------
    # Adjoint state and target data
    
    if param.Train:
        # Require a model to be used
        param.Use_Model = True
        
        # Initialize adjoint state data
        names_A = ['rho_A','rhoU_A','rhoV_A','rhoW_A','rhoE_A']
        for name in EOS.sc_names:
            names_A.append(name + '_A')
        Q_A = Data.State(names_A, decomp)
            
        # Load target data
        if hasattr(cfg,'load_target_data'):
            Q_T = cfg.load_target_data(EOS, names, decomp, cfg, grid, param, comms)
        else:
            raise Exception('Need to define inputConfig.load_target_data()')

        # Loss function from driver
        try:
            param.loss = cfg.loss
        except:
            raise Exception('Need to define inputConfig.loss()')

    else:
        Q_A = None
        Q_T = None

        
    # --------------------------------------------------------------
    # Initialize the neural network model

    if (param.Use_Model):
        # Function to apply model (called from RHS)
        try:
            param.apply_model = cfg.apply_model
            param.apply_model_wall = cfg.apply_model_wall
        except:
            raise Exception('Need to define inputConfig.apply_model()')

        # Model definition from the driver script
        try:
            param.model = cfg.define_model()
            
            wallModel = False
            if wallModel:
                #modelWall = Model.MLP(2, [32, 32, 32], 6)
                modelWall = Model.NeuralNetworkModel_ELU_dist_wall(500, 2, 5, 1.0)
                # Loading wall model
                #wallModelPath = '/data/SPARTA_rebuild/sparta/examples/flat_plate/aPriori_pyflow_targ_u_T_new_2.pt'
                wallModelPath = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_dist_wall_scalar_trans_model_aPost/saveMod_wallModel_7'#'/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_withWall_restTrained/saveMod_wallModel_34'
                modelWall.load_state_dict(torch.load(wallModelPath))
                modelWall.to(cfg.device)
                
            
        except:
            raise Exception('Need to define inputConfig.define_model()')

        # # Synchronize model across all processes
        # for param_i in param.model.parameters():
        #     tensor_i = param_i.data.cpu().numpy()
        #     tensor_i = comms.comm.allreduce(tensor_i, op = MPI.SUM) #Does not work for GPU??
        #     param_i.data = torch.DoubleTensor( tensor_i/np.sqrt(np.float(comms.size)) ) 

        # Load existing model?
        if (cfg.Load_Model):
            print('Model Loaded')
            param.model.load_state_dict(torch.load(cfg.modelName_read))

        # Move model to GPU
        param.model.to(cfg.device)

        if (param.Train):
            # Initialize optimizer 
            param.optimizer = optim.RMSprop(param.model.parameters(), lr=cfg.LR)
            
            if wallModel:
                optimizer_wall = optim.RMSprop(modelWall.parameters(), lr=cfg.LR)
                #wallOptimPath = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_dist_wall_aPost/saveOpt_wallModel_0'
                #optimizer_wall.load_state_dict(torch.load(wallOptimPath))
            
            # Load existing optimizer?
            if (cfg.Load_Model):
                param.optimizer.load_state_dict(torch.load(cfg.optimizerName_read))
                print('Optimizer Loaded')
                for param_group in param.optimizer.param_groups:
                    param_group['lr'] = cfg.LR
            
        else:
            param.optimizer = None
            param.Nsteps_Optim = cfg.Nsteps

    else:
        param.model     = None
        param.optimizer = None
        param.Nsteps_Optim = cfg.Nsteps

        
    # --------------------------------------------------------------
    # Boundary Conditions
        
    # Set up absorbing layer activation functions
    if hasattr(grid,'get_absorbing_layers'):
        # The initialized grid class has a dedicated farfield function
        grid.get_absorbing_layers(cfg,decomp)
        
    else:
        # Use the generic routine in the Grid module
        #   NOTE: assumes rectilinear XY boundaries
        Grid.get_absorbing_layers(grid,cfg,decomp)

    # Set initial target solutions for absorbing layers
    Boundary.update_farfield_targets(Q, param, grid, cfg)
        
    def enforce_inlet(q,param):  # JFM MOVE TO DRIVER
        # Resets inlet profiles
        if (decomp.iproc > 0): return

        if cfg.IC_opt=="planar_jet_spatial_RANS" or cfg.IC_opt=="planar_jet_spatial_RANS_coflow" :
            q['rhoU'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[1,:]
            q['rhoV'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[2,:]
            q['rhok'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[5,:] 
            q['rhoeps'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[6,:] 

        if cfg.IC_opt=="planar_jet_spatial_lam" :
            q['rhoU'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[1,:]
            q['rhoV'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[2,:]
            q['rhoW'].interior()[0,:,0] = q['rho'].interior()[0,:,0]*param.Q_BC_left[3,:]

        if  cfg.IC_opt=="planar_jet_spatial_turb":
            q['rhoU'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * (( (param.Q_BC_left[1,:,None]) * (1 + 0.1 * (torch.rand(grid.X[0,:,:].shape,device=param.device) - 0.5) )))
            q['rhoV'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * param.Q_BC_left[2,:,None]
            q['rhoW'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * (((param.Q_BC_left[2,:,None]) + (( (param.Q_BC_left[1,:,None]) *
                                                                                                          (0 + 0.05 * (torch.rand(grid.X[0,:,:].shape, device=param.device) - 0.5) )))))

        if  cfg.IC_opt=="planar_jet_spatial_turb_coflow":
            shear_layer_thickness = 0.03
            u_y =  0.5*(torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 - 1.0)/shear_layer_thickness))     
            q['rhoU'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * (( (param.Q_BC_left[1,:,None]) * (1 + u_y * 0.1 * (torch.rand(grid.X[0,:,:].shape,device=param.device) - 0.5) )))
            q['rhoV'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * param.Q_BC_left[2,:,None]
            q['rhoW'].interior()[0,:,:] = q['rho'].interior()[0,:,:] * (((param.Q_BC_left[2,:,None]) + (( u_y * (0 + 0.05 * (torch.rand(grid.X[0,:,:].shape,device=param.device) - 0.5) )))))
               
            
    # Option to add random noise
    if (hasattr(cfg,'add_noise')):
        if (cfg.add_noise):
            amp  = 0.25
            rand = torch.from_numpy(np.random.rand(decomp.nx_,decomp.ny_,decomp.nz_)).to(param.device) - 0.5
            Q['rhoU'].add( amp*rand )
            rand = torch.from_numpy(np.random.rand(decomp.nx_,decomp.ny_,decomp.nz_)).to(param.device) - 0.5
            Q['rhoV'].add( amp*rand )
            if (grid.ndim==3):
                rand = torch.from_numpy(np.random.rand(decomp.nx_,decomp.ny_,decomp.nz_)).to(param.device) - 0.5
                Q['rhoW'].add( amp*rand )
            del rand

            # Enforce walls
            for var in ('rhoU','rhoV','rhoW'):
                if (grid.BC_eta_bot=='wall' and param.jproc==0):
                    Q[var].interior()[:,0,:] = 0.0
                if (grid.BC_eta_top=='wall' and param.jproc==param.npy-1):
                    Q[var].interior()[:,-1,:] = 0.0
                Q[var].update_border()

            # Need to reset total energy to avoid T fluctuations (unphysical)
    
    # --------------------------------------------------------------
    # Body forces
    param.bodyforce = Bodyforce.Bodyforce( cfg, comms, decomp, grid, Q )
    
    # --------------------------------------------------------------
    # Main loop

    # solver_mode options:
    #   unsteady_RK4 - all forward simulations (unsteady & steady pseudo-time); unsteady adjoint simulations
    #   steady_adjoint_RK4 - steady pseudo-time adjoint simulations
    #   steady_Newton - steady Newton forward & adjoint simulations
    if hasattr(cfg,'solver_mode'):
        solver_mode = cfg.solver_mode
    else:
        solver_mode = 'unsteady_RK4'

    # Some checks
    if (solver_mode=='steady_adjoint_RK4' and (not param.Train)):
        solver_mode = 'unsteady_RK4'
        if (param.rank==0): print(' --> steady_adjoint_RK4 mode only for training; defaulting to unsteady_RK4')

    # Allocate temporary array for gradients in shock-capturing scheme
    if param.artDiss:
        tmp_grad = Data.PCL_Var(decomp,'tmp_grad')
        if EOS.dimensional:
            raise Exception('PyFlowCL.py: Artificial dissipation requires the dimensionless CPG EOS')
    else:
        tmp_grad = None
        
    # Create a pointer to the RHS function
    try:
        if (grid.ndim==1):
            rhs = RHS(grid, metrics, param, EOS, tmp_grad).NS_1D
        elif (grid.ndim==2):
            if param.RANS:
                rhs = RHS(grid, metrics, param, EOS, tmp_grad).NS_2D_RANS 
            
            elif solver_mode == 'steady_Newton':
                #rhs_jvp = RHS(grid, metrics, param, EOS,  lowpass_filter, tmp_grad ).NS_2D_jvp
                RHS_obj = RHS(grid, metrics, param, EOS, tmp_grad, jvp=True )
                rhs_jvp = RHS_obj.NS_2D
                
                wm_index_1 = 2
                wm_index_2 = 3
                wm_index_3 = 4
                wm_index_4 = 5

                PhysParams = param
            else:
                rhs = RHS(grid, metrics, param, EOS, tmp_grad).NS_2D
                
                #rhs = RHS(grid, metrics, param, EOS, tmp_grad).NS_2D_Euler  ### JFM TESTING
                # if decomp.rank==0:
                #     print('--> PyFlowCL.py: WARNING: Using 2D Euler Fluxes')
                
        elif (grid.ndim==3):
            rhs = RHS(grid, metrics, param, EOS, tmp_grad).NS_3D
    except:
        raise Exception('PyFlowCL.py: Could not configure RHS')

    # Allocate RK4 state memory
    Q_tmp = Data.State(names, decomp)
        
    if param.Train:
        # Create a pointer to the adjoint RHS
        Loss_der = Adjoint_RHS(comms, grid, metrics, param, rhs_jvp, EOS).Loss_der
        #rhs_A = Adjoint_RHS(comms, grid, metrics, param, rhs, EOS).RHS
        
        # Allocate RK4 state memory for adjoint equations
        Q_A_tmp = Data.State(names_A, decomp)
    else:
        rhs_A   = None
        Q_A_tmp = None

    # Initialize simulation monitor
    monitor = Monitor.PCL_Monitor(cfg,grid,param,comms,decomp)
    
    # Timing - move to Monitor.py
    time1 = time.time()
    time0 = time1

    # Assuming param.Nsteps_Optim = cfg.Nsteps for Train = False

    F_res = None
    
    # Switch for different values of solver_mode
    if (solver_mode == 'unsteady_RK4'):   # --------------------------------------------------------------------
        # All forward simulations (unsteady & steady pseudo-time); unsteady adjoint simulations
        if param.Train:
            J_start = param.loss(comms, grid, param, metrics, Q, Q_T, None)
            
        # 1. Outer loop
        for m in range(cfg.Nsteps//param.Nsteps_Optim):

            # Checkpoint lists for Q and dt
            if param.Train:
                Q_ls  = []
                dt_ls = []
                param.optimizer.zero_grad()

            # 2. Forward inner loop
            for n in range(Nstart+m*param.Nsteps_Optim, Nstart+(m+1)*param.Nsteps_Optim):
                
                # Update time step size
                predict_dt(grid, param, EOS, comms, Q)

                # Change Mach number if specified by user
                if hasattr(cfg, 'change_Mach') and cfg.change_Mach:
                    # wt = 0 at start, = 1 at finish
                    wt = (t - t_start)/cfg.change_Mach_time_window
                    if wt >= 1.0:
                        cfg.change_Mach = False
                    if n == Nstart:
                        Ma_start = EOS.Ma
                        
                    Ma_out = wt*cfg.change_Mach_target + (1.0 - wt)*Ma_start
                    Conversion.change_Mach(Q, EOS.Ma, Ma_out, EOS)
                    EOS.Ma = Ma_out

                    # Adjust rho*E farfield targets
                    Boundary.update_farfield_targets(Q, param, grid, cfg, names=['rhoE',])

                # Monitoring
                if ((m*param.Nsteps_Optim+n) % cfg.N_monitor == 0):
                    time1 = monitor.step(cfg,grid,param,EOS,comms,decomp,n,t,param.dt,Q,Q_A,Q_T,time1,F_res)

                filter_delq = False
                   
                with torch.inference_mode():
                    if False:
                        k1, _ = rhs( Q ); Q_tmp.copy_sum( Q, param.dt * 0.5 * k1 )
                        k2, _ = rhs( Q_tmp ); Q_tmp.copy_sum( Q, param.dt * 0.5 * k2 )
                        k3, _ = rhs( Q_tmp ); Q_tmp.copy_sum( Q, param.dt * k3 )
                        k4, _ = rhs( Q_tmp )

                        # RK4 update
                        k2 *= 2.0
                        k3 *= 2.0
                        k1 += k2
                        k1 += k3
                        k1 += k4
                        k1 /= 6.0
                        Q.add( param.dt * k1 )
                    else:
                        k1, _ = rhs( Q )
                        
                        R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5,0,0),"constant",0)  ## FIX! HARDCODED for 2D!

                        if filter_delq:
                            if (use_filter and (n%cfg.Nsteps_filter==0)):
                                k1 = R(k1)
                                lowpass_filter.apply(k1, names=('rho','rhoU','rhoV','rhoW','rhoE'))
                                k1 = k1[:,5:-5,5:-5,:]
                                
                        Q.add( param.dt * k1 )

                        F_res = R(k1)

                # Save checkpointed values
                if param.Train:
                    Q_checkpoint = Data.State(names, decomp)
                    Q_checkpoint.deepcopy(Q)
                    Q_ls.append(Q_checkpoint)
                    dt_ls.append(param.dt)
                    
                # Apply low-pass filtering
                if not filter_delq:
                    if (use_filter and (n%cfg.Nsteps_filter==0)):
                        lowpass_filter.apply(Q)
                    
                if (cfg.IC_opt=="planar_jet_spatial_lam" or
                        cfg.IC_opt=="planar_jet_spatial_turb" or
                        cfg.IC_opt=="planar_jet_spatial_RANS" or
                        cfg.IC_opt=="planar_jet_spatial_turb_coflow" or
                        cfg.IC_opt=="planar_jet_spatial_RANS_coflow"):
                    enforce_inlet( Q , param)  # Doing this because filter changes the inflow BC

                # JFM: Need to implement 2D stats from 3D data
                #if n >= cfg.turb_stat_start:
                #    Stats.save(n, Q, cfg, grid, decomp ,param, t)

                # Advance time
                t += param.dt
                
            # DONE 2. FORWARD INNER LOOP

            # Adjoint verification
            #if (abs(perturb) > 0.0):
            if (perturb is not None):
                J = param.loss(comms, grid, param, metrics, Q, Q_T, None)

            # 3. Optimizer step
            elif param.Train:
                # Load target data at this time level
                Q_T = cfg.load_target_data(decomp, Q_T, t)
                
                # Evaluate the objective function
                J = param.loss(comms, grid, param, metrics, Q, Q_T, None)
                
                # Get adjoint ICs
                IC.get_IC_Adjoint(grid, Q, Q_A, param)  # NEED TO INCLUDE TARGET DATA
                
                if (param.rank==0): print('Starting adjoint iteration, loss = {:8.3e}'.format(J))
                t_ = t

                # 4. Adjoint inner loop
                for n,(Q_,dt_) in enumerate(zip(reversed(Q_ls), reversed(dt_ls))):
                    
                    _, k1_A = rhs_A(  Q_, Q_A, Q_T)
                    Q_A_tmp.copy_sum( Q_A, dt_*0.5*k1_A )
                    _, k2_A = rhs_A(  Q_, Q_A_tmp, Q_T)
                    Q_A_tmp.copy_sum( Q_A, dt_*0.5*k2_A )
                    _, k3_A = rhs_A(  Q_, Q_A_tmp, Q_T)
                    Q_A_tmp.copy_sum( Q_A, dt_*k3_A );
                    _, k4_A = rhs_A(  Q_, Q_A_tmp, Q_T)
                    
                    # RK4 update
                    k2_A *= 2.0
                    k3_A *= 2.0
                    k1_A += k2_A
                    k1_A += k3_A
                    k1_A += k4_A
                    k1_A /= 6.0
                    Q_A.add( dt_*k1_A )

                    # Backward time
                    t_ -= dt_

                    #if (param.rank==0): print('Adjoint step ',n, t_)
                    monitor.unsteady_adjoint_step(grid, comms, n, t_, k1_A)

                # DONE 4. ADJOINT INNER LOOP
                
                # Evaluate the objective function
                #J = param.loss(comms, grid, param, metrics, Q, Q_T, None)
                
                # Update NN parameters
                optimizer_step(comms, cfg, param)
                
                if (param.rank==0): print('Finished adjoint iteration')
                    
                # Cleanup
                for Q_ in Q_ls: del Q_
                del Q_ls, dt_ls

            # DONE 3. OPTIMIZER STEP

        # DONE 1. OUTER LOOP
        


    elif (solver_mode == 'steady_adjoint_RK4'):   # ------------------------------------------------------------
        # Steady pseudo-time adjoint simulations
        J_start,_,_,_,_ = param.loss(comms, grid, param, metrics,EOS, Q, Q_T, None)
        
        for n in range(Nstart, Nstart+cfg.Nsteps):
            param.optimizer.zero_grad()

            # Monitoring
            if (n % cfg.N_monitor == 0):
                time1 = monitor.step(cfg,grid,param,EOS,comms,decomp,n,t,param.dt,Q,Q_A,Q_T,time1)
            
            k1, k1_A = rhs_A(  Q, Q_A, Q_T)
            # Q_tmp.copy_sum(   Q,   param.dt*0.5*k1 )
            # Q_A_tmp.copy_sum( Q_A, param.dt*0.5*k1_A )
            
            # k2, k2_A = rhs_A(  Q_tmp, Q_A_tmp, Q_T)
            # Q_tmp.copy_sum(   Q,   param.dt*0.5*k2 )
            # Q_A_tmp.copy_sum( Q_A, param.dt*0.5*k2_A )
            
            # k3, k3_A = rhs_A(  Q_tmp, Q_A_tmp, Q_T)
            # Q_tmp.copy_sum(   Q,   param.dt*k3 )
            # Q_A_tmp.copy_sum( Q_A, param.dt*k3_A )
            
            # k4, k4_A = rhs_A(  Q_tmp, Q_A_tmp, Q_T)
            
            # RK4 update
            # k2 *= 2.0
            # k3 *= 2.0
            # k1 += k2
            # k1 += k3
            # k1 += k4
            # k1 /= 6.0
            Q.add( param.dt*k1 )
            
            # k2_A *= 2.0
            # k3_A *= 2.0
            # k1_A += k2_A
            # k1_A += k3_A
            # k1_A += k4_A
            # k1_A /= 6.0
            Q_A.add( param.dt*k1_A )
            
            if True:#( n % cfg.Nsteps_Optim == 0 ):
                # Evaluate the objective function
                J,_,_,_,_ = param.loss(comms, grid, param, metrics, EOS, Q, Q_T, None)
                
                # Evaluate RHS residual -- residual should converge to
                # zero as the solution converges to a steady state.
                resid_F = torch.mean(torch.abs( k1  [1,:,:,:] ))
                resid_A = torch.mean(torch.abs( k1_A[1,:,:,:] ))

                # Update the optimizer state
                optimizer_step(comms, cfg, param)

            if (n % cfg.N_monitor == 0):
                # Write training progress to terminal
                if (param.rank==0):
                    print('Forward Resid: {:8.4e}\tAdj Resid: {:8.4e}\t     Loss: {:8.4e}\t{:8.4e}'
                          .format(resid_F, resid_A, J/J_start, J))

            # Apply low-pass filtering
            # if (use_filter) and (n%cfg.Nsteps_filter==0):
            #     lowpass_filter.apply(Q)
            
            # Advance time
            t += param.dt
    
            
    # elif (solver_mode == 'psuedo_steady'):
        
        
    #     with torch.inference_mode():
    #         k1, _ = rhs(Q)
    #         Q.add( param.dt * k1 )
            
    #     if param.train:
    #         _, k1_A = rhs_A(  Q_, Q_A, Q_T)
    #         Q_A.add( dt_*k1_A )
            
    #         if 
    #         monitor.unsteady_adjoint_step(grid, comms, n, t_, k1_A)
            
            

    elif (solver_mode == 'steady_Newton'):    # ----------------------------------------------------------------
        # Steady Newton forward & adjoint simulations
        # raise Exception('PyFlowCL.py: steady Newton solver not yet implemented')

        if param.Train:
            LossList = []
            LossList1 = []
            LossList2 = []
            LossList3 = []
            LossList4 = []
            AdjResList = []
            LRList     = []
            Q_T = cfg.load_target_data(EOS, names, decomp, cfg, grid, param, comms)
            J_start,_,_,_,_ = param.loss(comms,grid,param,metrics,EOS,Q,Q_T,None)
            print('Starting Loss : {}'.format(J_start.detach().cpu().numpy()))
            LossList.append(J_start.detach().cpu().numpy())
            
        # Creating tensor copies of PCL Var object
        # Need to henralize for 1D/3D
        F  = torch.zeros([4]+list(Q["rhoU"].var.shape), dtype=param.WP).to(param.device)
            
            
        # calculating indices for optimization domain
        xLeftCut  =  +6#grid.xIndOptLeft  + 5 
        xRightCut =  -6#grid.xIndOptRight - 5 
        yTopCut   =  -6#grid.yIndOptTop   - 5 
        yBotCut   =  +6#grid.yIndOptBot   + 5 
        
        T_prof   = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/TWallProf_M7_lowDens_256.npy')).to(param.device)
        U_prof   = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/uWallProf_M7_lowDens_256.npy')).to(param.device)


        # 1. Forward Outer loop
        for m in range(cfg.Nsteps//param.Nsteps_Optim):

            
            # padding and reshaping function 
            def padReshp(xVec, isTensor=False, pad=True):
                
                # Reshape
                xVec = xVec.reshape(F.shape[0],(F.shape[1]-(xLeftCut-xRightCut)),(F.shape[2]-(yBotCut-yTopCut)),F.shape[3])

                # xVec = xVec.reshape(F.shape[0],(F.shape[1]-12),(F.shape[2]-12),F.shape[3])
                
                # Converting to tensor
                #xVec = torch.DoubleTensor(xVec).to(param.device)
                
                # Padding
                if pad:
                    R = lambda L: torch.nn.functional.pad(L,(0,0,-yTopCut,yBotCut,-xRightCut,xLeftCut),"constant",0)
                    xVec = torch.stack((R(xVec[0,:,:,:]),
                                        R(xVec[1,:,:,:]),
                                        R(xVec[2,:,:,:]),
                                        R(xVec[3,:,:,:])),dim=0)
                
                return xVec


            # Checkpoint lists for Q and dt
            if param.Train:
                Q_ls  = []
                dt_ls = []
                param.optimizer.zero_grad()

            
            # Normalization (free-stream)
            F_normConst = torch.ones_like(F)
            if EOS.dimensional:
                F_normConst[:,:,0] = EOS.rho0
                F_normConst[:,:,1] = EOS.rho0 * EOS.U0
                
                e0 = EOS.get_internal_energy_TY(EOS.T0)
                rhoE0 =  (EOS.rho0 * e0 + 0.5*(EOS.rho0 * EOS.U0**2)) 
                F_normConst[:,:,1] = rhoE0     
                
            
            #F_tmp  = torch.zeros([4]+list(Q["rhoU"].var.shape)).to(param.device)
            F[0] = Q["rho"].var.to(decomp.device)
            F[1] = Q["rhoU"].var.to(decomp.device)
            F[2] = Q["rhoV"].var.to(decomp.device)
            F[3] = Q["rhoE"].var.to(decomp.device)

            # Threshold for bracketing 
            threshFac = 5e-1#1e-4##
            delqThresh = threshFac * torch.sqrt(F.shape[1]*F.shape[2]*(torch.amax(F[0][5:-5,5:-5,0])**2 +
                                                                       torch.amax(F[1][5:-5,5:-5,0])**2 +
                                                                       torch.amax(F[2][5:-5,5:-5,0])**2 +
                                                                       torch.amax(F[3][5:-5,5:-5,0])**2))

            # 2. Forward Inner Loop 
            if (m==0 and param.Train) :
                nStart = Nstart
                nEnd   = Nstart + int(1.0e4)
            
            else:
                nStart = Nstart+m*param.Nsteps_Optim
                nEnd   = Nstart+(m+1)*param.Nsteps_Optim

            # Save Model
            if (param.Train and cfg.Save_Model and param.rank==0):
                param.model.cpu()
                torch.save( param.model.state_dict(),     cfg.modelName_save)    
                torch.save( param.optimizer.state_dict(), cfg.optimizerName_save)
                param.model.to(param.device)
                
                if wallModel:
                    modelNameFin = cfg.modelName_save + '_wallModel_' + 'Init'
                    optimizerNameFin = cfg.optimizerName_save + '_wallModel_' + 'Init'
                    modelWall.cpu()
                    torch.save( modelWall.state_dict(),     modelNameFin)    
                    torch.save( optimizer_wall.state_dict(), optimizerNameFin)
                    modelWall.to(param.device)
                
            
            for n in range(nStart, nEnd):

                with torch.inference_mode():
                
                    ###########################################
                    ###### RK4 Check (To check if RHS_jvp is fine)##########################
                    k1 = rhs_jvp( F ); #F_tmp = F + param.dt * 0.5 * k1#Q_tmp.copy_sum( Q, param.dt * 0.5 * k1 )
                    # k2 = rhs_jvp( F_tmp ); F_tmp = F + param.dt * 0.5 * k2#Q_tmp.copy_sum( Q, param.dt * 0.5 * k2 )
                    # k3 = rhs_jvp( F_tmp ); F_tmp = F + param.dt * k3 #Q_tmp.copy_sum( Q, param.dt * k3 )
                    # k4 = rhs_jvp( F_tmp )
    
                    # # RK4 update
                    # k2 *= 2.0
                    # k3 *= 2.0
                    # k1 += k2
                    # k1 += k3
                    # k1 += k4
                    # k1 /= 6.0
                    F += param.dt * k1#Q.add( param.dt * k1 )
                    ##############################################
    
    
                    # Residual Check 
                    F_res = rhs_jvp(F)   ## Can replace this with Flux and avoid another RHS eval
                    #print(F_res.shape)
                    F_res_norm = torch.norm(F_res[:,30:-30,30:-30,:]/F_normConst[:,30:-30,30:-30,:])
                    # stepNorm = torch.norm(delq)
                    
                    monFlag = False
    
                    if ((m*param.Nsteps_Optim+n) % cfg.N_monitor == 0):
    
                        monFlag = True
                        # Old Residuals
                        print('F res : {}'.format(F_res_norm))
                        # print('Step norm : {}'.format(stepNorm))
                        print('dt : {}'.format(param.dt))
                        
                        # Residuals as computed from normalized RHS
                        q0_res_old = (abs(torch.mean(abs(F_res[0][30:-30,30:-30,0]/F_normConst[0][30:-30,30:-30,0]))))  # denomintors are Ref values
                        q1_res_old = (abs(torch.mean(abs(F_res[1][30:-30,30:-30,0]/F_normConst[1][30:-30,30:-30,0]))))
                        q2_res_old = (abs(torch.mean(abs(F_res[2][30:-30,30:-30,0]/F_normConst[2][30:-30,30:-30,0]))))
                        q3_res_old = (abs(torch.mean(abs(F_res[3][30:-30,30:-30,0]/F_normConst[3][30:-30,30:-30,0]))))
                       
                        # print("residuals are ", q0_res.numpy() ,q1_res.numpy() ,q2_res.numpy() ,q3_res.numpy())
                        print("old residuals are ",
                              q0_res_old.cpu().numpy(),
                              q1_res_old.cpu().numpy(),
                              q2_res_old.cpu().numpy(),
                              q3_res_old.cpu().numpy())
                        time1 = monitor.step(cfg,grid,param,EOS,comms,decomp,n,t,param.dt,Q,Q_A,Q_T,time1,F_res)
    
                    
                    # One extra time-step for grad
                    # k1 = rhs_jvp( F );
                    # F += param.dt * k1
                    
                               
                # Slip-wall 
                SC = []
    
                rho_upd = F[0][5:-5,5:-5,:].detach()
                rhoU_upd = F[1][5:-5,5:-5,:].detach()
                rhoV_upd = F[2][5:-5,5:-5,:].detach()
                rhoE_upd = F[3][5:-5,5:-5,:].detach()

                # Recomputing primitives
                u_upd = (F[1][5:-5,5:-5,:]/F[0][5:-5,5:-5,:]).detach()
                v_upd = (F[2][5:-5,5:-5,:]/F[0][5:-5,5:-5,:]).detach()
                E_upd = (F[3][5:-5,5:-5,:]/F[0][5:-5,5:-5,:]).detach()
                
                T_upd, p_upd, e_upd = EOS.get_TPE_tensor(F, SC)
                T_upd = T_upd[5:-5,5:-5,:].detach().clone()
                    
                # model inference 
                wallModel = False
                
                # Importing wall model for inference 

                if wallModel:
                    
                     #model_inputs = torch.stack(((T_upd[grid.wallPoint:, 0, :].detach()/EOS.T0).squeeze(), (u_upd[grid.wallPoint:, 0, :].detach()/EOS.U0).squeeze()),dim=1)
                     #model_outputs = modelWall.forward(model_inputs)
                    Vs, T_jump = param.apply_model_wall(modelWall, F, grid,
                                                  metrics, param, EOS)
                     
                    lamb = 0.0
                    Vs = (1-lamb)*U_prof + lamb*Vs
                    T_jump = (1-lamb)*T_prof + lamb*T_prof 
                    
                with torch.inference_mode():

                    # Creating copy to pass through filters
                    Q["rho"].copy(F[0][5:-5,5:-5,:])  
                    Q["rhoU"].copy(F[1][5:-5,5:-5,:])
                    Q["rhoV"].copy(F[2][5:-5,5:-5,:])
                    Q["rhoE"].copy(F[3][5:-5,5:-5,:])

            
            # 3. Adjoint Solve 
            if param.Train:

                #torch.autograd.set_detect_anomaly(True)
                # Resetting the time-step
                # param.dt = dt_init * 1e-3
                dt_imp   = param.dt * 5.0e-1
                

                LossCurr, LossCurr1, LossCurr2, LossCurr3, LossCurr4 = param.loss(comms,grid,param,metrics,EOS,Q,Q_T,None)
                LossCurr = torch.sum((Vs[:-56] - Q_T['U'].interior()[grid.wallPoint:-56, 0, 0])**2 * (1/EOS.U0**2)) + torch.sum((T_jump[:-56] - Q_T['T'].interior()[grid.wallPoint:-56, 0, 0])**2 * (1/EOS.T0**2))
                
                print('{}.  Loss_wall : {}'.format(m, LossCurr.detach().cpu().numpy()))

                # Computing derivative of the Loss function
                dJdU = Loss_der(Q,Q_T)[:,(xLeftCut-5):(xRightCut+5),(yBotCut-5):(yTopCut+5),:]#[:,grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop,:]##[:,1:-1,1:-1,:]#
                # print(dJdU.shape)

                # Temp variable for dJdU
                Flux_A  = -dJdU.squeeze().flatten()

                # Setting up jvp function
                # Direct solve 
                
                M_x = lambda xVec: (((torch.autograd.functional.jvp(rhs_jvp, F, padReshp(xVec))[1])))[:,xLeftCut:xRightCut,yBotCut:yTopCut,:].flatten()

                # Implicit solve 
                # M_x_imp = lambda xVec: (((torch.autograd.functional.jvp(rhs_jvp, F.detach(), padReshp(xVec))[1])+ (1/dt_imp)*padReshp(xVec))[:,5:-5,5:-5,:]).flatten()

                # Solving adjoint equation
                (uHat) = bicgstabHB(M_x, Flux_A.detach(), 8000)
                #(uHat) = gmresHB(M_x, Flux_A.detach(), 1000)

                uHat = padReshp(uHat)

                # Clearing gradients
                param.optimizer.zero_grad()

                # Calling RHS to create graph 
                Flux = rhs_jvp(F)

                # Inner product (uHat^T * f)
                L = 1 * torch.inner(uHat.flatten(), Flux.flatten())

                # Backward pass
                L.backward()
                # LossCurr.backward()

                # Updating parameters
                #LRList.append(param.optimizer.param_groups[0]['lr'])
                optimizer_step(comms, cfg, param, m)
                
                
                # Updating wall model parameters
                if False:
                    optimizer_wall.zero_grad()
                    LossCurr.backward()
                    optimizer_step_wm(comms, cfg, param, m, modelWall, optimizer_wall)
                    
                

                # Plotting loss
                plt.semilogy(LossList, label='Full')
                #plt.semilogy(LossList1, label = '$ \\rho $')
                #plt.semilogy(LossList2, label = '$\\rho U$')
                #plt.semilogy(LossList3, label = '$\\rho V$')
                #plt.semilogy(LossList4, label = '$\\rho E$')
                plt.xlabel('# Optimization iterations')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(cfg.outDir + '/LossConvergence.pdf')
                plt.close()
                
                # Plotting adjoint residual 
                # plt.semilogy(AdjResList)
                # plt.xlabel('# Optimization iterations')
                # plt.ylabel('Adjoint residual')
                # plt.savefig(cfg.outDir + 'AdjConvergence.pdf')
                # plt.close()

                # Plotting LR Evolution 
                plt.semilogy(LRList)
                plt.xlabel('# Optimization iterations')
                plt.ylabel('Learning Rate')
                plt.savefig(cfg.outDir + '/LREvolve.pdf')
                plt.close()

                # # Plotting parameter evolution
                # plt.plot(RHS_obj.constList)
                # plt.xlabel('# Optimization iterations')
                # plt.ylabel('$\Theta$')
                # plt.savefig(cfg.outDir + 'paramEvol.png')
                # plt.close()
                
                # # Plotting loss vs theta
                # plt.semilogy(RHS_obj.constList, LossList)
                # plt.ylabel('Loss')
                # plt.xlabel('$\Theta$')
                # plt.savefig(cfg.outDir + 'LossVsTheta.png')
                # plt.close()

            

            

    elif (solver_mode == 'DPLR'):    # ----------------------------------------------------------------
        # Data-parallel line relaxation (Wright, Candler, & Bose AIAA J 1998)

        # Save the RHS Jacobian when computing metrics
        metrics.save_jac = True

        # All forward simulations (unsteady & steady pseudo-time); unsteady adjoint simulations
        if param.Train:
            J_start = param.loss(comms, grid, param, metrics, Q, Q_T, None)
            
        # 1. Outer loop
        for m in range(cfg.Nsteps//param.Nsteps_Optim):

            # Checkpoint lists for Q and dt
            if param.Train:
                Q_ls  = []
                dt_ls = []
                param.optimizer.zero_grad()

            # 2. Forward inner loop
            for n in range(Nstart+m*param.Nsteps_Optim, Nstart+(m+1)*param.Nsteps_Optim):
                
                # Update time step size
                predict_dt(grid, param, EOS, comms, Q)

                # Monitoring
                if ((m*param.Nsteps_Optim+n) % cfg.N_monitor == 0):
                    time1 = monitor.step(cfg,grid,param,EOS,comms,decomp,n,t,param.dt,
                                         Q,Q_A,Q_T,time1,F_res)

                # Save checkpointed values
                if param.Train:
                    Q_checkpoint = Data.State(names, decomp)
                    Q_checkpoint.deepcopy(Q)
                    Q_ls.append(Q_checkpoint)
                    dt_ls.append(param.dt)

                # Advance the state
                with torch.inference_mode():

                    ## WARNING! NOT SET UP FOR PERIODIC BOUNDARIES
                    
                    # ----------------------------------------------------------------
                    # Explicit Euler RHS
                    k1, _ = rhs( Q )
                    k1 *= param.dt

                    # ----------------------------------------------------------------
                    # LHS sub-matrices
                    A_hat, B_hat, C_hat, D_hat, E_hat = metrics.get_dplr_matrices(param.dt)
                    
                    # LHS matrix - y-contiguous
                    # Diagonal blocks - A
                    n1,n2,nx,ny,nz = A_hat.shape
                    Nx = nx*ny*nz
                    N1 = Nx*n1
                    N2 = Nx*n2
                    # Swap ny and nz
                    A_hat = torch.swapaxes(A_hat, 4, 3) # (n1,n2,nx,nz,ny)
                    A_hat = A_hat.reshape(n1,n2,Nx)
                    # Move Nx to leading dimension
                    A_hat = torch.permute(A_hat, (2,0,1)).cpu().numpy() # (Nx,n1,n2)
                    # Indices
                    indptr = np.arange(Nx+1)
                    indices = indptr[:-1]
                    # BSR
                    A_BSR = sparse.bsr_matrix((A_hat, indices, indptr), shape=(N1,N2))

                    # Superdiagonal blocks - B
                    # Swap ny and nz
                    B_hat = torch.swapaxes(B_hat, 4, 3) # (n1,n2,nx,nz,ny)
                    B_hat = B_hat.reshape(n1,n2,Nx)
                    # Move Nx to leading dimension
                    B_hat = torch.permute(B_hat, (2,0,1)).cpu().numpy()
                    # Indices
                    indptr = np.arange(Nx)
                    indptr = np.concatenate([indptr, indptr[-1:]])
                    indices = np.arange(Nx) + 1
                    # BSR
                    B_BSR = sparse.bsr_matrix((B_hat, indices, indptr), shape=(N1,N2))

                    # Subdiagonal blocks - C
                    # Swap ny and nz
                    C_hat = torch.swapaxes(C_hat, 4, 3) # (n1,n2,nx,nz,ny)
                    C_hat = C_hat.reshape(n1,n2,Nx)
                    # Move Nx to leading dimension
                    C_hat = torch.permute(C_hat, (2,0,1)).cpu().numpy()
                    # Indices
                    indptr = np.concatenate([[0,], np.arange(Nx)])
                    indices = np.arange(Nx)
                    # BSR
                    C_BSR = sparse.bsr_matrix((C_hat, indices, indptr), shape=(N1,N2))

                    # Combined LHS matrix in BSR format
                    A_BSR = A_BSR + B_BSR - C_BSR

                    # ----------------------------------------------------------------
                    # RHS matrix - x-contiguous
                    # Superdiagonal blocks - D
                    # Swap nx and nz
                    D_hat = torch.swapaxes(D_hat, 4, 2) # (n1,n2,ny,nz,nx)
                    D_hat = D_hat.reshape(n1,n2,Nx)
                    D_hat = torch.permute(D_hat, (2,0,1)).cpu().numpy()
                    # Indices
                    indptr = np.arange(Nx)
                    indptr = np.concatenate([indptr, indptr[-1:]])
                    indices = np.arange(Nx) + 1
                    # BSR
                    D_BSR = sparse.bsr_matrix((D_hat, indices, indptr), shape=(N1,N2))

                    # Subdiagonal blocks - E
                    # Swap nx and nz
                    E_hat = torch.swapaxes(E_hat, 4, 2) # (n1,n2,ny,nz,nx)
                    E_hat = E_hat.reshape(n1,n2,Nx)
                    E_hat = torch.permute(E_hat, (2,0,1)).cpu().numpy()
                    # Indices
                    indptr = np.concatenate([[0,], np.arange(Nx)])
                    indices = np.arange(Nx)
                    # BSR
                    E_BSR = sparse.bsr_matrix((E_hat, indices, indptr), shape=(N1,N2))

                    # Combined RHS matrix in BSR format
                    RHS_BSR = E_BSR - D_BSR
                    
                    # ----------------------------------------------------------------
                    # Initial guess
                    # Remove Dirichlet boundaries from linear system
                    # X
                    if not metrics.periodic_xi:
                        if metrics.iproc==0:
                            k1 = k1[:,1:]
                        if metrics.iproc==metrics.npx-1:
                            k1 = k1[:,:-1]
                    # Y
                    if not metrics.periodic_eta:
                        if metrics.jproc==0:
                            k1 = k1[:,:,1:]
                        if metrics.jproc==metrics.npy-1:
                            k1 = k1[:,:,:-1]
                            
                    # Remove dQ_rhoW
                    R_dt = torch.cat((k1[:3], k1[4:]))  ## NOTE! 2D!  # (neq, nx, ny, nz)
                    R_dt = torch.permute(R_dt, (1, 3, 2, 0)).cpu().numpy() # (nx, nz, ny, neq)

                    dQ = sparse.linalg.spsolve(A_BSR, R_dt.ravel())  # (nx, nz, ny, neq)
                    dQ = dQ.reshape((nx, nz, ny, n1)).swapaxes(0, 2) # (ny, nz, nx, neq)

                    R_dt = R_dt.swapaxes(0, 2) # (ny, nz, nx, neq)

                    # ----------------------------------------------------------------
                    # Iteration
                    for k in range(cfg.DPLR_kmax):
                        # RHS vector
                        R_vec = RHS_BSR @ dQ.ravel() + R_dt.ravel() # (ny, nz, nx, neq)

                        # Solve for dQ
                        R_vec = R_vec.reshape((ny, nz, nx, n1)).swapaxes(0, 2) # (nx, nz, ny, neq)
                        dQ = sparse.linalg.spsolve(A_BSR, R_vec.ravel())

                        dQ = dQ.reshape((nx, nz, ny, n1)).swapaxes(0, 2) # (ny, nz, nx, neq)

                    # ----------------------------------------------------------------
                    # Finalize dQ
                    dQ = dQ.transpose((3, 2, 0, 1)) # (neq, nx, ny, nz)
                    dQ = torch.from_numpy(dQ).to(param.device)
                    
                    # Add back dQ_rhoW
                    dQ = torch.cat((dQ[:-1], 0.0*dQ[-1:], dQ[-1:])) ## NOTE! 2D

                    # Pad dQ with zeros for Dirichlet boundaries
                    # X
                    if not metrics.periodic_xi:
                        if metrics.iproc==0:
                            dQ = torch.cat((0.0*dQ[:,:1], dQ), dim=1)
                        if metrics.iproc==metrics.npx-1:
                            dQ = torch.cat((dQ, 0.0*dQ[:,-1:]), dim=1)
                    # Y
                    if not metrics.periodic_eta:
                        if metrics.jproc==0:
                            dQ = torch.cat((0.0*dQ[:,:,:1], dQ), dim=2)
                        if metrics.jproc==metrics.npy-1:
                            dQ = torch.cat((dQ, 0.0*dQ[:,:,-1:]), dim=2)

                    # Q_{n+1} = Q_n + dQ
                    Q.add( dQ )
                    
                        
                    
                # Apply low-pass filtering
                if (use_filter and (n%cfg.Nsteps_filter==0)):
                    lowpass_filter.apply(Q)

                # Advance time
                t += param.dt
        

    else:
        raise Exception('PyFlowCL.py: solver_mode option '+solver_mode+' not recognized')
        

    # Done time-stepping
    monitor.step(cfg,grid,param,EOS,comms,decomp,n+1,t,param.dt,Q,Q_A,Q_T,time1)

    if (param.rank==0): print('Done solving, elapsed={:9.5f}'.format(time1-time0))

    
    # Enable return values for adjoint verification
    if (perturb is not None):
        return J, Q_A
    else:
        return


# --------------------------------------------------------------
# Update the optimizer state and model parameters
# --------------------------------------------------------------
def optimizer_step(comms, cfg, param, m=0):
    # Average gradients across all processes
    # for param_i in param.model.parameters():
    #     # IMPORTANT: A FACTOR OF -1.0 IS HERE
    #     tensor_i = 1.0*param_i.grad.data.cpu().numpy()
    #     tensor_i = comms.comm.allreduce(tensor_i, op=MPI.SUM)
    #     tensor_i /= float(comms.size)
    #     param_i.grad.data = torch.DoubleTensor( tensor_i ).to(param.device)


    # print('LR  :  {}'.format(print(param.optimizer.param_groups[0]['lr'])))
    # Update the model parameters
    param.optimizer.step() 
    # Save Model
    if (cfg.Save_Model and param.rank==0):
        modelNameFin = cfg.modelName_save + '_' + str(m + cfg.restNum)
        optimizerNameFin = cfg.optimizerName_save + '_' + str(m + cfg.restNum)
        param.model.cpu()
        torch.save( param.model.state_dict(),     modelNameFin)    
        torch.save( param.optimizer.state_dict(), optimizerNameFin)
        param.model.to(param.device)


# --------------------------------------------------------------
# Update the optimizer state and wall model parameters
# --------------------------------------------------------------
def optimizer_step_wm(comms, cfg, param, m=0, modelWall = None, optimizer=None):
    # Average gradients across all processes
    # for param_i in param.model.parameters():
    #     # IMPORTANT: A FACTOR OF -1.0 IS HERE
    #     tensor_i = 1.0*param_i.grad.data.cpu().numpy()
    #     tensor_i = comms.comm.allreduce(tensor_i, op=MPI.SUM)
    #     tensor_i /= float(comms.size)
    #     param_i.grad.data = torch.DoubleTensor( tensor_i ).to(param.device)


    # print('LR  :  {}'.format(print(param.optimizer.param_groups[0]['lr'])))
    # Update the model parameters
    optimizer.step() 
    # Save Model
    if (cfg.Save_Model and param.rank==0):
        modelNameFin = cfg.modelName_save + '_wallModel_' + str(m + cfg.restNum)
        optimizerNameFin = cfg.optimizerName_save + '_wallModel_' + str(m + cfg.restNum)
        modelWall.cpu()
        torch.save( modelWall.state_dict(),     modelNameFin)    
        torch.save( optimizer.state_dict(), optimizerNameFin)
        modelWall.to(param.device)


# --------------------------------------------------------------
# Predict CFL number and limit time-step size
# --------------------------------------------------------------
def predict_dt(grid, param, EOS, comms, q_cons):
    # Primitives (Cartesian)
    rho = q_cons['rho'].interior()
    u   = q_cons['rhoU'].interior()/rho
    v   = q_cons['rhoV'].interior()/rho
    w   = q_cons['rhoW'].interior()/rho

    # Transformed velocity components
    u_tilde = grid.xi_x_Jac[:,:,None]  * u + grid.xi_y_Jac[:,:,None]  * v
    v_tilde = grid.eta_x_Jac[:,:,None] * u + grid.eta_y_Jac[:,:,None] * v

    # Cartesian sound speed
    c = EOS.get_soundspeed_q(q_cons)

    # Transformed sound speeds
    c_trans_u = c * torch.sqrt(grid.xi_x_Jac**2  + grid.xi_y_Jac**2)[:,:,None]
    c_trans_v = c * torch.sqrt(grid.eta_x_Jac**2 + grid.eta_y_Jac**2)[:,:,None]

    # Maximum right-running wave speeds
    u_tilde_max_dx = torch.max((u_tilde + c_trans_u) * grid.inv_trans_Jac[:,:,None]) / grid.d_xi
    v_tilde_max_dx = torch.max((v_tilde + c_trans_v) * grid.inv_trans_Jac[:,:,None]) / grid.d_eta
    #w_max_dx = torch.max(w + c) / grid.d_z
    #u_max_dx = torch.max(torch.max(u_tilde_max_dx, v_tilde_max_dx), w_max_dx)
    u_max_dx = torch.max(u_tilde_max_dx, v_tilde_max_dx)

    # CFL number
    CFL = comms.parallel_max( u_max_dx.cpu().numpy() ) * param.dt

    # Predict the new time step size based on dt, max_dt, CFL, and max_CFL
    dt_old = param.dt
    if (CFL==0): 
        dt = param.max_dt
    else:
        dt = min(param.max_CFL/CFL*dt_old, param.max_dt)
    if (dt>dt_old): 
        alpha = 0.7
        dt = alpha*dt + (1.0-alpha)*dt_old

    # Save to global data
    param.CFL = CFL
    param.dt  = dt


    
def predict_dt_old(grid, param, EOS, comms, q_cons):
    # Compute primitives
    rho = q_cons['rho'].interior()
    u   = q_cons['rhoU'].interior()/rho
    v   = q_cons['rhoV'].interior()/rho
    w   = q_cons['rhoW'].interior()/rho

    # Max local velocity
    u_max_dx = torch.maximum( u/grid.Dx[:,:,None], torch.maximum( v/grid.Dy[:,:,None], w/grid.Dz ))
    u_max_dx = torch.amax(u_max_dx, dim=(0,1,2)).cpu().numpy()

    # Max local sound speed
    min_dx   = torch.minimum(grid.Dx,
                             torch.minimum(grid.Dy,
                                           torch.tensor((grid.Dz,),dtype=param.WP).to(param.device)))
    c_max_dx = EOS.get_soundspeed_q(q_cons) / min_dx[:,:,None]
    c_max_dx = torch.amax(c_max_dx, dim=(0,1,2)).cpu().numpy()
    
    # CFL number
    CFL = comms.parallel_max( max( u_max_dx, c_max_dx ) ) * param.dt
    #CFL = comms.parallel_max( u_max_dx + c_max_dx ) * param.dt

    # Predict the new time step size based on dt, max_dt, CFL, and max_CFL
    dt_old = param.dt
    if (CFL==0): 
        dt = param.max_dt
    else:
        dt = min(param.max_CFL/CFL*dt_old, param.max_dt)
    if (dt>dt_old): 
        alpha = 0.7
        dt = alpha*dt + (1.0-alpha)*dt_old

    # Save to global data
    param.CFL = CFL
    param.dt  = dt
    if False:
        if (param.rank==0):print('--- dt = {} ---'.format(dt))
