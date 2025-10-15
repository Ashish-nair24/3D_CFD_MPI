"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file SPCL_Diffusion.py
@author Jonathan F. MacArt

"""

__copyright__ = """
Copyright (c) 2023 Jonathan F. MacArt
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
import time
from mpi4py import MPI

from .. import Grid, Metrics, Data
from ..Library import Parallel

from scipy.linalg import solve_banded


# ----------------------------------------------------
# Lightweight class for running parameters
# ----------------------------------------------------
class Param:
    def __init__(self, cfg, decomp, WP, WP_np):
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

        # Diffusivity (constant)
        self.alpha = cfg.alpha

            
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

    # Extract parameters from the input config
    param = Param(cfg, decomp, WP, WP_np)

    # Initial CFL/DN estimates
    dx_min = min(torch.amin(grid.Dx), torch.amin(grid.Dy)).cpu().numpy()
    DN  = param.alpha * param.dt / dx_min**2
    if (param.rank==0): print('Initial DN = {:7.3e}'.format(DN))

        
    # --------------------------------------------------------------
    # Initial condition
    
    names = ['U',]
    Q = Data.State(names, decomp)
    
    t = 0; Nstart = 0
    
    if (cfg.dfName_read is not None):
        # Load restart file
        Nstart, t, dt_tmp = Data.read_data(cfg.dfName_read, cfg, decomp, Q)
        if (dt_tmp > 1e-16): param.dt = dt_tmp
        if (decomp.rank==0):
            print(' --> Restarting from {} at it={}, t={:9.4e}'.format(cfg.dfName_read,Nstart,t))

    else:
        # Get ICs
        #IC.get_IC(cfg.IC_opt, grid, Q, param, EOS)

        thk = 0.2
        Q['U'].copy( torch.tanh(grid.Eta[:,:,None]/thk) * (1.0 - grid.Xi[:,:,None]) +
                     (torch.sin(16* np.pi * grid.Eta[:,:,None])/2 + 0.5) * grid.Xi[:,:,None] )

        ## TBD
        
        
    
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

    if (solver_mode=='implicit_euler' or
        solver_mode=='steady_bicgstab' or
        solver_mode=='line_relaxation'):
        jvp = True
    else:
        jvp = False
        
    #rhs = RHS(grid, metrics, param, jvp).diffusion_2D
    # Testing
    RHS_C = RHS(grid, metrics, param, jvp)
    rhs     = RHS_C.diffusion_2D_2CD
    rhs_xi  = RHS_C.diffusion_2D_2CD_xi
    rhs_eta = RHS_C.diffusion_2D_2CD_eta
    rhs_mix = RHS_C.diffusion_2D_2CD_mix

    # Allocate RK4 state memory
    Q_tmp = Data.State(names, decomp)
    
    # Timing
    time1 = time.time()
    time0 = time1

    if (solver_mode == 'unsteady_RK4'):   # --------------------------------------------------------------------
        # All forward simulations (unsteady & steady pseudo-time); unsteady adjoint simulations
        
        for n in range(Nstart, Nstart+cfg.Nsteps):

            if ((n-Nstart)%cfg.N_monitor == 0):
                q_cpu = torch.unsqueeze(Q['U'].interior(), dim=0).cpu().numpy()
                Data.write_data(cfg, grid, decomp, q_cpu, names, n, t, param.dt)
                   
            with torch.inference_mode():
                k1 = rhs( Q['U'].var ); Q_tmp.copy_sum( Q, param.dt * 0.5 * k1 )
                k2 = rhs( Q_tmp['U'].var ); Q_tmp.copy_sum( Q, param.dt * 0.5 * k2 )
                k3 = rhs( Q_tmp['U'].var ); Q_tmp.copy_sum( Q, param.dt * k3 )
                k4 = rhs( Q_tmp['U'].var )
                
                # RK4 update
                k2 *= 2.0
                k3 *= 2.0
                k1 += k2
                k1 += k3
                k1 += k4
                k1 /= 6.0
                Q.add( param.dt * k1 )

                # Advance time
                t += param.dt

            if ((n+1)%cfg.N_monitor==0 and decomp.rank==0):
                time1 = monitor(n+1, t, Q, param.dt*k1[0], param, time1)
                
        # DONE FORWARD LOOP
        

    elif (solver_mode == 'explicit'):   # -----------------------------------------
        for n in range(Nstart, Nstart+cfg.Nsteps):

            if ((n-Nstart)%cfg.N_monitor == 0):
                q_cpu = torch.unsqueeze(Q['U'].interior(), dim=0).cpu().numpy()
                Data.write_data(cfg, grid, decomp, q_cpu, names, n, t, param.dt)
                   
            with torch.inference_mode():
                k1 = rhs( Q['U'].var )
                Q.add( param.dt * k1 )
                
                #k1  = rhs_exp( Q['U'].var )
                #k1 += rhs_imp( Q['U'].var )
                #Q.add( param.dt * k1 )

                # RK2 - midpoint method
                #k1 = rhs( Q['U'].var )
                #Q_tmp.copy_sum( Q, param.dt * 0.5 * k1 )
                #k1 = rhs( Q_tmp['U'].var )
                #Q.add( param.dt * k1 )

                # Advance time
                t += param.dt

            if ((n+1)%cfg.N_monitor==0 and decomp.rank==0):
                time1 = monitor(n+1, t, Q, param.dt*k1[0], param, time1)
                
        # DONE FORWARD LOOP
        

    elif (solver_mode == 'line_relaxation'):   # -----------------------------------------
        for n in range(Nstart, Nstart+cfg.Nsteps):

            if ((n-Nstart)%cfg.N_monitor == 0):
                q_cpu = torch.unsqueeze(Q['U'].interior(), dim=0).cpu().numpy()
                Data.write_data(cfg, grid, decomp, q_cpu, names, n, t, param.dt)
                   
            with torch.inference_mode():
                _dt = param.dt
                
                # Xi terms
                k1_xi = rhs_xi( Q['U'].var )[0] * _dt

                # Coordinate transforms on interior
                a = (RHS_C.xi_xx + RHS_C.xi_yy)/(2.0*RHS_C.d_xi) * param.alpha
                b = (RHS_C.xi_x**2 + RHS_C.xi_y**2)/RHS_C.d_xi2 * param.alpha

                x = Q['U'].interior()
                
                # Remove Dirichlet boundaries
                a = a[1:-1,1:-1,:]
                b = b[1:-1,1:-1,:]
                k1_xi = k1_xi[1:-1,1:-1,:]
                a = torch.swapaxes(a, 1, 0).contiguous()
                b = torch.swapaxes(b, 1, 0).contiguous()
                k1_xi = torch.swapaxes(k1_xi, 1, 0).contiguous()
                
                # Tridiagonal matrix
                L = -_dt*(b - a)
                D = 1.0 + _dt*2.0*b
                U = -_dt*(b + a)
                
                L[:,:-1,:] = L[:,1:,:]
                L[:,-1,:] = 0.0
                U[:,1:,:] = U[:,:-1,:]
                U[:,0,:] = 0.0
                
                Tri = torch.stack((U.ravel(), D.ravel(), L.ravel()), dim=0).cpu().numpy()
                
                # RHS vector
                B = k1_xi.ravel()
                
                x = solve_banded((1,1), Tri, B)
                x = torch.from_numpy(x).reshape(param.ny_-2, param.nx_-2, param.nz_)
                x = torch.swapaxes(x, 1, 0).contiguous()
                x = torch.nn.functional.pad(x, (0,0,1,1,1,1), "constant", 0)
                
                Q.add(x[None,...])

                dQ = x
                

                # Eta terms
                k1_eta = rhs_eta( Q['U'].var )[0] * _dt

                # Coordinate transforms on interior
                a = (RHS_C.eta_xx + RHS_C.eta_yy)/(2.0*RHS_C.d_eta) * param.alpha
                b = (RHS_C.eta_x**2 + RHS_C.eta_y**2)/RHS_C.d_eta2 * param.alpha

                x = Q['U'].interior()

                # Remove Dirichlet boundaries
                a = a[1:-1,1:-1,:]
                b = b[1:-1,1:-1,:]
                k1_eta = k1_eta[1:-1,1:-1,:]
                
                # Tridiagonal matrix
                L = -_dt*(b - a)
                D = 1.0 + _dt*2.0*b
                U = -_dt*(b + a)

                L[:,:-1,:] = L[:,1:,:]
                L[:,-1,:] = 0.0
                U[:,1:,:] = U[:,:-1,:]
                U[:,0,:] = 0.0
                
                Tri = torch.stack((U.ravel(), D.ravel(), L.ravel()), dim=0).cpu().numpy()

                # RHS vector
                B = k1_eta.ravel()
                
                x = solve_banded((1,1), Tri, B)
                x = torch.from_numpy(x).reshape(param.nx_-2, param.ny_-2, param.nz_)
                x = torch.nn.functional.pad(x, (0,0,1,1,1,1), "constant", 0)
                
                Q.add(x[None,...])

                dQ += x

                # Cross terms
                k1_mix = rhs_mix( Q['U'].var )
                Q.add(_dt * k1_mix)
                dQ += _dt * k1_mix[0]

                # Advance time
                t += param.dt

            if ((n+1)%cfg.N_monitor==0 and decomp.rank==0):
                time1 = monitor(n+1, t, Q, dQ, param, time1)
                
        # DONE FORWARD LOOP
            

    elif (solver_mode == 'implicit_euler'):    # ----------------------------------------------------------------
        # Implicit Euler solution
        for n in range(Nstart, Nstart+cfg.Nsteps):

            if ((n-Nstart)%cfg.N_monitor == 0):
                q_cpu = torch.unsqueeze(Q['U'].interior(), dim=0).cpu().numpy()
                Data.write_data(cfg, grid, decomp, q_cpu, names, n, t, param.dt)
                
            F = Q['U'].var.clone()

            M_x = lambda xVec: ((-(torch.autograd.functional.jvp(rhs, F.detach(), padReshp(xVec, param))[1]) +
                                 (1/param.dt)*padReshp(xVec, param))[5:-5,5:-5,:]).flatten()

            Flux = rhs(F)[5:-5,5:-5,:].flatten()

            # Solve
            delq = bicgstabHB(M_x, Flux.detach(), maxiter=10)

            # Update
            delq = padReshp(delq, param)
            F += delq

            Q['U'].copy_full(F)

            # Advance time
            t += param.dt
            
            time1 = monitor(n, t, Q, delq, param, time1)
            

    elif (solver_mode == 'steady_bicgstab'):    # ----------------------------------------------------------------
        # Steady solution using bicgstab 
        n = 0
        F = Q['U'].var.clone()

        M_x = lambda xVec: ((-(torch.autograd.functional.jvp(rhs,
                                                             F.detach(),
                                                             padReshp(xVec, param))[1]))[5:-5,5:-5,:]).flatten()

        Flux = rhs(F)[5:-5,5:-5,:].flatten()

        # Solve
        delq = bicgstabHB(M_x, Flux.detach(), maxiter=10000)

        # Update
        delq = padReshp(delq, param)
        F += delq

        Q['U'].copy_full(F)

        time1 = monitor(n, t, Q, delq, param, time1)

    else:
        raise Exception('PyFlowCL.py: solver_mode option '+solver_mode+' not recognized')
        

    if (param.rank==0): print('Done solving, elapsed={:9.5f}'.format(time1-time0))
    
    q_cpu = torch.unsqueeze(Q['U'].interior(), dim=0).cpu().numpy()
    Data.write_data(cfg, grid, decomp, q_cpu, names, n, t, param.dt)



# --------------------------------------------------------------
# Right-hand-side function
# --------------------------------------------------------------
class RHS:
    def __init__(self, grid, metrics, param, jvp=False):
        self.grid = grid
        self.metrics = metrics
        self.param = param
        self.jvp = jvp
        
        self.imin_ = param.imin_;  self.imax_ = param.imax_
        self.jmin_ = param.jmin_;  self.jmax_ = param.jmax_
        self.kmin_ = param.kmin_;  self.kmax_ = param.kmax_
        
        self.d_xi  = grid.d_xi
        self.d_eta = grid.d_eta
        
        self.d_xi2  = grid.d_xi**2
        self.d_eta2 = grid.d_eta**2

        self.xi_x = grid.xi_x[...,None]
        self.xi_y = grid.xi_y[...,None]
        self.eta_x = grid.eta_x[...,None]
        self.eta_y = grid.eta_y[...,None]
        
        self.xi_xx = grid.xi_xx
        self.xi_yy = grid.xi_yy
        self.eta_xx = grid.eta_xx
        self.eta_yy = grid.eta_yy

        return

    def diffusion_2D(self, U):
        # Assumes U contains full overlaps
        
        # Compute 1st derivatives - extended interior for 2nd derivatives
        dU_dx, dU_dy, _ = self.metrics.grad_node(U, compute_extended=True)

        # Compute 2nd derivatives
        d2U_dx2, _, _ = self.metrics.grad_node(dU_dx, extended_input=True, compute_dy=False)
        _, d2U_dy2, _ = self.metrics.grad_node(dU_dy, extended_input=True, compute_dx=False)

        qdot = self.param.alpha * (d2U_dx2 + d2U_dy2)

        # Dirichlet walls
        qdot[0,:,:] = 0.0
        qdot[:,0,:] = 0.0
        qdot[-1,:,:] = 0.0
        qdot[:,-1,:] = 0.0

        if self.jvp:
            # Padding to maintain dimensional consistency 
            R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
            return R(qdot)
        else:
            return qdot[None,...]


    def grad_xi_2CD(self, u):
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = self.kmin_
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = self.kmax_
        
        # 2CD computational-space first and second derivatives
        # assuming: no periodic boundaries, no MPI decomp
        # xi
        u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
        u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
        u2m0 = u[imin_+2:imax_  ,jmin_:jmax_,kmin_:kmax_]
        u1m1 = u[imin_+1:imax_-1,jmin_:jmax_,kmin_:kmax_]
        u0m2 = u[imin_  :imax_-2,jmin_:jmax_,kmin_:kmax_]
        um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
        um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]
        
        u_xi = torch.cat(( (u1 - u0)/self.d_xi,
                           (u2m0 - u0m2)/(2.0 * self.d_xi),
                           (um2 - um1)/self.d_xi ), dim=0)

        u_xi_xi = torch.cat(( (u1 - u0)/self.d_xi2,
                              (u2m0 - 2.0*u1m1 + u0m2)/self.d_xi2,
                              (um2 - um1)/self.d_xi2 ), dim=0)
        
        return u_xi, u_xi_xi


    def grad_mixed_2CD(self, u):
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = self.kmin_
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = self.kmax_
        
        # 2CD computational-space first and second derivatives
        # assuming: no periodic boundaries, no MPI decomp
        # xi
        u0   = u[imin_  :imin_+1,jmin_:jmax_,kmin_:kmax_]
        u1   = u[imin_+1:imin_+2,jmin_:jmax_,kmin_:kmax_]
        u2m0 = u[imin_+2:imax_  ,jmin_:jmax_,kmin_:kmax_]
        u1m1 = u[imin_+1:imax_-1,jmin_:jmax_,kmin_:kmax_]
        u0m2 = u[imin_  :imax_-2,jmin_:jmax_,kmin_:kmax_]
        um2  = u[imax_-2:imax_-1,jmin_:jmax_,kmin_:kmax_]
        um1  = u[imax_-1:imax_  ,jmin_:jmax_,kmin_:kmax_]

        # Mixed second derivative
        u_xi_eta = torch.cat(( (u1 - u0),
                               (u2m0 - 2.0*u1m1 + u0m2),
                               (um2 - um1) ), dim=0)
        # eta
        u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
        u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
        u2m0 = u[imin_:imax_,jmin_+2:jmax_  ,kmin_:kmax_]
        u1m1 = u[imin_:imax_,jmin_+1:jmax_-1,kmin_:kmax_]
        u0m2 = u[imin_:imax_,jmin_  :jmax_-2,kmin_:kmax_]
        um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
        um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
        
        # Finalize mixed second derivative
        u_xi_eta += torch.cat(( (u1 - u0),
                                (u2m0 - 2.0*u1m1 + u0m2),
                                (um2 - um1) ), dim=1)
        u_xi_eta /= (self.d_xi * self.d_eta)
        
        return u_xi_eta


    def grad_eta_2CD(self, u):
        imin_ = self.imin_; jmin_ = self.jmin_; kmin_ = self.kmin_
        imax_ = self.imax_; jmax_ = self.jmax_; kmax_ = self.kmax_
        
        # 2CD computational-space first and second derivatives
        # assuming: no periodic boundaries, no MPI decomp
        # eta
        u0   = u[imin_:imax_,jmin_  :jmin_+1,kmin_:kmax_]
        u1   = u[imin_:imax_,jmin_+1:jmin_+2,kmin_:kmax_]
        u2m0 = u[imin_:imax_,jmin_+2:jmax_  ,kmin_:kmax_]
        u1m1 = u[imin_:imax_,jmin_+1:jmax_-1,kmin_:kmax_]
        u0m2 = u[imin_:imax_,jmin_  :jmax_-2,kmin_:kmax_]
        um2  = u[imin_:imax_,jmax_-2:jmax_-1,kmin_:kmax_]
        um1  = u[imin_:imax_,jmax_-1:jmax_  ,kmin_:kmax_]
        
        u_eta = torch.cat(( (u1 - u0)/self.d_eta,
                            (u2m0 - u0m2)/(2.0 * self.d_eta),
                            (um2 - um1)/self.d_eta ), dim=1)

        u_eta_eta = torch.cat(( (u1 - u0)/self.d_eta2,
                                (u2m0 - 2.0*u1m1 + u0m2)/self.d_eta2,
                                (um2 - um1)/self.d_eta2 ), dim=1)

        return u_eta, u_eta_eta

    # All terms
    def diffusion_2D_2CD(self, u):
        # Get derivatives
        u_xi, u_xi_xi    = self.grad_xi_2CD(u)
        u_xi_eta         = self.grad_mixed_2CD(u)
        u_eta, u_eta_eta = self.grad_eta_2CD(u)
        
        # Compute physical-space 2nd derivatives explicitly
        F_xi = ((self.xi_xx + self.xi_yy) * u_xi +
                (self.xi_x**2 + self.xi_y**2) * u_xi_xi)
        
        F_eta = ((self.eta_xx + self.eta_yy) * u_eta +
                 (self.eta_x**2 + self.eta_y**2) * u_eta_eta)

        F_xi_eta = 2.0*(self.xi_x * self.eta_x +
                        self.xi_y * self.eta_y) * u_xi_eta

        qdot = self.param.alpha * (F_xi + F_eta + F_xi_eta)
        
        # Dirichlet walls
        qdot[0,:,:] = 0.0
        qdot[:,0,:] = 0.0
        qdot[-1,:,:] = 0.0
        qdot[:,-1,:] = 0.0

        return qdot[None,...]

    # Only xi terms
    def diffusion_2D_2CD_xi(self, u):
        # Get derivatives
        u_xi, u_xi_xi = self.grad_xi_2CD(u)
        
        # Compute physical-space 2nd derivatives explicitly
        F_xi = ((self.xi_xx + self.xi_yy) * u_xi +
                (self.xi_x**2 + self.xi_y**2) * u_xi_xi)

        qdot = self.param.alpha * F_xi
        
        # Dirichlet walls
        qdot[0,:,:] = 0.0
        qdot[:,0,:] = 0.0
        qdot[-1,:,:] = 0.0
        qdot[:,-1,:] = 0.0

        return qdot[None,...]

    # Only mixed terms
    def diffusion_2D_2CD_mix(self, u):
        # Get derivatives
        u_xi_eta = self.grad_mixed_2CD(u)
        
        # Compute physical-space 2nd derivatives explicitly
        F_xi_eta = 2.0*(self.xi_x * self.eta_x +
                        self.xi_y * self.eta_y) * u_xi_eta

        qdot = self.param.alpha * F_xi_eta
        
        # Dirichlet walls
        qdot[0,:,:] = 0.0
        qdot[:,0,:] = 0.0
        qdot[-1,:,:] = 0.0
        qdot[:,-1,:] = 0.0

        return qdot[None,...]

    # Only eta terms
    def diffusion_2D_2CD_eta(self, u):
        # Get derivatives
        u_eta, u_eta_eta = self.grad_eta_2CD(u)
        
        # Compute physical-space 2nd derivatives explicitly
        F_eta = ((self.eta_xx + self.eta_yy) * u_eta +
                 (self.eta_x**2 + self.eta_y**2) * u_eta_eta)

        qdot = self.param.alpha * F_eta
        
        # Dirichlet walls
        qdot[0,:,:] = 0.0
        qdot[:,0,:] = 0.0
        qdot[-1,:,:] = 0.0
        qdot[:,-1,:] = 0.0

        return qdot[None,...]
        
    
# padding and reshaping function 
def padReshp(xVec, param, pad=True):
    # Reshape
    xVec = xVec.reshape(param.nx_, param.ny_, param.nz_)
    
    # Padding
    if pad:
        R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
        xVec = R(xVec)
        
    return xVec


# --------------------------------------------------------------
# BiCGStab solver
# --------------------------------------------------------------      
def bicgstabHB(jvp, F, maxiter=5):
    
    # Algo 7.7 from Yousuf Saad
    
    # Step 1. Initial guess
    x = torch.zeros_like(F)
    r = F - jvp(x)
    rStar = r
    
    # Step 2.
    p = r
    
    # Step 3.
    for i in range(maxiter):
        
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
        
        if torch.linalg.norm(r) <= 1e-16:
            break
        
    print('it=',i,', norm=',torch.linalg.norm(r))
        
    return x


def monitor(n, t, Q, k1, param, time1):
    maxU = torch.amax(torch.abs(Q['U'].interior()), dim=(0,1,2)).cpu().numpy()
    resU = torch.amax(torch.abs(k1), dim=(0,1,2)).cpu().numpy()
    time2 = time.time()
    print(f'{n: 8d}, t = {t:.4e}, max u = {maxU:.4e}, res U = {resU:.4e}, time = {time2-time1:.4f}')
    time1 = time2
    return time1
