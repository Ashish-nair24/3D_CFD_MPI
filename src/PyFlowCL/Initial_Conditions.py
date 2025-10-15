"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Initial_Conditions.py

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

#import matplotlib.pyplot as plt
import numpy as np
import torch
from .Utilities import Shock_Relations as shock
from scipy.optimize import root
import h5py
from . import Data

# ----------------------------------------------------
# Top-level IC driver
#   This calls a function named "get_IC_<IC_opt>"
#   For new cases, you need to supply the function!
# ----------------------------------------------------
def get_IC(IC_opt, grid, Q, param, EOS):
    return globals()['get_IC_'+IC_opt](grid, Q, param, EOS)

# ----------------------------------------------------
# Initial conditions - uniform flow
# ----------------------------------------------------
def get_IC_uniform(grid, Q, param, EOS):
    rho = torch.tensor((EOS.rho0,), dtype=param.WP).to(param.device)
    Q['rho'].copy( rho )
    Q['rhoU'].copy( EOS.U0 * rho )
    Q['rhoV'].copy( 0.0*rho )
    Q['rhoW'].copy( 0.0*rho )

    # Internal energy from pressure
    p = EOS.p0
    e = EOS.get_internal_energy_RPY(rho, p)
    
    # Finalize total energy
    Q['rhoE'].copy(rho*e + 0.5*(Q['rhoU'].interior()**2 +
                                Q['rhoV'].interior()**2 +
                                Q['rhoW'].interior()**2)/rho)
    del e
    
    return

# ----------------------------------------------------
# Initial conditions - uniform flow with an
# isothermal lower wall
# ----------------------------------------------------
def get_IC_uniform_isothermal_wall(grid, Q, param, EOS):

    # Version 2: Explicit shock position; post-shock stagnation
    
    # Centerline function: Tapers p, T, rho gradients to zero away from centerline
    cl_thk = 0.2
    #cl_fn  = torch.exp(-(grid.Xi[:,:,None] - 0.5)**2 / cl_thk**2)
    cl_fn = 1.0

    # Shock jump function: Positions the normal shock jump relations a distance "shock_pos" (in eta-units)
    # away from the lower wall
    shock_pos = 0.2
    shock_thk = 0.05
    shock_fn = 0.5*(1.0 - torch.tanh((grid.Eta[:,:,None] - shock_pos)/shock_thk)) * cl_fn

    # Normal-shock jump relations
    T1 = EOS.T0
    p1 = EOS.p0
    T2 = T1 * shock.jump_T(EOS.Ma, EOS.gamma)
    p2 = p1 * shock.jump_P(EOS.Ma, EOS.gamma)
    M2 = shock.jump_M2(EOS.Ma, EOS.gamma)

    # T, p across the shock
    T = T1 + (T2 - T1) * shock_fn
    p = p1 + (p2 - p1) * shock_fn

    # Stagnation function: blends the post-shock quantities to stagnation quantities at eta=0
    thk = 0.05
    stag_fn = 1.0 - torch.tanh(grid.Eta[:,:,None]/thk)
    
    # Post-shock stagnation p, T at lower wall
    T0 = T2 * shock.isen_T_ratio(M2, EOS.gamma)
    p0 = p2 * shock.isen_p_ratio(M2, EOS.gamma)
    T += (T0 - T2) * stag_fn * cl_fn
    p += (p0 - p2) * stag_fn * cl_fn

    # rho, e from pressure and temperature
    rho = EOS.get_density(p, T)
    e = EOS.get_internal_energy_TY(T)

    # Velocity goes to zero at the wall
    u = EOS.U0 * (1.0 - stag_fn)
    
    Q['rho'].copy( rho )
    Q['rhoU'].copy( rho * u )
    Q['rhoV'].copy( 0.0*rho )
    Q['rhoW'].copy( 0.0*rho )
    
    # Finalize total energy
    Q['rhoE'].copy( rho*e + 0.5*(Q['rhoU'].interior()**2 +
                                 Q['rhoV'].interior()**2 +
                                 Q['rhoW'].interior()**2)/rho )
    return

# ----------------------------------------------------
# Initial conditions - field for turbulent spatial planar jet 3D + std k-epsilon
# ----------------------------------------------------
def get_IC_planar_jet_spatial_RANS(grid, Q, param, EOS):
    #Does not work with MPI
    
        shear_layer_thickness = 0.03
        u_y =  (0.5*(torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 - 1.0)/shear_layer_thickness))).to(param.device) #giving tanh profile

        U = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)
        V = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)

        U[0,:,:] = 0.0
        U[-1,:,:] = 0.0
        U[0,:,:] = U[0,:,:] + u_y[:,None]

        V[:,0,:] = 0.04
        V[:,-1,:] = -0.04  #giving entrianment velocity

        avg_in = (torch.sum(torch.mean((U[0,:]),dim = 1)*grid.Dy[0,:]) + 2*torch.sum(torch.mean((V[:,0]),dim = 0)*grid.Dx[:,0])).to(param.device)
        print("inlet_massflow_averaged",avg_in)   # Should be 1
        
        U[-1,:,:] = U[-1,:,:] + avg_in/torch.sum(grid.Dy[-1,:])
        print(torch.sum(grid.Dy[-1,:]))

        avg_out = torch.sum(torch.mean((U[-1,:]),dim = 1)*grid.Dy[-1,:])
        print("outlet_mass_flow",avg_out)  # should be equal to inlet

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*U)
        Q['rhoV'].copy( V*rho )
        Q['rhoW'].copy( 0.0*rho )  
        Q['rhok'].copy( 3.2*(0.1)**2*rho) # 0.1 is turbulent intensity  (throughout the field)
  
        Q['rhok'].interior()[:,0] = 0.001
        Q['rhok'].interior()[:,-1] = 0.001  #giving some value at boundaries
        Q['rhoeps'].copy( 0.09**(0.75)*(Q['rhok'].interior()/Q["rho"].interior())**(1.5)*rho ) #from openfoam 
        
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) + (Q['rhok'].interior()/Q["rho"].interior()) ))
        del e
        
        return

# ----------------------------------------------------
# Initial conditions - field for turbulent spatial planar jet 3D + std k-epsilon
# ----------------------------------------------------
def get_IC_planar_jet_spatial_RANS_coflow(grid, Q, param, EOS):
    #Does not work with MPI, build initial file in serial on CPU
    
        shear_layer_thickness = 0.03
        
        Cf_c1 = 18.5  # centre points for the co-flow jets. 
        Cf_c2 = 21.5 # centre points for the co-flow jets. 
        
        #u_y = 1/3.88 + 0.5*(torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 - 1.0)/shear_layer_thickness))
        u_y =   0.5/3.88 * (torch.tanh(((grid.Y[0,:] - Cf_c1) * 2.0 + 1.43)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - Cf_c1) * 2.0 - 1.43)/shear_layer_thickness)) \
            + 0.5*(torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5) * 2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5) * 2.0 - 1.0)/shear_layer_thickness)) \
            +  0.5/3.88 * (torch.tanh(((grid.Y[0,:] - Cf_c2) * 2.0 + 1.43)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - Cf_c2) * 2.0 - 1.43)/shear_layer_thickness)).to(param.device) #giving tanh profile

        U = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)
        V = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)

        U[0,:,:] = 0.0
        U[-1,:,:] = 0.0
        U[0,:,:] = U[0,:,:] + u_y[:,None].to(param.device)

        V[:,0,:] = 0.04
        V[:,-1,:] = -0.04  #giving entrianment velocity

        avg_in = (torch.sum(torch.mean((U[0,:]),dim = 1)*grid.Dy[0,:]) + 2*torch.sum(torch.mean((V[:,0]),dim = 0)*grid.Dx[:,0])).to(param.device)
        print("inlet_massflow_averaged",avg_in)   # Should be 1
        
        U[-1,:,:] = U[-1,:,:] + avg_in/torch.sum(grid.Dy[-1,:])
        print(torch.sum(grid.Dy[-1,:]))

        avg_out = torch.sum(torch.mean((U[-1,:]),dim = 1)*grid.Dy[-1,:])
        print("outlet_mass_flow",avg_out)  # should be equal to inlet

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*U)
        Q['rhoV'].copy( V*rho )
        Q['rhoW'].copy( 0.0*rho )  
        Q['rhok'].copy( 3.2*(0.1)**2*rho) # 0.1 is turbulent intensity  (throughout the field)
  
        Q['rhok'].interior()[:,0] = 0.001
        Q['rhok'].interior()[:,-1] = 0.001  #giving some value at boundaries
        
        #u_y =  (0.5*(torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:] - grid.Lx2*0.5)*2.0 - 1.0)/shear_layer_thickness))  ).to(param.device)   
        
        #Q['rhok'].interior()[0,:,0] = 3.2*(0.1)**2*rho
        
        Q['rhoeps'].copy( 0.09**(0.75)*(Q['rhok'].interior()/Q["rho"].interior())**(1.5)*rho ) #from openfoam 
        
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = (p / ((param.gamma-1.0) * rho * param.Ma**2)).to(param.device)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) + (Q['rhok'].interior()/Q["rho"].interior()) ))
        del e
        
        return
    
# ----------------------------------------------------
# Initial conditions - field for turbulent spatial planar jet 3D with coflow
# ----------------------------------------------------
def get_IC_planar_jet_spatial_turb_coflow(grid, Q, param, EOS):
    #Does not work with MPI, build initial file in serial
    
        shear_layer_thickness = 0.03
        Cf_c1 = 18.5  # centre points for the co-flow jets. 
        Cf_c2 = 21.5 # centre points for the co-flow jets. 
        
        u_y =   0.5/3.88 * (torch.tanh(((grid.Y[0,:,0] - Cf_c1) * 2.0 + 1.43)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:,0] - Cf_c1) * 2.0 - 1.43)/shear_layer_thickness)) \
            + 0.5*(torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5) * 2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5) * 2.0 - 1.0)/shear_layer_thickness)) \
            +  0.5/3.88 * (torch.tanh(((grid.Y[0,:,0] - Cf_c2) * 2.0 + 1.43)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:,0] - Cf_c2) * 2.0 - 1.43)/shear_layer_thickness))
        
        U_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        W_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        V_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0

        U = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        W = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        V = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0

        U[0,:,:] = 0.0
        U[-1,:,:] = 0.0
        U[0,:,:] = U[0,:,:] + 1.0*u_y[:,None]

        V[:,0,:] = 0.04   #giving entrianment velocity
        V[:,-1,:] = -0.04

        avg_in = torch.sum(torch.mean((U[0,:,:]),dim = 1)*grid.Dy[0,:]) + 2*torch.sum((torch.mean(V[:,0,:],dim=1))*grid.Dx[:,0])
        print("inlet_mass_flow",avg_in) # Should be 1

        W[0,190:210,:] = (W_fluc_temp[0,190:210,:])
        
        W[0,190:210,:] = (W_fluc_temp[0,190:210,:])
        
        U[-1,:,:] = U[-1,:,:] + avg_in/torch.sum(grid.Dy[-1,:])

        avg_out = torch.sum(torch.mean((U[-1,:]),dim = 1)*grid.Dy[-1,:])
        print("outlet_mass_flow",avg_out) #should be equal to inlet, conserving mass

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*U)
        Q['rhoV'].copy( V*rho )
        Q['rhoW'].copy( W*rho )     
      
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) ))
        del e
        
        return
    
# ----------------------------------------------------
# Initial conditions - field for turbulent spatial planar jet 3D
# ----------------------------------------------------
def get_IC_planar_jet_spatial_turb(grid, Q, param, EOS):
    #Does not work with MPI, build initial file in serial
    
        shear_layer_thickness = 0.03
        u_y =  (0.5*(torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5)*2.0 + 1.0)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5)*2.0 - 1.0)/shear_layer_thickness))).to(param.device)
        
        #U_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
       # W_fluc_temp = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)
        #V_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0

        U = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)
        W = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)
        V = (torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0).to(param.device)

        U[0,:,:] = 0.0
        U[-1,:,:] = 0.0
        U[0,:,:] = U[0,:,:] + 1.0*u_y[:,None]

        V[:,0,:] = 0.04   #giving entrianment velocity
        V[:,-1,:] = -0.04

        avg_in = (torch.sum(torch.mean((U[0,:,:]),dim = 1)*grid.Dy[0,:]) + 2*torch.sum((torch.mean(V[:,0,:],dim=1))*grid.Dx[:,0])).to(param.device)
        print("inlet_mass_flow",avg_in) # Should be 1

        # W[0,190:210,:] = (W_fluc_temp[0,190:210,:])
        
        # W[0,190:210,:] = (W_fluc_temp[0,190:210,:])
        
        U[-1,:,:] = U[-1,:,:] + avg_in/torch.sum(grid.Dy[-1,:])

        avg_out = (torch.sum(torch.mean((U[-1,:]),dim = 1)*grid.Dy[-1,:])).to(param.device)
        print("outlet_mass_flow",avg_out) #should be equal to inlet, conserving mass

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*U)
        Q['rhoV'].copy( V*rho )
        Q['rhoW'].copy( W*rho )     
      
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) ))
        del e
        
        return
        
# ----------------------------------------------------
# Initial conditions - field for turbulent temporal planar jet 3D
# ----------------------------------------------------
def get_IC_planar_jet_temporal_turb(grid, Q, param, EOS):
    #Does not work with MPI, build initial file in serial
    
        shear_layer_thickness = 0.1
        u_y =  0.5*(torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5)*2.0 + 0.5)/shear_layer_thickness) - torch.tanh(((grid.Y[0,:,0] - grid.Lx2*0.5)*2.0 - 0.5)/shear_layer_thickness))
        u_avg = torch.sum(u_y[475:525]*grid.Dy[0,475:525]) # u_avg is half

        data = h5py.File("PP_field_640.h5",'r+')  # Obtaining field from NGA spectral genrator
        
        U_fluc = data["rhoU"][()]
        V_fluc = data["rhoV"][()]
        W_fluc = data["rhoW"][()]

        U_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        W_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0
        V_fluc_temp = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0

        U_fluc_temp[:,394:406,:] = torch.from_numpy(U_fluc[:-1,394:406,:grid.Nx3]) # These points are H/3 from centre line, depends on grid size
        W_fluc_temp[:,394:406,:] = torch.from_numpy(W_fluc[:-1,394:406,:grid.Nx3])

        U = u_y[None,:,None] +  U_fluc_temp
        W = W_fluc_temp
        V = V_fluc_temp
        flux = torch.sum(U[0,:,0]*grid.Dy[0,:]) 
       
        print("flux at the entry is", flux)
        flux = torch.sum(U[-1,:,0]*grid.Dy[0,:])
        print("flux at the exit is", flux)


        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*U )
        Q['rhoV'].copy( 0.0*rho )
        Q['rhoW'].copy(W*rho )


        
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) ))
        del e
        
        return

# ----------------------------------------------------
# Initial conditions - field for laminar temporal planar jet 2D
# ----------------------------------------------------
def get_IC_planar_jet_temporal_lam(grid, Q, param, EOS):
    
    # Equations are taken from https://www.youtube.com/watch?v=hNc4PPWZtS8 and wiki
   
        # inlet
        J = 1000
        rho = 1000
        nu = 0.01
        N = int(grid.Nx2)
        y = torch.linspace(-int(grid.Lx2*0.5),int(grid.Lx2*0.5),N,dtype=param.WP).to(param.device)
       
        #outlet
       
        u_max = (((3*J*J)/(32*rho**2*nu))**(1/3))*(0.9823 + grid.X[:,0])**(-1/3)  #0.9823 is the point at which inlet vel width is 1

        xai = (grid.Y[0,:] - grid.Lx2*0.5)*((J/(48*rho*nu**2))**(1/3))*(0.9823 + grid.X[:,:])**(-2/3)
       
        y1 = (12 * ((6*rho*nu**2*(0.9823 + grid.X[:,0])**2)/J)**(1/3))

        u_out = torch.zeros(N).to(param.device)
        u_out = u_max[:,None]*(1/torch.cosh(xai))**2
        
        ## To have uniform flow in longitudnal direction
        u_out = torch.tile(torch.unsqueeze(u_out[0,:],1),(1,N)) 
        u_out = u_out.transpose(1,0)
        u_out = u_out[:-1,:]
        plt.contourf(u_out.cpu())
        plt.show()
        flux = torch.sum(u_out[0,:]*u_out[0,:]*grid.Dy[0,:]) #=1
       
        print("flux at the entry is", flux)
        flux = torch.sum(u_out[-1,:]*u_out[-1,:]*grid.Dy[0,:]) #=1
        print("flux at the exit is", flux)   # Momentum flux is constant
        print("velocity width for the first point is",y1[0])

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*u_out[:,:,None] )
        Q['rhoV'].copy( 0.0*rho )
        Q['rhoW'].copy( 0.0*rho )
        
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) ))
        del e
        
        return

# ----------------------------------------------------
# Initial conditions - field for laminar spatial planar jet 2D
# ----------------------------------------------------
def get_IC_planar_jet_spatial_lam(grid, Q, param, EOS):
    
    # Equations are taken from https://www.youtube.com/watch?v=hNc4PPWZtS8 and wiki
   
        # inlet
        J = 1000
        rho = 1000
        nu = 0.01
        N = int(grid.Nx2)
       
        #outlet
       
        u_max = (((3*J*J)/(32*rho**2*nu))**(1/3))*(0.9823 + grid.X[:,0])**(-1/3)  #0.9823 is the point at which inletvel width is 1

        xai = (grid.Y[0,:] - grid.Lx2*0.5)*((J/(48*rho*nu**2))**(1/3))*(0.9823 + grid.X[:,:])**(-2/3)
       
        y1 = (12 * ((6*rho*nu**2*(0.9823 + grid.X[:,0])**2)/J)**(1/3))

        u_out = torch.zeros(N).to(param.device)
        u_out = u_max[:,None]*(1/torch.cosh(xai))**2
        u_out1 = u_out*0.0
        u_out1[0,:] = u_out[0,:] 
        
        u_out1[-1,:] = (torch.sum(u_out1[0,:]*u_out1[0,:]*grid.Dy[0,:]) / torch.sum(grid.Dy[0,:]))**0.5
        u_out = u_out1 
        plt.plot(grid.Y[0,:],u_out[0,:])
        u_avg = torch.sum(u_out[0,:]*grid.Dy[0,:])
        
        V = torch.empty((grid.Nx1,grid.Nx2,grid.Nx3)) * 0.0

        plt.contourf(u_out.cpu())
        plt.show()
        flux = torch.sum(u_out[0,:]*u_out[0,:]*grid.Dy[0,:]) #=1
       
        print("flux at the entry is", flux)
        flux = torch.sum(u_out[-1,:]*u_out[-1,:]*grid.Dy[0,:]) #=1
        print("flux at the exit is", flux)
        print("velocity width for the first point is",y1[0])
        

        rho = torch.tensor((1.0,), dtype=param.WP).to(param.device)
        Q['rho'].copy( rho )
        Q['rhoU'].copy( rho*u_out[:,:,None] )
        Q['rhoV'].copy( V*rho )
        Q['rhoW'].copy( 0.0*rho )
      
        # Internal energy from pressure
        #   p_infty = 1/gamma
        p = 1.0 / param.gamma
        e = p / ((param.gamma-1.0) * rho * param.Ma**2)
        
        # Finalize total energy
        Q['rhoE'].copy( rho*( e + 0.5*((Q['rhoU'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoV'].interior()/Q["rho"].interior())**2 +
                                    (Q['rhoW'].interior()/Q["rho"].interior())**2 ) ))
        del e
        
        return


# ----------------------------------------------------
# Initial conditions - sine wave
# ----------------------------------------------------
def get_IC_sine(grid, Q, param, EOS):
        
    rho = torch.tensor((EOS.rho0,), dtype=param.WP).to(param.device)
    
    Q['rho'].copy( rho )
    Q['rhoU'].copy( rho * EOS.U0 * torch.sin(8 * np.pi * grid.xi_grid_mod[:,None,None] / 32.0) )
    Q['rhoV'].copy( 0.0*rho )
    Q['rhoW'].copy( 0.0*rho )

    # Internal energy from pressure
    p = EOS.p0
    e = EOS.get_internal_energy_RPY(rho, p)
    
    # Finalize total energy
    Q['rhoE'].copy( rho*( e + 0.5*(Q['rhoU'].interior()**2 +
                                   Q['rhoV'].interior()**2 +
                                   Q['rhoW'].interior()**2) ) )
    del e
    
    return


# ----------------------------------------------------
# Initial conditions - channel flow
# ----------------------------------------------------
def get_IC_channel(grid, Q, param, EOS):
    
    nx = len(grid.xi_grid_mod)
    ny = len(grid.eta_grid_mod)
    if (grid.ndim==3):
        nz = len(grid.z_grid_mod)
    else:
        nz = 1

    # Parabolic velocity profile (NGA/src/tools/init_flow/channel)
    if (nz>1):
        U_profile = EOS.U0 * (1.0 - (2.0 * grid.Y / grid.Lx2)**2)
    else:
        U_profile = EOS.U0 * (1.0 - (2.0 * grid.Y[:,:,None] / grid.Lx2)**2)
        
    rho = torch.tensor((EOS.rho0,), dtype=param.WP).to(param.device)
    Q['rho'].copy( rho )
    Q['rhoU'].copy( rho * 3.0/2.0 * U_profile )
    Q['rhoV'].copy( 0.0*rho )
    Q['rhoW'].copy( 0.0*rho )
    
    # Add random noise to U and V
    amp  = 0.25
    if (EOS.Re > 1000):
        rand = torch.from_numpy(np.random.rand(nx,ny,nz)).to(param.device) - 0.5
        Q['rhoU'].add( amp*rand )
        if (nz>1):
            rand = torch.from_numpy(np.random.rand(nx,ny,nz)).to(param.device) - 0.5
            Q['rhoW'].add( amp*rand )
        del rand

    # For faster transition
    if (nz>1):
        # Fluctuations in Z for U
        Q['rhoU'].add( amp * EOS.U0 *
                       torch.cos( 16.0*np.pi * grid.Z / grid.Lx3 ) )
        # Fluctuations in X for W
        Q['rhoW'].add( amp * EOS.U0 *
                       torch.cos( 16.0*np.pi * grid.X / grid.Lx1 ) )

    # Enforce walls
    if (not grid.periodic_eta):
        if (grid.BC_eta_bot=='wall' and param.jproc==0):
            Q['rhoU'].interior()[:,0,:]  = 0
            Q['rhoV'].interior()[:,0,:]  = 0
            Q['rhoW'].interior()[:,0,:]  = 0
        if (grid.BC_eta_top=='wall' and param.jproc==param.npy-1):
            Q['rhoU'].interior()[:,-1,:] = 0 
            Q['rhoV'].interior()[:,-1,:] = 0 
            Q['rhoW'].interior()[:,-1,:] = 0 
    
    # Internal energy from pressure
    p = EOS.p0
    e = EOS.get_internal_energy_RPY(rho, p)
    
    # Finalize total energy
    Q['rhoE'].copy( rho*( e + 0.5*(Q['rhoU'].interior()**2 +
                                   Q['rhoV'].interior()**2 +
                                   Q['rhoW'].interior()**2) ) )
    del e
    
    return


# ----------------------------------------------------
# Initial conditions - advecting vortex
# ----------------------------------------------------
def get_IC_vortex(grid, Q, param, EOS):
    imin_ = param.imin_; imax_ = param.imax_
    jmin_ = param.jmin_; jmax_ = param.jmax_
    kmin_ = param.kmin_; kmax_ = param.kmax_
    if (grid.ndim==2):
        X = grid.X[:,:,None]
        Y = grid.Y[:,:,None]
    else:
        X = grid.X
        Y = grid.Y
        
    rho = torch.tensor((EOS.rho0,), dtype=param.WP).to(param.device)
    Q['rho'].copy( rho )
    Q['rhoW'].copy( 0.0*rho )
    
    # Velocity
    Rc2= 0.5**2
    C  = 0.02*EOS.U0*np.sqrt(Rc2)
    xc = 0.5*grid.Lx1
    yc = 0.5*grid.Lx2
    r2  = ((X-xc)**2 + (Y-yc)**2)/Rc2
    
    Q['rhoU'].var[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = EOS.U0 - C*(Y-yc)/Rc2*torch.exp(-r2/2.0)
    Q['rhoV'].var[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = C*(X-xc)/Rc2*torch.exp(-r2/2.0)

    # Save pressure in internal energy
    p_back = EOS.p0
    Q['rhoE'].var[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = p_back - rho*C**2/(2.0*Rc2)*torch.exp(-r2)

    # Internal energy from pressure
    e = EOS.get_internal_energy_RPY(rho, Q['rhoE'].var)
    
    # Finalize total energy
    Q['rhoE'].var = e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)
    
    # Finalize conserved quantities and update borders
    for name in Q.names[1:]:
        Q[name].mul( Q['rho'].interior() )

    return


# ----------------------------------------------------
# Initial conditions - shear layer
# ----------------------------------------------------
def get_IC_shear_layer(grid, Q, param, EOS):

    # Grid
    thk = 0.5 * EOS.L0
    nx = len(grid.xi_grid_mod)
    ny = len(grid.eta_grid_mod)
    if (grid.ndim==3):
        Y  = grid.Y - 0.5*grid.Lx2
        nz = len(grid.z_grid_mod)
    else:
        Y  = grid.Y[:,:,None] - 0.5*grid.Lx2
        nz = 1
        
    # Nonreacting or reacting
    if EOS.combustion:
        # Set up species
        Q['rhoY_H2'].copy( EOS.Y0[EOS.TC.species_names.index('H2')] *
                           (1.0 - 0.5*(torch.tanh(Y/thk) + 1.0)) )
        Q['rhoY_O2'].copy( EOS.Y0[EOS.TC.species_names.index('O2')] *
                           (1.0 - 0.5*(torch.tanh(-Y/thk) + 1.0)) )
        SC = []
        for name in EOS.sc_names:
            SC.append(Q[name].interior())

        # Initial temperature
        T = EOS.T0 + 700.0 * torch.exp(-Y**2 / (2.0*thk**2))
        
        # Initial density
        rho = EOS.get_density(EOS.p0, T, SC)

    else:
        # density
        rho = torch.ones_like(Y) * EOS.rho0
        
    Q['rho'].copy( rho )
    
    # velocity
    Q['rhoU'].copy( rho * EOS.U0 * (1.0 - (torch.tanh(Y/thk) + 1.0)) )

    # For faster transition
    accelerate = False
    if hasattr(EOS, 'Re'):
        if (EOS.Re > 150):
            accelerate = True
    else:
        accelerate = True

    if accelerate:
        #rand = torch.from_numpy(np.random.rand(nx,ny,nz)).to(param.device) - 0.5
        rand = torch.from_numpy(np.random.default_rng().random((nx,ny,nz), dtype=param.WP_np)).to(param.device) - 0.5
        amp  = 0.2 * EOS.U0
        Q['rhoU'].add(  rho*amp*rand )
        Q['rhoV'].copy( rho*amp*rand )
        Q['rhoW'].copy( rho*amp*rand )

    # For absorbing boundary conditions
    # Lower wall
    if (param.jproc==0):
        Q['rhoU'].var[:,:param.jmin_+1,:] = rho[:1,:1,:1] * EOS.U0
        Q['rhoV'].var[:,:param.jmin_+1,:] = 0.0
        Q['rhoW'].var[:,:param.jmin_+1,:] = 0.0
    # Upper wall
    if (param.jproc==param.npy-1):
        Q['rhoU'].var[:,param.jmax_-1:,:] = -rho[:1,-1:,-1:] * EOS.U0
        Q['rhoV'].var[:,param.jmax_-1:,:] =  0.0
        Q['rhoW'].var[:,param.jmax_-1:,:] =  0.0

    # Nonreacting or reacting
    if EOS.combustion:
        # Internal energy from T and Y
        e = EOS.get_internal_energy_TY( T, SC )

        # Finalize species
        for name in EOS.sc_names:
            Q[name].mul( rho )

    else:
        # Internal energy from pressure
        p = EOS.p0
        e = EOS.get_internal_energy_RPY(Q['rho'].interior(), p)
    
    # Finalize total energy
    Q['rhoE'].copy( Q['rho'].interior() * e +
                    0.5*(Q['rhoU'].interior()**2 +
                         Q['rhoV'].interior()**2 +
                         Q['rhoW'].interior()**2)/Q['rho'].interior() )

    # Scalars
    if ('rhoZmix' in EOS.sc_names):
        Q['rhoZmix'].copy( Q['rho'].interior() * (1.0 - 0.5*(torch.tanh(Y/thk) + 1.0)) )
    
    return


# ------------------------------------------------------
# Initial conditions - Sod shock problem - dimensionless
# ------------------------------------------------------
def get_IC_Sod(grid, Q, param, EOS):
    # Dimensionless only
    if EOS.dimensional:
        raise Exception('Initial_Conditions.py: Sod problem requires a dimensionless EOS')
    
    # Density
    one = torch.tensor((1.0,), dtype=param.WP).to(param.device)
    Q['rho'].copy( one )
    
    # Velocity
    Q['rhoU'].copy( 0.0*one )
    Q['rhoV'].copy( 0.0*one )
    Q['rhoW'].copy( 0.0*one )
    
    # Pressure
    #p = one
    p = one * EOS.p0
    Q['rhoE'].copy( p )

    fac = 1.0
    thk = 0.0005*grid.Lx1
    if grid.Nx1 > grid.Nx2:
        Q['rho'].copy(  fac*(1.0 - 0.875*0.5*(torch.tanh((grid.X[:,:,None]-0.5)/thk) + 1.0)) )
        Q['rhoE'].copy( fac*(1.0 - 0.9*  0.5*(torch.tanh((grid.X[:,:,None]-0.5)/thk) + 1.0)) )
    elif grid.Nx2 > grid.Nx1:
        Q['rho'].copy(  fac*(1.0 - 0.875*0.5*(torch.tanh((grid.Y[:,:,None]-0.5)/thk) + 1.0)) )
        Q['rhoE'].copy( fac*(1.0 - 0.9*  0.5*(torch.tanh((grid.Y[:,:,None]-0.5)/thk) + 1.0)) )

    # Piston velocity - not part of original Sod problem
    #vel = 1.0
    #Q['rhoU'].copy(  vel*(1.0 - 0.5*(torch.tanh((grid.Xi[:,:,None]-0.5)/thk) + 1.0)) )

    # Internal energy from pressure
    e = EOS.get_internal_energy_RPY(Q['rho'].var, Q['rhoE'].var)
    
    # Finalize total energy
    Q['rhoE'].var = e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)
    
    # Finalize conserved quantities and update borders
    for name in Q.names[1:]:
        Q[name].mul( Q['rho'].interior() )

    return


# ----------------------------------------------------
# Initial conditions - Oblique shock
#    Must provide grid.delta (deflection angle)
# ----------------------------------------------------
def get_IC_oblique(grid, Q, param, EOS):

    p1   = EOS.p0
    rho1 = EOS.rho0

    # Oblique shock angle (weak-shock solution)
    theta,M2,p_ratio,rho_ratio = shock.oblique_theta(EOS.Ma, grid.delta, EOS.gamma)

    if (theta is not None):
        # Downstream quantities
        p2   = p1*p_ratio
        rho2 = rho1*rho_ratio
        u1t  = EOS.U0*np.sin(np.radians(theta))
        u1n  = EOS.U0*np.cos(np.radians(theta))
        u2n  = u1n/rho_ratio
        U2   = np.sqrt(u2n**2 + u1t**2)
        #print(p2,rho2,M2,U2)

        if (grid.ndim==2):
            x = grid.X[:,:,None]
            y = grid.Y[:,:,None]
        else:
            x = grid.X
            y = grid.Y
                
        # Shock location
        yshock = (x+1.0)*np.tan(np.radians(theta))
        # Blending function
        w2 = 0.5*(torch.tanh(yshock-y) + 1.0)
        w1 = 1.0 - w2

        # Initial conditions
        Q['rho' ].copy( w1*rho1 + w2*rho2 )
        Q['rhoU'].copy( w1*EOS.U0  + w2*U2 * np.cos(np.radians(grid.delta)) )
        Q['rhoV'].copy( w1*0.0     + w2*U2 * np.sin(np.radians(grid.delta)) )
        Q['rhoE'].copy( w1*p1 + w2*p2 )
        
    else:
        # No attached shock solution
        Q['rho' ].copy( rho1 )
        Q['rhoU'].copy( EOS.U0 )
        Q['rhoE'].copy( p1 )

    # Internal energy
    e = EOS.get_internal_energy_RPY(Q['rho'].var, Q['rhoE'].var)
    
    # Finalize total energy
    Q['rhoE'].var = e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)
    
    # Finalize conserved quantities and update borders
    for name in Q.names[1:]:
        Q[name].mul( Q['rho'].interior() )

    return


# ----------------------------------------------------
# Initial conditions - flow over cylinder
# ----------------------------------------------------
def get_IC_cylinder(grid, Q, param, EOS):
    if (grid.ndim==2):
        X = grid.X[:,:,None]
        Y = grid.Y[:,:,None]
    else:
        X = grid.X
        Y = grid.Y

    zero = torch.tensor((0.0,), dtype=param.WP).to(param.device)
    one  = torch.tensor((1.0,), dtype=param.WP).to(param.device)
    Q['rho'].copy( one * EOS.rho0 )
    Q['rhoU'].copy( zero )
    Q['rhoV'].copy( zero )
    Q['rhoW'].copy( zero )
    
    # u-velocity
    R   = (torch.sqrt(X**2 + Y**2) - grid.R_min)/EOS.L0
    # x <= 0 : U=0 already
    #
    # 0 < x < 1 : exp profile
    mask = R.le(0.0) + R.ge(1.0)
    vel  = torch.tensor(EOS.U0 / (1 + torch.exp(1.0/R - 1.0/(1.0-R))),
                        dtype=param.WP).to(param.device)
    vel[mask] = zero
    Q['rhoU'].add( vel )
    #
    # x >= 1 : U0
    mask = R.lt(1.0)
    vel  = zero*R + one*EOS.U0
    vel[mask]  = zero
    Q['rhoU'].add( vel )

    # Internal energy from pressure
    p = EOS.p0
    e = EOS.get_internal_energy_RPY(Q['rho'].var, p)
    
    # Finalize total energy
    Q['rhoE'].var = e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)
    
    # Finalize conserved quantities and update borders
    for name in Q.names[1:]:
        Q[name].mul( Q['rho'].interior() )

    return

# ----------------------------------------------------
# Initial conditions - flow over a flat plate 
# ----------------------------------------------------
def get_IC_flat_plate(grid,Q,param,EOS):
    if (grid.ndim==2):
        X = grid.X[:,:,None]
        Y = grid.Y[:,:,None]
    else:
        X = grid.X
        Y = grid.Y

    # zero = torch.tensor((0.0,), dtype=torch.float64).to(param.device)
    # one  = torch.tensor((1.0,), dtype=torch.float64).to(param.device)
    # Q['rho'].copy( one )
    # Q['rhoU'].copy( zero )
    # Q['rhoV'].copy( zero )
    # Q['rhoW'].copy( zero )
    
    # # u-velocity
    # # x <= 0 : U=0 already
    # #
    # # 0 < x < 1 : exp profile
    # # mask = Y.le(0.0) + Y.ge(0.1)
    # # vel  = param.U0 / (1 + torch.exp(1.0/X - 1.0/(1.0-X)))
    # # vel[mask] = zero
    # # Q['rhoU'].add( vel )
    # #
    # # x >= 1 : U0
    # mask = Y.lt(0.0001)
    # vel  = zero*X + one*param.U0
    # vel[mask]  = zero
    # Q['rhoU'].add( vel )

    # # Internal energy from pressure
    # #   p_infty = 1/gamma
    # p = 1.0 / param.gamma
    # e = p / ((param.gamma-1.0) * one * param.Ma**2)
    
    # # Finalize total energy
    # Q['rhoE'].var = e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)
    
    # # Finalize conserved quantities and update borders
    # for name in Q.names[1:]:
    #     Q[name].mul( Q['rho'].interior() )
    
    
    # # Loading DSMC wall-profiles
    # T_prof   = torch.DoubleTensor(np.load('M7Tests/wallProfs/TWallProf_M7_highDens_256.npy')).to(param.device)
    # U_prof   = torch.DoubleTensor(np.load('M7Tests/wallProfs/uWallProf_M7_highDens_256.npy')).to(param.device)
    # V_prof   = torch.DoubleTensor(np.load('M7Tests/wallProfs/vWallProf_M7_highDens_256.npy')).to(param.device)
    # rho_prof = torch.DoubleTensor(np.load('M7Tests/wallProfs/rhoWallProf_M7_highDens_256.npy')).to(param.device)

    # T_prof   = torch.DoubleTensor(np.load('dsmcDat/highDensity/wallProfs/TWallProf_M7_highDens_256.npy')).to(param.device)
    # U_prof   = torch.DoubleTensor(np.load('dsmcDat/highDensity/wallProfs/uWallProf_M7_highDens_256.npy')).to(param.device)
    # V_prof   = torch.DoubleTensor(np.load('dsmcDat/highDensity/wallProfs/vWallProf_M7_highDens_256.npy')).to(param.device)
    # rho_prof = torch.DoubleTensor(np.load('dsmcDat/highDensity/wallProfs/rhoWallProf_M7_highDens_256.npy')).to(param.device)

    T_prof   = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/TWallProf_M7_lowDens_256.npy')).to(param.device)
    U_prof   = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/uWallProf_M7_lowDens_256.npy')).to(param.device)
    V_prof   = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/vWallProf_M7_lowDens_256.npy')).to(param.device)
    rho_prof = torch.DoubleTensor(np.load('../../flat_plate_2D/M7Tests_lowDense/wallProfs/rhoWallProf_M7_lowDens_256.npy')).to(param.device)

    # Adding extra dimension
    T_prof = T_prof[:, None]
    U_prof = U_prof[:, None]
    V_prof = V_prof[:, None]
    rho_prof = rho_prof[:, None]
    
    
    # Setting uniform flow
    if not EOS.dimensional:
        rho = torch.tensor((1.0,), dtype=torch.float64).to(param.device)
        p = EOS.p0
        e = EOS.get_internal_energy_RPY(Q['rho'].var, p)
        
    else:
        rho = EOS.rho0 * torch.tensor((1.0,), dtype=torch.float64).to(param.device)        
        T = torch.ones_like(Q['rho'].var) * EOS.T0
        
        if not param.adiabatic:    
            # Setting dirichlet values from DSMC
            print('Setting dirich temp')
            T[param.imin_:param.imax_, param.jmin_:param.jmax_, :][grid.wallPoint:, 0, :] = T_prof#300.#292.#622.682#EOS.T0#
        
        e = EOS.get_internal_energy_TY(T)


    Q['rho'].copy( rho )
    Q['rhoU'].copy( rho*EOS.U0 )
    Q['rhoV'].copy( 0.0 *rho )
    Q['rhoW'].copy( 0.0*rho )


    # Setting inital conditions for Burger's equations
    # arr1 = X >= 0.1
    # arr2 = X <= 0.2
    # arr3 = Y >= 0.2
    # arr4 = Y <= 0.3
    # arr5 = arr1 & arr2 & arr3 & arr4

    # Q['rhoU'].interior()[arr5] = 1.0
    #print(grid.lol)

    if not param.adiabatic:
        # Setting dirichlet values from DSMC
        print('Setting dirich vals')
        Q['rho'].interior()[grid.wallPoint:, 0, :]  = rho_prof#EOS.rho0#1.8423e-05#EOS.rho0
        Q['rhoU'].interior()[grid.wallPoint:, 0, :] = Q['rho'].interior()[grid.wallPoint:, 0, :] * U_prof#0.0#EOS.rho0 * 0.0#EOS.U0 #EOS.rho0 * 0.0#
        #Q['rhoU'].interior()[0, :, :] = 2.0
        Q['rhoV'].interior()[grid.wallPoint:, 0, :] = Q['rho'].interior()[grid.wallPoint:, 0, :] * V_prof
    


    # Internal energy from pressure
    #   p_infty = 1/gamma

    
    # # Finalize total energy
    # Q['rhoE'].copy( rho*( e + 0.5*(Q['rhoU'].interior()**2 +
    #                                Q['rhoV'].interior()**2 +
    #                                Q['rhoW'].interior()**2) ) )
    
    # # Enforce Dirichlet no-slip walls
    # #   Could use law of the wall...
    # Q['rhoE'].sub( 0.5*( Q['rhoU'].interior()**2 +
    #                       Q['rhoV'].interior()**2 +
    #                       Q['rhoW'].interior()**2 )/Q['rho'].interior() )
    
    #mask = torch.ones_like(grid.Eta[:,:,None])
    #indx = Y.eq(0.0) #+ X.ge(4.5)
    #mask[indx] = 0.0
    
    #Q['rhoU'].mul_( mask )  # rho*U

    
    # thk = 0.1 
    # Q['rhoU'].mul(torch.tanh(grid.Eta[:,:,None]/thk) )  # rho*U
    # Q['rhoV'].mul( torch.tanh(grid.Eta[:,:,None]/thk) )  # rho*v
    # Q['rhoW'].mul( torch.tanh(grid.Eta[:,:,None]/thk) )  # rho*W
    
    
    
    # # Reinitialising inlet
    # mask = torch.ones_like(grid.Eta[:,:,None])
    # indx = X.lt(1.0) #+ X.ge(4.5)
    

    # # Manually setting left 50 cells (need to change to more general implementation)
    # mask[0:grid.wallPoint,:,:] = 0.0
    
    # Q['rhoU'].mul_( mask )  # rho*U
    # Q['rhoV'].mul_( mask )  # rho*v
    # Q['rhoW'].mul_( mask )  # rho*W
    
    # mask = torch.zeros_like(grid.Eta[:,:,None])
    # indx = X.lt(1.0) #+ X.ge(4.5)
    # mask[0:grid.wallPoint,:,:] = rho*EOS.U0
    
    # Q['rhoU'].add( mask )  # rho*U
        

    # # Adjust rhoE
    # Q['rhoE'].add_( 0.5*( Q['rhoU'].interior()**2 +
    #                       Q['rhoV'].interior()**2 +
    #                       Q['rhoW'].interior()**2 )/Q['rho'].interior() )
    # Q['rhoE'].update_border()

    # Finalize total energy
    Q['rhoE'].var = (Q['rho'].var * e + 0.5*(Q['rhoU'].var**2 + Q['rhoV'].var**2 + Q['rhoW'].var**2)/Q['rho'].var) 

    
    return


# ----------------------------------------------------
# Initial conditions - adjoint field
# ----------------------------------------------------
def get_IC_Adjoint(grid, Q, Q_A, param):

    zero = torch.tensor((0.0,), dtype=param.WP).to(param.device)
    
    Q_A['rho_A' ].copy( zero )
    Q_A['rhoU_A'].copy( zero )
    Q_A['rhoV_A'].copy( zero )
    Q_A['rhoW_A'].copy( zero )
    Q_A['rhoE_A'].copy( zero )

    return


# ----------------------------------------------------
# Initial conditions - isotropic turbulence
# ----------------------------------------------------
def get_IC_isotropic(grid, Q, param, EOS):
    # density
    rho = 1.0
    Q[0,:,:] = rho
    
    # velocity
    nk = len(grid.xi_grid_mod)//2 + 1
    nx = len(grid.xi_grid_mod)
    ny = len(grid.eta_grid_mod)
    
    le = 0.5 # energetic length scale
    Ut = 1000.0 # Fluctuations
    
    ke = 2.0*np.pi/le
    dk = 2.0*np.pi/grid.Lx1
    kc = ny//2 * dk
    eps = ke/1e6
    amp_disc = dk
    spec_amp = 16*np.sqrt(2.0/np.pi)*Ut**2/ke

    # Compute the spectrum
    ak = torch.zeros((nk,ny),dtype=torch.complex128).to(param.device)
    bk = torch.zeros((nk,ny),dtype=torch.complex128).to(param.device)
    Uk = torch.zeros((nx,ny),dtype=torch.complex128).to(param.device)
    Vk = torch.zeros((nx,ny),dtype=torch.complex128).to(param.device)
    for i in range(nk):
        for j in range(ny):
            # Random numbers
            rand = np.random.rand(1)[0]
            psr  = 2.0*np.pi*(rand-0.5)
            rand = np.random.rand(1)[0]
            ps1  = 2.0*np.pi*(rand-0.5)
            rand = np.random.rand(1)[0]
            ps2  = 2.0*np.pi*(rand-0.5)

            # Wavenumbers
            kx = i*dk
            ky = j*dk
            if (j>nk): ky = -(ny-j)*dk
            kk = np.sqrt(kx**2 + ky**2)

            # Spectra
            spect = spec_amp * (kk/ke)**4 * np.exp(-2.0*(kk/ke)**2)

            # Coefficients
            if ((kk>eps) and (kk<=kc)):
                ak[i,j] = amp_disc*np.sqrt(spect/(2.0*np.pi*kk**2))*np.exp(1j*ps1)*np.cos(psr)
                bk[i,j] = amp_disc*np.sqrt(spect/(2.0*np.pi*kk**2))*np.exp(1j*ps2)*np.sin(psr)

                # Compute the Fourier coefficients of velocity
                if (kk<eps):
                    Uk[i,j] = (bk[i,j] - ak[i,j])/np.sqrt(2.0)
                else:
                    Uk[i,j] = (bk[i,j]*ky - ak[i,j]*kx)/kk
                
                Vk[i,j] = -bk[i,j]

    # Cleanup
    del ak,bk
    
    # Zero the oddball

    # Inverse transform the coefficients
    U = torch.fft.ifft2(Uk)
    V = torch.fft.ifft2(Vk)
    #U = torch.fft.ifft2(torch.cat((Uk, Uk[:,:]),dim=0))
    #V = torch.fft.ifft2(torch.cat((Vk, Uk[:,:]),dim=0))

    # rho*U
    Q[1,:,:] = rho*torch.real(U)
    Q[2,:,:] = rho*torch.real(V)
    
    # Internal energy from pressure
    e = 1.0 / ((param.gamma-1.0)*Q[0,:,:]*param.Ma**2)
    # Finalize total energy
    Q[3,:,:] = Q[0,:,:]*e + 0.5*(Q[1,:,:]**2 + Q[2,:,:]**2)/Q[0,:,:]

    return


# ----------------------------------------------------
# Initial conditions - supersonic nozzle flow
# ----------------------------------------------------
def get_IC_supersonic_nozzle(grid, Q, param, EOS):
    # sets conditions to vary only in x
    # computes conditions using the area-mach relation
    # assumptions:
    #   - isentropic
    #   - nozzle throat is at x=0
    # THIS FUNCTION IS ONLY SETUP TO WORK IN SERIAL.

    # find the mach number using the area mach relation
    def area_mach_relation(M, area_ratio):
        g_term1 = (EOS.gamma+1)
        g_term2 = (EOS.gamma-1)
        g_term3 = g_term1/2/g_term2
        return ((g_term1/2)**(-g_term3)*(1+g_term2/2*M**2)**(g_term3)/M
                - area_ratio
                )
    
    if grid.ndim == 3:
        gridY = grid.Y[:, :, 0]
        gridX = grid.X[:, :, 0]
    else:
        gridY = grid.Y
        gridX = grid.X

    areas = gridY[:, -1] - gridY[:, 0]
    area_ratios = areas/torch.min(areas)
    
    x_centerline = gridX[:, grid.Nx2//2]
    subsonic = x_centerline < 0

    res_sub = root(lambda M: area_mach_relation(M, area_ratios.cpu().numpy()),
               np.ones_like(areas.cpu())*0.1)
    res_sup = root(lambda M: area_mach_relation(M, area_ratios.cpu().numpy()),
               np.ones_like(areas.cpu())*2)
    
    M = torch.empty((grid.Nx1,), dtype=param.WP, device=param.device)
    M[subsonic] = torch.tensor(res_sub.x[subsonic.cpu()], 
                               dtype=torch.float64, 
                               device=param.device)
    M[~subsonic] = torch.tensor(res_sup.x[~subsonic.cpu()], 
                               dtype=torch.float64, 
                               device=param.device)
    M_initial = M[0]
    
    gamma_m1 = EOS.gamma - 1

    # stagnation properties (assumes isentropic)
    T0 = EOS.T0*(1 + 0.5*gamma_m1*M_initial**2)
    p0 = EOS.p0*(1 + 0.5*gamma_m1*M_initial**2)**(EOS.gamma/gamma_m1)

    isentropic_term = (1 + 0.5*gamma_m1*M**2)

    temperature = T0/isentropic_term
    pressure = p0/isentropic_term**(EOS.gamma/(EOS.gamma-1))
    speed_of_sound = torch.sqrt(EOS.gamma*EOS.Rgas*temperature)
    velocity = M*speed_of_sound
    rho = EOS.get_density(pressure, temperature)
    
    rho = rho[:, None, None]*torch.ones((param.nx_, param.ny_, param.nz_),
                                        dtype=param.WP).to(param.device)
    Q['rho'].copy(rho)
    Q['rhoU'].copy(rho*velocity[:, None, None])
    Q['rhoV'].copy(0.0*rho)
    Q['rhoW'].copy(0.0*rho)

    # Internal energy from pressure
    e = EOS.get_internal_energy_RPY(rho, pressure[:, None, None])

    # Finalize total energy
    Q['rhoE'].copy(rho*e + 0.5*(Q['rhoU'].interior()**2 +
                                Q['rhoV'].interior()**2 +
                                Q['rhoW'].interior()**2)/rho)
    del e

    
# Initial conditions - airfoil
# ----------------------------------------------------
def get_IC_airfoil_dlsgs(grid, Q, param, EOS):
    rho = torch.tensor((EOS.rho0,), dtype=torch.float64).to(param.device)
    AoA = torch.tensor(grid.AoA) if hasattr(grid, 'AoA') else torch.tensor(0.0)
    x_comp = torch.cos(AoA * torch.pi / 180.0)
    y_comp = torch.sin(AoA * torch.pi / 180.0)
    Q['rho'].copy(rho)
    Q['rhoU'].copy(rho * x_comp * EOS.U0)
    Q['rhoV'].copy(rho * y_comp*EOS.U0)
    Q['rhoW'].copy(0.0 * rho)

    # Internal energy from pressure
    p = EOS.p0
    e = EOS.get_internal_energy_RPY(rho, p)

    # Finalize total energy
    Q['rhoE'].copy(rho * (e + 0.5 * (Q['rhoU'].interior() ** 2 +
                                     Q['rhoV'].interior() ** 2 +
                                     Q['rhoW'].interior() ** 2)))
    del e

    return
