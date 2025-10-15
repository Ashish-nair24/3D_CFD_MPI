"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file RHS.py

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
import copy

from . import Operator as op
        
from . import Data
from .Library import Parallel

#import cProfile

class RHS:
    def __init__(self, grid, metrics, param, EOS, tmp_grad=0, jvp=False):
        
        self.grid = grid
        self.metrics = metrics
        self.param = param
        self.EOS = EOS
        self.tmp_grad = tmp_grad
        self.jvp = jvp
        self.constList = []
        self.modelOutputs = None
        self.modMu = None
        self.modKappa = None
        
        return
    
    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 1D-y
    #   For RANS of channels, temporal jets, etc.
    #   Collocated visc terms - conservative form
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    def NS_1D(self, q):
        
        # Extract conserved variables
        rho  = q['rho']
        rhoU = q['rhoU']
        rhoE = q['rhoE']

        # Compute primitives including overlaps
        u = rhoU.var/rho.var
        T, p, e = self.EOS.get_TPE(q)

        # Compute 1st derivatives - true interior
        drhoE_dy = self.metrics.grad_node( rhoE.var )[1]
        dp_dy    = self.metrics.grad_node( p )[1]
        
        # Compute 1st derivatives - extended interior for 2nd derivatives
        drho_dy  = self.metrics.grad_node( rho.var,  compute_extended=True )[1]
        drhoU_dy = self.metrics.grad_node( rhoU.var, compute_extended=True )[1]
        dT_dy    = self.metrics.grad_node( T, compute_extended=True, Neumann=False )[1]

        # Velocity gradients - extended interior
        # du
        du_dy = (drhoU_dy - self.metrics.full2ext( u ) * drho_dy) / self.metrics.full2ext( rho.var )

        # Variable transport properties
        mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
        
        mu_eff    = mu
        kappa_eff = kappa
        beta_art  = 0.0
        
        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                                 self.metrics.ext2int( mu_eff ),
                                                 self.metrics.ext2int( du_dy ),
                                                 None)
            
        # Viscous stress
        sigma_12 = mu_eff*( du_dy )
        sigma_12_dy = self.metrics.grad_node(sigma_12, extended_input=True)[1]

        # Heat flux
        q_2 = -kappa_eff * dT_dy
        q_2_dy = self.metrics.grad_node(q_2, extended_input=True)[1]

        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        p = self.metrics.full2int( p )
        drho_dy  = self.metrics.ext2int( drho_dy )
        drhoU_dy = self.metrics.ext2int( drhoU_dy )
        du_dy    = self.metrics.ext2int( du_dy )
        sigma_12 = self.metrics.ext2int( sigma_12 )

        
        # Compute RHS terms on true interior
        # Momentum equation - x
        conv  = 0.0
        pres  = 0.0
        visc  = sigma_12_dy
        qdot1 = pres - conv + visc + srcU

        # Continuity; y- and z-momentum
        qdot0 = 0.0*qdot1
        qdot2 = 0.0*qdot1
        qdot3 = 0.0*qdot1
        
        # Total energy equation
        conv  = 0.0
        pres  = 0.0
        visc  = u*sigma_12_dy + sigma_12*du_dy
        diff  = q_2_dy
        qdot4 = visc - conv - pres - diff + u*srcU

        
        # Closure model
        if (self.param.Use_Model):
            # Dictionary of all possible model inputs
            input_dict = {'u':u, 'du_dy':du_dy}

            qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
                        'qdot3':qdot3, 'qdot4':qdot4}

            model_outputs = self.param.apply_model(self.param.model, input_dict, qdot_dict)
        else:
            model_outputs = None
    

        # Boundary conditions
        # Dirichlet BCs on -/+ eta
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            qdot1[:,0,:] = 0.0
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            qdot4[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho.interior() )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU.interior() )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE.interior() )

        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho.interior() )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU.interior() )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE.interior() )

            
        return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4),dim=0), model_outputs


    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 2D
    #   Collocated visc terms - conservative form
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    def NS_2D_RANS(self, q):

        # k-epsilon coefficients
        C_mu      = 0.09
        Pr_t      = 0.9
        sigma_k   = 1.00
        sigma_eps = 1.30
        C_1eps    = 1.44
        C_2eps    = 1.92

        # Extract conserved variables
        rho  = q['rho']
        rhoU = q['rhoU']
        rhoV = q['rhoV']
        rhoE = q['rhoE']
        rhok = q['rhok']
        rhoeps = q['rhoeps']

        # Compute primitives including overlaps
        u   = rhoU.var/rho.var
        v   = rhoV.var/rho.var
        k   = rhok.var/rho.var
        e   = rhoE.var/rho.var - 0.5*(u**2 + v**2) - k
        eps = rhoeps.var/rho.var
        u = rhoU.var/rho.var
        p   = (self.param.gamma-1.0) * rho.var * e * self.param.Ma**2
        T   = self.param.gamma * e * self.param.Ma**2 
        #T, p, e = self.EOS.get_TPE(q)

        # Compute 1st derivatives - true interior
        drhoE_dx,drhoE_dy = self.metrics.grad_node( rhoE.var )[:2]
        dp_dx,dp_dy       = self.metrics.grad_node( p )[:2]
        drhok_dx,drhok_dy = self.metrics.grad_node( rhok.var )[:2]
        drhoeps_dx,drhoeps_dy = self.metrics.grad_node( rhoeps.var )[:2]
        
        # Compute 1st derivatives - extended interior for 2nd derivatives
        drho_dx,drho_dy   = self.metrics.grad_node( rho.var,  compute_extended=True )[:2]
        drhoU_dx,drhoU_dy = self.metrics.grad_node( rhoU.var, compute_extended=True )[:2] 
        drhoV_dx,drhoV_dy = self.metrics.grad_node( rhoV.var, compute_extended=True )[:2]
        dT_dx,dT_dy       = self.metrics.grad_node( T, compute_extended=True, Neumann=False )[:2]
        dk_dx,dk_dy       = self.metrics.grad_node( k, compute_extended=True, Neumann=False )[:2]
        deps_dx,deps_dy   = self.metrics.grad_node( eps, compute_extended=True, Neumann=False )[:2]

        # Velocity gradients - extended interior
        # du
        du_dx = (drhoU_dx - self.metrics.full2ext( u ) * drho_dx) / self.metrics.full2ext( rho.var )
        du_dy = (drhoU_dy - self.metrics.full2ext( u ) * drho_dy) / self.metrics.full2ext( rho.var )
        # dv
        dv_dx = (drhoV_dx - self.metrics.full2ext( v ) * drho_dx) / self.metrics.full2ext( rho.var )
        dv_dy = (drhoV_dy - self.metrics.full2ext( v ) * drho_dy) / self.metrics.full2ext( rho.var )
        
        # Velocity divergence
        div_vel = du_dx + dv_dy

        error = 1E-12
        # Variable transport properties
        mu = ((self.param.gamma-1.0) * self.metrics.full2ext(T))**0.7
        mu_t = self.param.Re * C_mu * (( self.metrics.full2ext(k)**2)/(self.metrics.full2ext(eps)+error))

        kappa = (mu/self.param.Pr) + (mu_t/Pr_t)

        # Artificial diffusivity (Kawai & Lele JCP 2008)
        if self.param.artDiss:
            # Exclude from adjoint calculation
            with torch.inference_mode():
                # Strain-rate magnitude
                S = op.strainrate_mag_2D(du_dx,du_dy,dv_dx,dv_dy)

                # Evaulate the artificial transport coefficients
                curl_u = dv_dx - du_dy
                mu_art,beta_art,kappa_art = op.art_diff4_D_2D(rho,drho_dx,drho_dy,S,self.tmp_grad,
                                                            div_vel,curl_u,T,e,
                                                            self.param.cs,self.grid,self.metrics,self.param.device)

                #mu_eff    = self.param.mu + mu_art *self.param.Re
                mu_eff    = mu + mu_art *self.param.Re
                kappa_eff = kappa + kappa_art *self.param.Re*self.param.Pr
                beta_art  = beta_art *self.param.Re

                #DN  = torch.amax(mu_eff)/self.param.Re * self.param.dt / self.param.dx_min**2
                #print(torch.amax(mu_art),
                #      torch.amax(kappa_art),
                #      torch.amax(beta_art),
                #      DN)
        else:
            mu_eff    = mu + mu_t #self.param.mu
            kappa_eff = kappa
            beta_art  = 0.0

        
        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                            self.metrics.ext2int( mu_eff ),
                                            self.metrics.ext2int( du_dy ),
                                            None)
            
        # Viscous stress
        #   Divergence terms are computed on true interior
        div_term = (beta_art - 2.0*mu_eff/3.0)*div_vel

        sigma_11 = 2.0*mu_eff*du_dx + div_term
        sigma_11_dx,sigma_11_dy = self.metrics.grad_node(sigma_11, extended_input=True, compute_dy=False)[:2]
        
        sigma_22 = 2.0*mu_eff*dv_dy + div_term
        sigma_22_dx,sigma_22_dy = self.metrics.grad_node(sigma_22, extended_input=True, compute_dx=False)[:2]
        
        sigma_12 = mu_eff*( du_dy + dv_dx )
        sigma_12_dx,sigma_12_dy = self.metrics.grad_node(sigma_12, extended_input=True)[:2]

        # Heat flux
        q_1 = -kappa_eff * dT_dx
        q_1_dx,q_1_dy = self.metrics.grad_node(q_1, extended_input=True, compute_dy=False)[:2]
        
        q_2 = -kappa_eff * dT_dy
        q_2_dx,q_2_dy = self.metrics.grad_node(q_2, extended_input=True, compute_dx=False)[:2]

        #model of k in energy eqn
        Emk_1 = (mu + mu_t/sigma_k) * dk_dx
        Emk_1_dx, Emk_1_dy = self.metrics.grad_node(Emk_1, extended_input=True, compute_dy=False)[:2]
        
        Emk_2 = (mu + mu_t/sigma_k) * dk_dy
        Emk_2_dx, Emk_2_dy = self.metrics.grad_node(Emk_2, extended_input=True, compute_dx=False)[:2]
        #model of eps in epsilon eqn
        Emeps_1 = (mu + mu_t/sigma_eps) * deps_dx
        Emeps_1_dx, Emeps_1_dy = self.metrics.grad_node(Emeps_1, extended_input=True, compute_dy=False)[:2]
        
        Emeps_2 = (mu + mu_t/sigma_eps) * deps_dy
        Emeps_2_dx, Emeps_2_dy = self.metrics.grad_node(Emeps_2, extended_input=True, compute_dx=False)[:2]

        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        v = self.metrics.full2int( v )
        p = self.metrics.full2int( p )
        drho_dx  = self.metrics.ext2int( drho_dx )
        drho_dy  = self.metrics.ext2int( drho_dy )
        drhoU_dx = self.metrics.ext2int( drhoU_dx )
        drhoU_dy = self.metrics.ext2int( drhoU_dy )
        drhoV_dx = self.metrics.ext2int( drhoV_dx )
        drhoV_dy = self.metrics.ext2int( drhoV_dy )
        div_vel  = self.metrics.ext2int( div_vel )
        du_dx    = self.metrics.ext2int( du_dx )
        du_dy    = self.metrics.ext2int( du_dy )
        dv_dx    = self.metrics.ext2int( dv_dx )
        dv_dy    = self.metrics.ext2int( dv_dy )
        sigma_11 = self.metrics.ext2int( sigma_11 )
        sigma_22 = self.metrics.ext2int( sigma_22 )
        sigma_12 = self.metrics.ext2int( sigma_12 )
        mu_t = self.metrics.ext2int( mu_t )

        # Compute RHS terms on true interior
        # Continuity equation
        qdot0 = -( drhoU_dx + drhoV_dy )
        
        # Momentum equation - x
        conv  = ( u * drhoU_dx +
                v * drhoU_dy +
                rhoU.interior() * div_vel )
        pres  = -dp_dx / self.param.Ma**2
        visc  = ( sigma_11_dx + sigma_12_dy ) / self.param.Re
        qdot1 = pres - conv + visc + srcU
        
        # Momentum equation - y
        conv  = ( u * drhoV_dx +
                v * drhoV_dy +
                rhoV.interior() * div_vel )
        pres  = -dp_dy / self.param.Ma**2
        visc  = ( sigma_12_dx + sigma_22_dy ) / self.param.Re
        qdot2 = pres - conv + visc

        # Momentum equation - z
        qdot3 = 0.0*qdot2
        
        # Total energy equation
        conv = ( u*drhoE_dx +
                v*drhoE_dy +
                rhoE.interior() * div_vel )
        pres = ( u*dp_dx + v*dp_dy + p*div_vel ) / self.param.Ma**2
        visc = ( u*( sigma_11_dx + sigma_12_dy ) +
                v*( sigma_12_dx + sigma_22_dy ) +
                sigma_11*du_dx + sigma_12*du_dy +
                sigma_12*dv_dx + sigma_22*dv_dy )/self.param.Re
        visc_k = (Emk_1_dx + Emk_2_dy) / self.param.Re
        diff  = ( q_1_dx + q_2_dy )/( self.param.Re * self.param.Ma**2)
        qdot4 = visc + visc_k - conv - pres - diff + u*srcU

        # K-equation
        conv = ( u * drhok_dx +
                v * drhok_dy +
                rhok.interior() * div_vel )

        S12 = 0.5 * (du_dy + dv_dx)
        S11 = 0.5 * (du_dx + du_dx)
        S22 = 0.5 * (dv_dy + dv_dy)

        prod = (2 * mu_t * (S11*S11 + S12*S12 + S12*S12 + S22*S22))/self.param.Re

        qdot5 = - conv + visc_k + prod - rhoeps.interior()#- 2 * rhoeps.interior()*self.param.Ma**2 #last term is YM, compressiblity effects

        # Epsilon equation
        conv = ( u * drhoeps_dx +
                v * drhoeps_dy +
                rhoeps.interior() * div_vel )
        visc_eps = (Emeps_1_dx + Emeps_2_dy) / self.param.Re
        prod = prod * C_1eps * (((self.metrics.full2int( eps )))/(self.metrics.full2int( k )))

        qdot6 = - conv + visc_eps + prod - C_2eps*rho.interior()*(((self.metrics.full2int( eps ))**2)/(self.metrics.full2int( k )))

        # Closure model
        if self.param.Use_Model:
            input_dict = {'u':u, 'v':v,  'p':p, 'du_dy':du_dy}

            qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
                        'qdot3':qdot3, 'qdot4':qdot4}
            
            model_outputs = self.param.apply_model(self.param.model, input_dict, qdot_dict, self.grid, self.metrics, self.param)
        else:
            model_outputs = None
        
        # Boundary conditions
        # Dirichlet BCs on -/+ eta
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            qdot4[:,0,:] = 0.0
            qdot5[:,0,:] = 0.0
            qdot6[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho.interior() )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV.interior() )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE.interior() )
            qdot5 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5,:,None,None] - rhok.interior() )
            qdot6 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[6,:,None,None] - rhoeps.interior())
        
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            qdot5[:,-1,:] = 0.0
            qdot6[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho.interior() )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV.interior() )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE.interior() )
            qdot5 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5,:,None,None] - rhok.interior() )
            qdot6 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[6,:,None,None] - rhoeps.interior() )

        # Apply farfield to non-periodic Xi boundaries
        if (not self.grid.periodic_xi and self.param.iproc==0):
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            qdot5[0,:,:] = 0.0
            qdot6[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):  # for spatial planar jets, inflow is constant so this attribute wont be present
                qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho.interior() )
                qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU.interior() )
                qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV.interior() )
                qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE.interior() )
                qdot5 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5,None,:,None] - rhok.interior() )
                qdot6 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[6,None,:,None] - rhoeps.interior() )

        if (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            qdot5[-1,:,:] = 0.0
            qdot6[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho.interior() )
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV.interior() )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE.interior() )
            qdot5 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5,None,:,None] - rhok.interior() )
            qdot6 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[6,None,:,None] - rhoeps.interior() )
            
        return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4,qdot5,qdot6),dim=0),model_outputs
    

            
    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 2D
    #   Collocated visc terms - conservative form
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    #@profile
    def NS_2D(self, q,  flag=False):
        # Extract conserved variables
        
        
        if self.jvp:
            rho  = q[0]
            rhoU = q[1]
            rhoV = q[2]
            rhoW = q[2]*0.0
            rhoE = q[3]
            
            flag = q.requires_grad
        else:
            rho  = q['rho'].var
            rhoU = q['rhoU'].var
            rhoV = q['rhoV'].var
            rhoW = q['rhoW'].var
            rhoE = q['rhoE'].var
            
        rhoSC = []
        for name in self.EOS.sc_names:
            rhoSC.append(q[name].var)

        # Compute primitives including overlaps
        u = rhoU/rho
        v = rhoV/rho
        E = rhoE/rho
        SC = []
        for isc in range(self.EOS.num_sc):
            SC.append(rhoSC[isc] / rho)
            
        # Thermochemistry
        if self.jvp:
            T, p, e = self.EOS.get_TPE_tensor(q, SC)
        else:
            T, p, e = self.EOS.get_TPE(q, SC)
        
        
        # Adiabatic boundary 
        if self.param.adiabatic:
            T[5:-5,5:-5,:][:,1,:] = T[5:-5,5:-5,:][:,0,:] 
        
        # Compute 1st derivatives - true interior
        if not self.param.upwind:
            dp_dx, dp_dy = self.metrics.grad_node( p )[:2]
        
        # Compute 1st derivatives - extended interior
        drho_dx, drho_dy   = self.metrics.grad_node( rho, compute_extended=True)[:2]
        drhoU_dx, drhoU_dy = self.metrics.grad_node(rhoU, compute_extended=True )[:2]
        drhoV_dx, drhoV_dy = self.metrics.grad_node(rhoV, compute_extended=True )[:2]

        du_dx, du_dy = self.metrics.grad_node(u, compute_extended=True)[:2]
        dv_dx, dv_dy = self.metrics.grad_node(v, compute_extended=True)[:2]
        dT_dx, dT_dy = self.metrics.grad_node(T, compute_extended=True)[:2]
        dE_dx, dE_dy = self.metrics.grad_node(E, compute_extended=True)[:2]

        if not self.param.upwind:
            duu_dx, _ = self.metrics.grad_node(u*u, compute_dy=False)[:2]
            duv_dx, duv_dy = self.metrics.grad_node(u*v)[:2]
            _, dvv_dy = self.metrics.grad_node(v*v, compute_dx=False)[:2]
            drhoE_dx, drhoE_dy = self.metrics.grad_node(rhoE, compute_extended=True)[:2]
            duE_dx, _ = self.metrics.grad_node(u*E, compute_dy=False)[:2]
            _, dvE_dy = self.metrics.grad_node(v*E, compute_dx=False)[:2]
            
            drhoUU_dx, _ = self.metrics.grad_node(rhoU*u, compute_dy=False, compute_extended=True)[:2]
            drhoUV_dx, drhoUV_dy = self.metrics.grad_node(rhoU*v, compute_extended=True)[:2]
            _, drhoVV_dy = self.metrics.grad_node(rhoV*v, compute_dx=False, compute_extended=True)[:2]
            drhoUE_dx, _ = self.metrics.grad_node(rhoU*E, compute_dy=False, compute_extended=True)[:2]
            _, drhoVE_dy = self.metrics.grad_node(rhoV*E, compute_dx=False, compute_extended=True)[:2]
            
            dpu_dx, _ = self.metrics.grad_node(p*u, compute_dy=False, compute_extended=True)[:2]
            _, dpv_dy = self.metrics.grad_node(p*v, compute_dx=False, compute_extended=True)[:2]


        # Velocity and mass divergence
        div_vel = du_dx + dv_dy

        # Transport coefficients
        # mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))



        # NN-augmentation
        if self.param.Use_Model:
            input_dict = q#{'rho':rho, 'rhoU':rhoU, 'rhoV':rhoV, 'rhoE':rhoE}

            #qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
            #             'qdot3':qdot3, 'qdot4':qdot4}
            #model_outputs = self.param.apply_model(self.param.model, input_dict, qdot_dict, self.grid,
            #                                       self.metrics, self.param)
            model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
                                                   self.metrics, self.param, self.EOS)
            
            if (self.param.model_type == 'visc' or self.param.model_type == 'viscNew'):
                #mu_NN = model_outputs[:,:,:,0]
                # print(torch.norm(model_outputs[:,:,:,0]))
                # print(self.grid.lol)
                # kappa_NN = ((self.EOS.base_mu * self.EOS.cp) / (self.EOS.Pr)) * model_outputs[:,:,:,1]

                mu_NN = model_outputs[:,:,:,0]
                kappa_NN = model_outputs[:,:,:,1]
                self.modelOutputs = model_outputs
                
                
                #print(torch.sum(torch.isnan(mu_NN)) + torch.sum(torch.isnan(mu_NN)))
            

                # masking out outputs 
                # mask = torch.ones_like(mu_NN)
                # mask[:self.grid.xIndOptLeft, :, :] = 0.0
                # mask[self.grid.xIndOptRight:, :, :] = 0.0
                # mask[:, :self.grid.yIndOptTop, :] = 0.0
                # mask[:, self.grid.yIndOptBot:, :] = 0.0

                # mu_NN = mu_NN * mask
                # kappa_NN = kappa_NN * mask


                # Might not work in parallel! Would have to include overlaps into model I think
                mu_NN = self.metrics.expand_overlaps(mu_NN)
                kappa_NN = self.metrics.expand_overlaps(kappa_NN)

                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))             
                #mu =  (mu + 0.0 * mu_NN)#mu + mu_NN #
                # kappa = (kappa  + kappa_NN) #kappa + kappa_NN

                mu =  mu * (1.0 + mu_NN)#mu + mu_NN #
                #kappa = (mu * self.EOS.cp / self.EOS.Pr) #* (1.0 + kappa_NN)
                kappa = kappa * (1.0  + kappa_NN) #kappa + kappa_NN
                
                self.modMu = mu
                self.modKappa = kappa

                #sourceGauss = model_outputs[:,:,:,0]
                #mu = self.metrics.expand_overlaps(model_outputs[:,:,:,0])

                """ mu = mu_NN
                kappa = kappa_NN """
                model_U = 0.0
            elif self.param.model_type == 'source' or self.param.model_type == 'const':
                model_U = model_outputs
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                
            elif self.param.model_type == 'constant':
                # print('Entering')
                # print(model_outputs)
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                mu        = mu * model_outputs # self.EOS.base_mu
                kappa     = mu * self.EOS.cp / self.EOS.Pr# model_outputs * kappa
                
                # if flag:
                #     self.constList.append(copy.deepcopy(model_outputs.detach().numpy()))
                #     print(model_outputs)

        else:
            model_outputs = None 
            # Transport coefficients
            mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))

            # # Computing source term
            # x_grid, y_grid = self.grid.get_xy(self.grid.xi_grid, self.grid.eta_grid)
            # X, Y = torch.meshgrid(x_grid, y_grid)
    
            # x0 = 0.25
            # y0 = 0.25
            # sigma_x = 0.1
            # sigma_y = 0.1
            # Amp = 1e0
    
            # sourceGauss = Amp * torch.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2)))
            # sourceGauss = sourceGauss[:,:,None]
            
            # Using only the base value 
            # mu    = self.EOS.base_mu
            # kappa = self.EOS.base_mu / (self.EOS.Pr * self.EOS.Ma**2)
            model_U = 0.0


        # Scalar gradients - extended interior for 2nd derivatives
        dSC_dx = []
        dSC_dy = []
        for isc in range(self.EOS.num_sc):
            _dx, _dy = self.metrics.grad_node( SC[isc], compute_extended=True )[:2]
            dSC_dx.append(_dx)
            dSC_dy.append(_dy)

        # Artificial diffusivity (Kawai & Lele JCP 2008)
        if self.param.artDiss:
            # Exclude from adjoint calculation
            with torch.inference_mode():
                # Strain-rate magnitude
                S = op.strainrate_mag_2D(du_dx ,du_dy ,dv_dx ,dv_dy )

                # Evaulate the artificial transport coefficients
                curl_u = dv_dx - du_dy
                mu_art,beta_art,kappa_art = op.art_diff4_D_2D(self.metrics.full2int(rho),
                                                              drho_dx ,drho_dy ,S,self.tmp_grad,
                                                              div_vel,curl_u,
                                                              self.metrics.full2int(T),
                                                              e,
                                                              self.EOS.get_soundspeed_q(q),
                                                              self.grid,self.metrics,self.param.device)

                mu_eff    = mu + mu_art
                kappa_eff = kappa + kappa_art *self.EOS.Re*self.EOS.Pr
                beta_art  = beta_art *self.EOS.Re
        else:
            mu_eff    = mu
            kappa_eff = kappa
            beta_art  = 0.0

        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                                 self.metrics.ext2int( mu_eff ),
                                                 self.metrics.ext2int( du_dy ),
                                                 None)

        # Viscous stress
        #   Divergence terms are computed on true interior
        div_term = (beta_art - 2.0*mu_eff/3.0)*div_vel

        sigma_11 = 2.0*mu_eff*du_dx + div_term
        sigma_22 = 2.0*mu_eff*dv_dy + div_term
        sigma_12 = mu_eff*( du_dy + dv_dx )

        # NOTE: Laplace viscous stress gradients do not currently include
        # artificial diffusivity
        dmu_eff_dx, dmu_eff_dy = self.metrics.grad_node(mu_eff, extended_input=True)[:2]
        dk_dx, dk_dy = self.metrics.grad_node(kappa_eff, extended_input=True)[:2]
        lap_u = self.metrics.lap(u)
        lap_v = self.metrics.lap(v)
        lap_T = self.metrics.lap(T)
        if self.param.lapStab:
            stab_strength = 1.0
            lap_rho = self.metrics.lap(rho)
            d2rho_dx2,_ = self.metrics.grad_node(drho_dx, compute_dy=False, extended_input=True)[:2]
            _,d2rho_dy2 = self.metrics.grad_node(drho_dy, compute_dx=False, extended_input=True)[:2]
            stab0 = stab_strength * (lap_rho - d2rho_dx2 - d2rho_dy2) / self.EOS.Re
            lap_E = self.metrics.lap(E)
            d2E_dx2,_ = self.metrics.grad_node(dE_dx, compute_dy=False, extended_input=True)[:2]
            _,d2E_dy2 = self.metrics.grad_node(dE_dy, compute_dx=False, extended_input=True)[:2]
            stab4 = stab_strength * (lap_E - d2E_dx2 - d2E_dy2) / self.EOS.Re / self.EOS.Pr
            if self.param.jproc == 0:
                stab0[:, :3, :] = 0.0
                stab4[:, :3, :] = 0.0
            if self.param.jproc == self.param.npy - 1:
                stab0[:, -4:, :] = 0.0
                stab4[:, -4:, :] = 0.0
        else:
            stab0 = 0.0
            stab4 = 0.0
        #stab0 = 0.0

        ddiv_vel_dx, ddiv_vel_dy = self.metrics.grad_node(div_vel, extended_input=True)[:2]

        # Species diffusion coefficients on extended interior
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.full2ext( SC[isc] )
        DIFF = self.EOS.get_species_diff_coeff(self.metrics.full2ext(T), SC)

        # Species diffusion terms - Fickian diffusion for now
        SC_diff_1_dx = []
        SC_diff_2_dy = []
        rho_ext = self.metrics.full2ext(rho)
        for isc,name in enumerate(self.EOS.sc_names):
            if (name == 'rhoZmix'):
                SC_diff_1_dx.append( 0.0 )
                SC_diff_2_dy.append( 0.0 )

            else:
                SC_diff_1 = rho_ext * DIFF[isc] * dSC_dx[isc]
                _dx, _ = self.metrics.grad_node(SC_diff_1,  extended_input=True, compute_dy=False)[:2]
                SC_diff_1_dx.append(_dx)

                SC_diff_2 = rho_ext * DIFF[isc] * dSC_dy[isc]
                _, _dy = self.metrics.grad_node(SC_diff_2,  extended_input=True, compute_dx=False)[:2]
                SC_diff_2_dy.append(_dy)

        # Heat flux
        q_1 = -kappa_eff * dT_dx
        q_1_dx,_ = self.metrics.grad_node(q_1, extended_input=True, compute_dy=False)[:2]
        
        q_2 = -kappa_eff * dT_dy
        _,q_2_dy = self.metrics.grad_node(q_2, extended_input=True, compute_dx=False)[:2]

        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        v = self.metrics.full2int( v )
        p = self.metrics.full2int( p )
        E = self.metrics.full2int(E)
        drho_dx   = self.metrics.ext2int( drho_dx )
        drho_dy   = self.metrics.ext2int( drho_dy )
        drhoU_dx  = self.metrics.ext2int( drhoU_dx )
        drhoU_dy  = self.metrics.ext2int( drhoU_dy )
        drhoV_dx  = self.metrics.ext2int( drhoV_dx )
        drhoV_dy  = self.metrics.ext2int( drhoV_dy )
        div_vel   = self.metrics.ext2int( div_vel )
        du_dx     = self.metrics.ext2int( du_dx )
        du_dy     = self.metrics.ext2int( du_dy )
        dv_dx     = self.metrics.ext2int( dv_dx )
        dv_dy     = self.metrics.ext2int( dv_dy )
        dE_dx     = self.metrics.ext2int( dE_dx )
        dE_dy     = self.metrics.ext2int( dE_dy )
        sigma_11  = self.metrics.ext2int( sigma_11 )
        sigma_22  = self.metrics.ext2int( sigma_22 )
        sigma_12  = self.metrics.ext2int( sigma_12 )
        if not self.param.upwind:
            drhoUU_dx = self.metrics.ext2int(drhoUU_dx)
            drhoUV_dx = self.metrics.ext2int(drhoUV_dx)
            drhoUV_dy = self.metrics.ext2int(drhoUV_dy)
            drhoVV_dy = self.metrics.ext2int(drhoVV_dy)
            drhoUE_dx = self.metrics.ext2int(drhoUE_dx)
            drhoVE_dy = self.metrics.ext2int(drhoVE_dy)
            dpu_dx    = self.metrics.ext2int(dpu_dx)
            dpv_dy    = self.metrics.ext2int(dpv_dy)
            drhoE_dx = self.metrics.ext2int(drhoE_dx)
            drhoE_dy = self.metrics.ext2int(drhoE_dy)
    
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.ext2int( SC[isc] )
            dSC_dx[isc] = self.metrics.ext2int( dSC_dx[isc] )
            dSC_dy[isc] = self.metrics.ext2int( dSC_dy[isc] )

        mu_eff = self.metrics.ext2int(mu_eff)
        kappa_eff = self.metrics.ext2int(kappa_eff)
        dT_dx = self.metrics.ext2int(dT_dx)
        dT_dy = self.metrics.ext2int(dT_dy)

        if self.param.upwind:
            if self.param.advection_scheme == 'upwind_StegerWarming':
                # Euler fluxes - Steger-Warming
                                
                div_fx, div_fy = self.metrics.Steger_Warming_Fluxes( rho,
                                                                     rhoU,
                                                                     rhoV, 
                                                                     rhoE,
                                                                     self.EOS )
            else:
                # Euler fluxes - Upwind or HLLE
                div_fx, div_fy, div_fz = self.metrics.Euler_Fluxes( rho,
                                                                    rhoU,
                                                                    rhoV,
                                                                    rhoW,
                                                                    rhoE,
                                                                    self.EOS )
            
        
        rho  = self.metrics.full2int( rho )
        rhoU = self.metrics.full2int( rhoU )
        rhoV = self.metrics.full2int( rhoV )
        rhoE = self.metrics.full2int( rhoE )
        for isc in range(self.EOS.num_sc):
            rhoSC[isc] = self.metrics.ext2int( rhoSC[isc] )
            
        stress_div_x = (
            mu_eff * lap_u +
            (1/3) * mu_eff * ddiv_vel_dx +
            dmu_eff_dx * (du_dx + du_dx) + dmu_eff_dy * (du_dy + dv_dx) -
            (2/3) * dmu_eff_dx * div_vel
        )
        stress_div_y = (
            mu_eff * lap_v +
            (1/3) * mu_eff * ddiv_vel_dy +
            dmu_eff_dx * (dv_dx + du_dy) + dmu_eff_dy * (dv_dy + dv_dy) -
            (2/3) * dmu_eff_dy * div_vel
        )
        q_div = - kappa_eff * lap_T - dk_dx * dT_dx - dk_dy * dT_dy
            
        # Compute RHS terms on true interior
        # Kennedy and Gruber (2008) convective term. Skew-symmetric forms for
        # continuity equation and pressure work.
        # Continuity equation
        if self.param.upwind:
            qdot0 = -(div_fx[0] + div_fy[0]) + stab0
            # qdot0 = 0.0 * (-(div_fx[0] + div_fy[0]) + stab0)
        else:
            u_dot_drho = u * drho_dx + v * drho_dy
            div_mass = drhoU_dx + drhoV_dy
            # qdot0 = -0.5 * (
            #     div_mass +
            #     rho * div_vel +
            #     u_dot_drho) + stab0

            qdot0 = 0.0 * (-0.5 * (
                div_mass +
                rho * div_vel +
                u_dot_drho) + stab0)

        # Momentum equation - x

        visc = stress_div_x
        if self.param.upwind:
            qdot1 = -(div_fx[1] + div_fy[1]) + visc #+ sourceGauss
            # qdot1 = visc
        else:
            conv = 0.25 * (
                drhoUU_dx + drhoUV_dy +
                rho * (duu_dx + duv_dy) + u * div_mass + u * drhoU_dx + v * drhoU_dy +
                u * u_dot_drho + rhoU * du_dx + rhoV * du_dy + rhoU * div_vel)
            pres  = -dp_dx * self.EOS.P_fac
            # qdot1 = pres - conv + visc + srcU
            qdot1 = pres - visc
            
        # Momentum equation - y
        visc = stress_div_y
        if self.param.upwind:
            qdot2 = -(div_fx[2] + div_fy[2]) + visc
            # qdot2 = 0.0 * visc
        else:
            conv = 0.25 * (
                drhoUV_dx + drhoVV_dy +
                rho * (duv_dx + dvv_dy) + v * div_mass + u * drhoV_dx + v * drhoV_dy +
                v * u_dot_drho + rhoU * dv_dx + rhoV * dv_dy + rhoV * div_vel)
            pres  = -dp_dy * self.EOS.P_fac
            # qdot2 = pres - conv + visc
            qdot2 = 0.0 * (pres - visc)
            

        # Momentum equation - z
        qdot3 = 0.0*qdot2
        
        # Total energy equation
        visc = (u * stress_div_x + v * stress_div_y +
                sigma_11 * du_dx + sigma_12 * du_dy +
                sigma_12 * dv_dx + sigma_22 * dv_dy)
        diff = q_div
        if self.param.upwind:
            # qdot4 = 0.0 * (-(div_fx[3] + div_fy[3]) + visc - diff) #+ stab4
            qdot4 = (-(div_fx[3] + div_fy[3]) + visc - diff)  #+ stab4
        else:
            conv = 0.25 * (
                drhoUE_dx + drhoVE_dy +
                rho * (duE_dx + dvE_dy) + E * div_mass + u * drhoE_dx + v * drhoE_dy +
                E * u_dot_drho + rhoU * dE_dx + rhoV * dE_dy + rhoE * div_vel)
            pres = 0.5 * (
                dpu_dx + dpv_dy +
                p * div_vel + u * dp_dx + v * dp_dy) * self.EOS.P_fac
            # qdot4 = (visc - conv - pres - diff + u * srcU + stab4)
            qdot4 = (visc - pres)

        # Species equations
        srcSC  = self.EOS.get_species_production_rates(rho, self.metrics.full2int(T), SC)
        qdotSC = []
        for isc in range(self.EOS.num_sc):
            conv = ( rhoU * dSC_dx[isc] +
                     rhoV * dSC_dy[isc] +
                     SC[isc] * (drhoU_dx + drhoV_dy) )
            diff = SC_diff_1_dx[isc] + SC_diff_2_dy[isc]
            qdot = diff - conv + srcSC[isc]
            qdotSC.append(qdot)
        

        # Closure model
        # if self.param.Use_Model:
        #     input_dict = q#{'u':u, 'v':v,  'p':p, 'du_dy':du_dy}

        #     qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
        #                  'qdot3':qdot3, 'qdot4':qdot4}
            
        #     model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
        #                                            self.metrics, self.param, self.EOS)
        # else:
        #     model_outputs = None

            
        # Boundary conditions
        # -----------------------------------------------------------------------
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            
            # dirichlet on rho
            # qdot0[:,0,:] = 0.0

            # dirichlet on U, V
            # qdot1[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= u[:, 0, :]

            # qdot2[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= v[:, 0, :]
            
            # dirichlet on rhoU, rhoV
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
            
            # Isothermal wall
            if not self.param.adiabatic:
                qdot4[:,0,:] = qdot0[:, 0, :]
                qdot4[:,0,:] *= E[:, 0, :]
            
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5+isc,:,None,None] - rhoSC[isc] )

        # -----------------------------------------------------------------------
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
            
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            # Farfield
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5+isc,:,None,None] - rhoSC[isc] )

        if (self.grid.BC_eta_top=='supersonic' and self.param.jproc==self.param.npy-1):
                # Supersonic outflow boundary - eta top
                qdot0[:,-1,:] = qdot0[:,-2,:]
                qdot1[:,-1,:] = qdot1[:,-2,:]
                qdot2[:,-1,:] = qdot2[:,-2,:]
                qdot4[:,-1,:] = qdot4[:,-2,:]

                # qdot0[:,-1,:] = 0.0
                # qdot1[:,-1,:] = 0.0
                # qdot2[:,-1,:] = 0.0
                # qdot4[:,-1,:] = 0.0

        # -----------------------------------------------------------------------
        # Xi left boundary
        if (hasattr(self.grid, "BC_xi_left") and self.grid.BC_xi_left=='supersonic' and self.param.iproc==0):
            # Supersonic outflow boundary
            qdot0[0,:,:] = qdot0[1,:,:]
            qdot1[0,:,:] = qdot1[1,:,:]
            qdot2[0,:,:] = qdot2[1,:,:]
            qdot4[0,:,:] = qdot4[1,:,:]
                
        elif (not self.grid.periodic_xi and self.param.iproc==0):
            # Farfield
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):
                
                if (not self.grid.BC_xi_left  == 'symmetric'):
                    qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho )
                    qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU )
                    qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV )
                    qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE )
                    # Species
                    for isc in range(self.EOS.num_sc):
                        qdotSC[isc] += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5+isc,None,:,None] - rhoSC[isc] )
                        
                else:
                    

                    # Symmetric boundary 
                    qdot0[0:self.grid.wallPoint,0,:] = qdot0[0:self.grid.wallPoint,1,:]  ## FIX! HARDCODED
                    qdot1[0:self.grid.wallPoint,0,:] = qdot1[0:self.grid.wallPoint,1,:]
                    #qdot2[0:51,0,:] = qdot2[0:51,1,:]
                    qdot4[0:self.grid.wallPoint,0,:] = qdot4[0:self.grid.wallPoint,1,:]
                    
                                
        # -----------------------------------------------------------------------
        # Xi right boundary
        if (hasattr(self.grid, "BC_xi_right") and self.grid.BC_xi_right=='supersonic' and
            self.param.iproc==self.param.npx-1):
            # Supersonic outflow boundary
            qdot0[-1,:,:] = qdot0[-2,:,:]
            qdot1[-1,:,:] = qdot1[-2,:,:]
            qdot2[-1,:,:] = qdot2[-2,:,:]
            qdot4[-1,:,:] = qdot4[-2,:,:]

            # qdot0[-1,:,:] = 0.0
            # qdot1[-1,:,:] = 0.0
            # qdot2[-1,:,:] = 0.0
            # qdot4[-1,:,:] = 0.0


        elif (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            # Farfield
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho )
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5+isc,None,:,None] - rhoSC[isc] )

        # Slip wall boundary 
        slipWall = True
        if slipWall:
            
            
            
            
            # print('Entering slip')
            # Under-relaxation factor
            facUR_U = 1.0
            facUR_T = 1.0

            # Updating the variables
            rho_upd  = rho.detach() + self.param.dt * qdot0.clone()
            rhoU_upd = rhoU.detach() + self.param.dt * qdot1.clone()
            rhoV_upd = rhoV.detach() + self.param.dt * qdot2.clone()
            rhoE_upd = rhoE.detach() + self.param.dt * qdot4.clone()
            q_upd    = torch.zeros(4, self.grid.Nx1+10, self.grid.Nx2+10, 1).to(self.param.device)

            q_upd[0][5:-5,5:-5,:] = rho_upd
            q_upd[1][5:-5,5:-5,:] = rhoU_upd
            q_upd[2][5:-5,5:-5,:] = rhoV_upd
            q_upd[3][5:-5,5:-5,:] = rhoE_upd   

            # Recomputing primitives
            u_upd = rhoU_upd/rho_upd
            v_upd = rhoV_upd/rho_upd
            E_upd = rhoE_upd/rho_upd
            

            T_upd, p_upd, e_upd = self.EOS.get_TPE_tensor(q_upd, SC)
            
            T_upd = self.metrics.full2int(T_upd)

            # indexes
            # wm_index_1 = 2
            # wm_index_2 = 3
            # wm_index_3 = 4
            # wm_index_4 = 5
            
            # extracting wall model outputs and reshpaing 
            
            if False:#self.param.Use_Model:
                # wmo_1 = (model_outputs[:,:,:,wm_index_1])
                # wmo_2 = (model_outputs[:,:,:,wm_index_2])
                # wmo_3 = (model_outputs[:,:,:,wm_index_3])
                # wmo_4 = (model_outputs[:,:,:,wm_index_4])
                
                model_inputs = torch.stack(((T_upd/self.EOS.T0).squeeze(), (u_upd/self.EOS.U0).squeeze()),dim=2)
                model_outputs = self.param.model.forward(model_inputs)
    
    
                # Creating masks 
                mask1 = torch.ones_like(T_upd)
                #mask1[self.grid.wallPoint:, 0, :] = 0.
                
                mask2 = torch.zeros_like(T_upd)
                #mask2[self.grid.wallPoint:, 0, :] = 1.
    
                # Defining the accomodation coefficients
                sig = 1.0 * mask1 + model_outputs[:,:,0].unsqueeze(dim=2) * mask2
                alph = 1.0 * mask1 + model_outputs[:,:,1].unsqueeze(dim=2) * mask2
                A    = (2.0/torch.pi) ** 0.5 * mask1 + model_outputs[:,:,2].unsqueeze(dim=2) * ((2.0/torch.pi) ** 0.5 + 1.) * mask2
                Twall = self.EOS.T0 * mask1 + model_outputs[:,:,3].unsqueeze(dim=2) * (1. + 300.) * mask2
                Umul = 1.0 * mask1 + model_outputs[:,:,4].unsqueeze(dim=2) * mask2
                Tmul = 1.0 * mask1 + model_outputs[:,:,5].unsqueeze(dim=2) * mask2
                
                
            else:
                sig = 1.0 
                alph = 1.0
                A    = (2.0/torch.pi) ** 0.5
                Twall = 300#292#self.EOS.T0
                Umul = 1.0
                Tmul = 1.0

            
            # Computing mean free paths
            mu, kappa = self.EOS.get_mu_kappa(T_upd)
            
            mfp    = (mu/rho_upd) * (torch.pi/(2 * self.EOS.Rgas * T_upd)) ** 0.5
            mfp_T  = (2/(self.EOS.gamma + 1)) * (kappa / (rho_upd * self.EOS.cv)) * (torch.pi / (2 * self.EOS.Rgas * T_upd)) ** 0.5

            # mfp = self.get_mfp()

            # Computing slip-velocity
            # Vs = facUR_U * A * (2-sig)/sig * mfp[grid.wallPoint:, 0, 0] * du_dy[grid.wallPoint:, 0, 0] # (u[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # u[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]  = Vs
            u_fac = Umul * A * (2-sig)/sig * mfp * (1 / self.grid.Y[self.grid.wallPoint+1, 1])
            #u_fac = facUR_U * A * (2-sig)/sig * 1 * (1 / self.grid.Y[self.grid.wallPoint:, 1])
            
            # cloning and shifting u values
            u_upd_shift = u_upd.clone()
            u_upd_shift[:, [0,1], :] = u_upd_shift[:, [1,0], :]
            
            Vs    = (u_fac / (1.0 + u_fac)) * u_upd_shift

            mask = torch.ones_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            u_upd = u_upd * mask
            
            mask = torch.zeros_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            Vs = Vs * mask
            
            u_upd  += Vs
            
            # u[self.grid.wallPoint:, 0, 0]  = Vs
            
            # Computing temperature jump 
            # delT = facUR_T * (2 - alph) / alph * mfp_T[grid.wallPoint:, 0, 0] * dT_dy[grid.wallPoint:, 0, 0] # (T[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # T[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]    = EOS.T0 + delT
            T_fac  = Tmul * ((2 - alph) / alph) * mfp_T * (1 / self.grid.Y[self.grid.wallPoint+1, 1])

            # Creating mask
            mask = torch.ones_like(T_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            T_upd = T_upd * mask
            
            mask = torch.zeros_like(self.metrics.full2int(T))
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            T_shifted = self.metrics.full2int(T).clone()
            T_shifted[:, [0,1], :] = T_shifted[:, [1,0], :]
            T_masked =  T_shifted * mask
            T_wall   = Twall * mask
            T_fac     = T_fac * mask

            #T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 0, 0] += (T_fac * T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 1, 0] + Twall) / (1.0 + T_fac)
            T_upd += (T_fac * T_masked + T_wall) / (1.0 + T_fac)

            e_upd = self.EOS.get_internal_energy_TY(T_upd)

            # Creating a mask
            mask = torch.ones_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            rhoU_upd = rhoU_upd * mask
            rhoE_upd = rhoE_upd * mask

            rhoU_upd += Vs * rho_upd
            rhoE_upd += (rho_upd * (e_upd + 1.0 / 2.0 * (u_upd**2 + v_upd**2) ))

            qdot1 = qdot1 * mask
            qdot4 = qdot4 * mask
            
            mask = torch.zeros_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0

            qdot1 += (rhoU_upd * mask - rhoU * mask) / self.param.dt
            qdot4 += (rhoE_upd * mask - rhoE * mask) / self.param.dt 


        if self.jvp:
            # Padding to maintain dimensional consistency 
            R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
            
            qdot_list = [qdot0,qdot1,qdot2,qdot3,qdot4] + qdotSC
            qdot_stack = torch.stack((R(qdot_list[0]), R(qdot_list[1]), R(qdot_list[2]), R(qdot_list[4])), dim=0)
            
            if flag:
                
                xLeftCut  =  self.grid.xIndOptLeft  + 5 
                xRightCut =  self.grid.xIndOptRight - 5 
                yTopCut   =  self.grid.yIndOptTop   - 5 
                yBotCut   =  self.grid.yIndOptBot   + 5 

                # Eta-direction
                qdot_stack[:, :xLeftCut, :, :] = q[:, :xLeftCut, :, :]
                qdot_stack[:, xRightCut:, :, :] = q[:, xRightCut:, :, :]
                
                # Xi-direction
                qdot_stack[:, :yBotCut, :, :] = q[:, :yBotCut, :, :]
                qdot_stack[:, yTopCut:, :, :] = q[:, yTopCut:, :, :]
                
    
            return qdot_stack

        else:
            return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4),dim=0), model_outputs

    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 2D
    #   Collocated visc terms - conservative form
    #   Anisotropic viscocities 
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    #@profile
    def NS_2D_anisotropic(self, q,  flag=False):
        # Extract conserved variables
        
        
        if self.jvp:
            rho  = q[0]
            rhoU = q[1]
            rhoV = q[2]
            rhoW = q[2]*0.0
            rhoE = q[3]
            
            flag = q.requires_grad
        else:
            rho  = q['rho'].var
            rhoU = q['rhoU'].var
            rhoV = q['rhoV'].var
            rhoW = q['rhoW'].var
            rhoE = q['rhoE'].var
            
        rhoSC = []
        for name in self.EOS.sc_names:
            rhoSC.append(q[name].var)

        # Compute primitives including overlaps
        u = rhoU/rho
        v = rhoV/rho
        E = rhoE/rho
        SC = []
        for isc in range(self.EOS.num_sc):
            SC.append(rhoSC[isc] / rho)
            
        # Thermochemistry
        if self.jvp:
            T, p, e = self.EOS.get_TPE_tensor(q, SC)
        else:
            T, p, e = self.EOS.get_TPE(q, SC)
        
        
        # Adiabatic boundary 
        if self.param.adiabatic:
            T[5:-5,5:-5,:][:,1,:] = T[5:-5,5:-5,:][:,0,:] 
        
        # Compute 1st derivatives - true interior
        if not self.param.upwind:
            dp_dx, dp_dy = self.metrics.grad_node( p )[:2]
        
        # Compute 1st derivatives - extended interior
        #drho_dx, drho_dy   = self.metrics.grad_node( rho, compute_extended=True)[:2]
        #drhoU_dx, drhoU_dy = self.metrics.grad_node(rhoU, compute_extended=True )[:2]
        #drhoV_dx, drhoV_dy = self.metrics.grad_node(rhoV, compute_extended=True )[:2]

        du_dx, du_dy = self.metrics.grad_node(u, compute_extended=True)[:2]
        dv_dx, dv_dy = self.metrics.grad_node(v, compute_extended=True)[:2]
        dT_dx, dT_dy = self.metrics.grad_node(T, compute_extended=True)[:2]
        #dE_dx, dE_dy = self.metrics.grad_node(E, compute_extended=True)[:2]

        if not self.param.upwind:
            duu_dx, _ = self.metrics.grad_node(u*u, compute_dy=False)[:2]
            duv_dx, duv_dy = self.metrics.grad_node(u*v)[:2]
            _, dvv_dy = self.metrics.grad_node(v*v, compute_dx=False)[:2]
            drhoE_dx, drhoE_dy = self.metrics.grad_node(rhoE, compute_extended=True)[:2]
            duE_dx, _ = self.metrics.grad_node(u*E, compute_dy=False)[:2]
            _, dvE_dy = self.metrics.grad_node(v*E, compute_dx=False)[:2]
            
            drhoUU_dx, _ = self.metrics.grad_node(rhoU*u, compute_dy=False, compute_extended=True)[:2]
            drhoUV_dx, drhoUV_dy = self.metrics.grad_node(rhoU*v, compute_extended=True)[:2]
            _, drhoVV_dy = self.metrics.grad_node(rhoV*v, compute_dx=False, compute_extended=True)[:2]
            drhoUE_dx, _ = self.metrics.grad_node(rhoU*E, compute_dy=False, compute_extended=True)[:2]
            _, drhoVE_dy = self.metrics.grad_node(rhoV*E, compute_dx=False, compute_extended=True)[:2]
            
            dpu_dx, _ = self.metrics.grad_node(p*u, compute_dy=False, compute_extended=True)[:2]
            _, dpv_dy = self.metrics.grad_node(p*v, compute_dx=False, compute_extended=True)[:2]


        # Velocity and mass divergence
        div_vel = du_dx + dv_dy

        # Transport coefficients
        # mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))



        # NN-augmentation
        if self.param.Use_Model:
            input_dict = q#{'rho':rho, 'rhoU':rhoU, 'rhoV':rhoV, 'rhoE':rhoE}

            #qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
            #             'qdot3':qdot3, 'qdot4':qdot4}
            #model_outputs = self.param.apply_model(self.param.model, input_dict, qdot_dict, self.grid,
            #                                       self.metrics, self.param)
            model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
                                                   self.metrics, self.param, self.EOS)
            
            if (self.param.model_type == 'visc' or self.param.model_type == 'viscNew'):
                #mu_NN = model_outputs[:,:,:,0]
                # print(torch.norm(model_outputs[:,:,:,0]))
                # print(self.grid.lol)
                # kappa_NN = ((self.EOS.base_mu * self.EOS.cp) / (self.EOS.Pr)) * model_outputs[:,:,:,1]

                mu_NN_11 = model_outputs[:,:,:,0]
                mu_NN_12 = model_outputs[:,:,:,1]
                mu_NN_22 = model_outputs[:,:,:,2]

                kappa_NN_1 = model_outputs[:,:,:,3]
                kappa_NN_2 = model_outputs[:,:,:,4]
                
                #print(torch.sum(torch.isnan(mu_NN)) + torch.sum(torch.isnan(mu_NN)))
            

                # masking out outputs 
                # mask = torch.ones_like(mu_NN)
                # mask[:self.grid.xIndOptLeft, :, :] = 0.0
                # mask[self.grid.xIndOptRight:, :, :] = 0.0
                # mask[:, :self.grid.yIndOptTop, :] = 0.0
                # mask[:, self.grid.yIndOptBot:, :] = 0.0

                # mu_NN = mu_NN * mask
                # kappa_NN = kappa_NN * mask


                # Might not work in parallel! Would have to include overlaps into model I think
                mu_NN_11 = self.metrics.expand_overlaps(mu_NN_11)
                mu_NN_12 = self.metrics.expand_overlaps(mu_NN_12)
                mu_NN_22 = self.metrics.expand_overlaps(mu_NN_22)

                kappa_NN_1 = self.metrics.expand_overlaps(kappa_NN_1)
                kappa_NN_2 = self.metrics.expand_overlaps(kappa_NN_2)

                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))             
                #mu =  (mu + 0.0 * mu_NN)#mu + mu_NN #
                # kappa = (kappa  + kappa_NN) #kappa + kappa_NN

                mu_11 =  mu * (1.0 + mu_NN_11)#mu + mu_NN #
                mu_12 =  mu * (1.0 + mu_NN_12)#mu + mu_NN #
                mu_22 =  mu * (1.0 + mu_NN_22)#mu + mu_NN #

                #kappa = (mu * self.EOS.cp / self.EOS.Pr) #* (1.0 + kappa_NN)
                kappa_1 = kappa * (1.0  + kappa_NN_1) #kappa + kappa_NN
                kappa_2 = kappa * (1.0  + kappa_NN_2) #kappa + kappa_NN

                #sourceGauss = model_outputs[:,:,:,0]
                #mu = self.metrics.expand_overlaps(model_outputs[:,:,:,0])

                """ mu = mu_NN
                kappa = kappa_NN """
                model_U = 0.0
            elif self.param.model_type == 'source' or self.param.model_type == 'const':
                model_U = model_outputs
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                
            elif self.param.model_type == 'constant':
                # print('Entering')
                # print(model_outputs)
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                mu        = mu * model_outputs # self.EOS.base_mu
                kappa     = mu * self.EOS.cp / self.EOS.Pr# model_outputs * kappa
                
                # if flag:
                #     self.constList.append(copy.deepcopy(model_outputs.detach().numpy()))
                #     print(model_outputs)

        else:
            model_outputs = None 
            # Transport coefficients
            mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))

            # anisotropic viscocities
            mu_11 = mu
            mu_12 = mu
            mu_22 = mu

            kappa_1 = kappa
            kappa_2 = kappa

            # # Computing source term
            # x_grid, y_grid = self.grid.get_xy(self.grid.xi_grid, self.grid.eta_grid)
            # X, Y = torch.meshgrid(x_grid, y_grid)
    
            # x0 = 0.25
            # y0 = 0.25
            # sigma_x = 0.1
            # sigma_y = 0.1
            # Amp = 1e0
    
            # sourceGauss = Amp * torch.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2)))
            # sourceGauss = sourceGauss[:,:,None]
            
            # Using only the base value 
            # mu    = self.EOS.base_mu
            # kappa = self.EOS.base_mu / (self.EOS.Pr * self.EOS.Ma**2)
            model_U = 0.0


        # Scalar gradients - extended interior for 2nd derivatives
        dSC_dx = []
        dSC_dy = []
        for isc in range(self.EOS.num_sc):
            _dx, _dy = self.metrics.grad_node( SC[isc], compute_extended=True )[:2]
            dSC_dx.append(_dx)
            dSC_dy.append(_dy)

        # Artificial diffusivity (Kawai & Lele JCP 2008)
        if self.param.artDiss:
            # Exclude from adjoint calculation
            with torch.inference_mode():
                # Strain-rate magnitude
                S = op.strainrate_mag_2D(du_dx ,du_dy ,dv_dx ,dv_dy )

                # Evaulate the artificial transport coefficients
                curl_u = dv_dx - du_dy
                mu_art,beta_art,kappa_art = op.art_diff4_D_2D(self.metrics.full2int(rho),
                                                              drho_dx ,drho_dy ,S,self.tmp_grad,
                                                              div_vel,curl_u,
                                                              self.metrics.full2int(T),
                                                              e,
                                                              self.EOS.get_soundspeed_q(q),
                                                              self.grid,self.metrics,self.param.device)

                mu_eff    = mu + mu_art
                kappa_eff = kappa + kappa_art *self.EOS.Re*self.EOS.Pr
                beta_art  = beta_art *self.EOS.Re
        else:
            mu_eff    = mu
            kappa_eff = kappa
            mu_eff_11 = mu_11
            mu_eff_12 = mu_12
            mu_eff_22 = mu_22
            
            kappa_eff_1 = kappa_1
            kappa_eff_2 = kappa_2
            beta_art  = 0.0

        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                                 self.metrics.ext2int( mu_eff ),
                                                 self.metrics.ext2int( du_dy ),
                                                 None)

        # Viscous stress
        #   Divergence terms are computed on true interior
        div_term_11 = (beta_art - 2.0*mu_eff_11/3.0)*div_vel
        div_term_22 = (beta_art - 2.0*mu_eff_22/3.0)*div_vel

        sigma_11 = 2.0*mu_eff_11*du_dx + div_term_11
        sigma_11_dx,_ = self.metrics.grad_node(sigma_11, extended_input=True, compute_dy=False)[:2]
        
        sigma_22 = 2.0*mu_eff_22*dv_dy + div_term_22
        _,sigma_22_dy = self.metrics.grad_node(sigma_22, extended_input=True, compute_dx=False)[:2]
        
        sigma_12 = mu_eff_12*( du_dy + dv_dx )
        sigma_12_dx,sigma_12_dy = self.metrics.grad_node(sigma_12, extended_input=True)[:2]

        # NOTE: Laplace viscous stress gradients do not currently include
        # artificial diffusivity
        # dmu_eff_dx, dmu_eff_dy = self.metrics.grad_node(mu_eff, extended_input=True)[:2]
        # dk_dx, dk_dy = self.metrics.grad_node(kappa_eff, extended_input=True)[:2]
        # lap_u = self.metrics.lap(u)
        # lap_v = self.metrics.lap(v)
        # lap_T = self.metrics.lap(T)
        if self.param.lapStab:
            stab_strength = 1.0
            lap_rho = self.metrics.lap(rho)
            d2rho_dx2,_ = self.metrics.grad_node(drho_dx, compute_dy=False, extended_input=True)[:2]
            _,d2rho_dy2 = self.metrics.grad_node(drho_dy, compute_dx=False, extended_input=True)[:2]
            stab0 = stab_strength * (lap_rho - d2rho_dx2 - d2rho_dy2) / self.EOS.Re
            lap_E = self.metrics.lap(E)
            d2E_dx2,_ = self.metrics.grad_node(dE_dx, compute_dy=False, extended_input=True)[:2]
            _,d2E_dy2 = self.metrics.grad_node(dE_dy, compute_dx=False, extended_input=True)[:2]
            stab4 = stab_strength * (lap_E - d2E_dx2 - d2E_dy2) / self.EOS.Re / self.EOS.Pr
            if self.param.jproc == 0:
                stab0[:, :3, :] = 0.0
                stab4[:, :3, :] = 0.0
            if self.param.jproc == self.param.npy - 1:
                stab0[:, -4:, :] = 0.0
                stab4[:, -4:, :] = 0.0
        else:
            stab0 = 0.0
            stab4 = 0.0
        #stab0 = 0.0

        ddiv_vel_dx, ddiv_vel_dy = self.metrics.grad_node(div_vel, extended_input=True)[:2]

        # Species diffusion coefficients on extended interior
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.full2ext( SC[isc] )
        DIFF = self.EOS.get_species_diff_coeff(self.metrics.full2ext(T), SC)

        # Species diffusion terms - Fickian diffusion for now
        SC_diff_1_dx = []
        SC_diff_2_dy = []
        rho_ext = self.metrics.full2ext(rho)
        for isc,name in enumerate(self.EOS.sc_names):
            if (name == 'rhoZmix'):
                SC_diff_1_dx.append( 0.0 )
                SC_diff_2_dy.append( 0.0 )

            else:
                SC_diff_1 = rho_ext * DIFF[isc] * dSC_dx[isc]
                _dx, _ = self.metrics.grad_node(SC_diff_1,  extended_input=True, compute_dy=False)[:2]
                SC_diff_1_dx.append(_dx)

                SC_diff_2 = rho_ext * DIFF[isc] * dSC_dy[isc]
                _, _dy = self.metrics.grad_node(SC_diff_2,  extended_input=True, compute_dx=False)[:2]
                SC_diff_2_dy.append(_dy)

        # Heat flux
        q_1 = -kappa_eff_1 * dT_dx
        q_1_dx,_ = self.metrics.grad_node(q_1, extended_input=True, compute_dy=False)[:2]
        
        q_2 = -kappa_eff_2 * dT_dy
        _,q_2_dy = self.metrics.grad_node(q_2, extended_input=True, compute_dx=False)[:2]
        
        ########### Only for comparisons
        div_term = (beta_art - 2.0*mu_eff/3.0)*div_vel

        sigma_11_plot = 2.0*mu_eff*du_dx + div_term
        sigma_22_plot = 2.0*mu_eff*dv_dy + div_term
        sigma_12_plot = mu_eff*( du_dy + dv_dx )
        
        q_1_plot = -kappa_eff * dT_dx
        q_1_dx,_ = self.metrics.grad_node(q_1, extended_input=True, compute_dy=False)[:2]
        
        q_2_plot = -kappa_eff * dT_dy
        _,q_2_dy = self.metrics.grad_node(q_2, extended_input=True, compute_dx=False)[:2]

        
        # # Coputing stress-divergence terms
        # dsigx_dx, dsigx_dy = self.metrics.grad_node(sigma_11+sigma_12, extended_input=True)[:2]
        # dsigy_dx, dsigy_dy = self.metrics.grad_node(sigma_22+sigma_12, extended_input=True)[:2]
        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        v = self.metrics.full2int( v )
        p = self.metrics.full2int( p )
        E = self.metrics.full2int(E)
        # drho_dx   = self.metrics.ext2int( drho_dx )
        # drho_dy   = self.metrics.ext2int( drho_dy )
        # drhoU_dx  = self.metrics.ext2int( drhoU_dx )
        # drhoU_dy  = self.metrics.ext2int( drhoU_dy )
        # drhoV_dx  = self.metrics.ext2int( drhoV_dx )
        # drhoV_dy  = self.metrics.ext2int( drhoV_dy )
        div_vel   = self.metrics.ext2int( div_vel )
        du_dx     = self.metrics.ext2int( du_dx )
        du_dy     = self.metrics.ext2int( du_dy )
        dv_dx     = self.metrics.ext2int( dv_dx )
        dv_dy     = self.metrics.ext2int( dv_dy )
        # dE_dx     = self.metrics.ext2int( dE_dx )
        # dE_dy     = self.metrics.ext2int( dE_dy )
        sigma_11  = self.metrics.ext2int( sigma_11 )
        sigma_22  = self.metrics.ext2int( sigma_22 )
        sigma_12  = self.metrics.ext2int( sigma_12 )
        if not self.param.upwind:
            drhoUU_dx = self.metrics.ext2int(drhoUU_dx)
            drhoUV_dx = self.metrics.ext2int(drhoUV_dx)
            drhoUV_dy = self.metrics.ext2int(drhoUV_dy)
            drhoVV_dy = self.metrics.ext2int(drhoVV_dy)
            drhoUE_dx = self.metrics.ext2int(drhoUE_dx)
            drhoVE_dy = self.metrics.ext2int(drhoVE_dy)
            dpu_dx    = self.metrics.ext2int(dpu_dx)
            dpv_dy    = self.metrics.ext2int(dpv_dy)
            drhoE_dx = self.metrics.ext2int(drhoE_dx)
            drhoE_dy = self.metrics.ext2int(drhoE_dy)
    
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.ext2int( SC[isc] )
            dSC_dx[isc] = self.metrics.ext2int( dSC_dx[isc] )
            dSC_dy[isc] = self.metrics.ext2int( dSC_dy[isc] )

        mu_eff = self.metrics.ext2int(mu_eff)
        kappa_eff = self.metrics.ext2int(kappa_eff)
        dT_dx = self.metrics.ext2int(dT_dx)
        dT_dy = self.metrics.ext2int(dT_dy)

        if self.param.upwind:
            if self.param.advection_scheme == 'upwind_StegerWarming':
                # Euler fluxes - Steger-Warming
                div_fx, div_fy = self.metrics.Steger_Warming_Fluxes( rho,
                                                                     rhoU,
                                                                     rhoV, 
                                                                     rhoE,
                                                                     self.EOS )
            else:
                # Euler fluxes - Upwind or HLLE
                div_fx, div_fy, div_fz = self.metrics.Euler_Fluxes( rho,
                                                                    rhoU,
                                                                    rhoV,
                                                                    rhoW,
                                                                    rhoE,
                                                                    self.EOS )
            
        
        rho  = self.metrics.full2int( rho )
        rhoU = self.metrics.full2int( rhoU )
        rhoV = self.metrics.full2int( rhoV )
        rhoE = self.metrics.full2int( rhoE )
        for isc in range(self.EOS.num_sc):
            rhoSC[isc] = self.metrics.ext2int( rhoSC[isc] )
            
        # stress_div_x = (
        #     mu_eff * lap_u +
        #     (1/3) * mu_eff * ddiv_vel_dx +
        #     dmu_eff_dx * (du_dx + du_dx) + dmu_eff_dy * (du_dy + dv_dx) -
        #     (2/3) * dmu_eff_dx * div_vel
        # )
        # stress_div_y = (
        #     mu_eff * lap_v +
        #     (1/3) * mu_eff * ddiv_vel_dy +
        #     dmu_eff_dx * (dv_dx + du_dy) + dmu_eff_dy * (dv_dy + dv_dy) -
        #     (2/3) * dmu_eff_dy * div_vel
        # )
        # q_div = - kappa_eff * lap_T - dk_dx * dT_dx - dk_dy * dT_dy
        
        # stress_div_x = dsigx_dx + dsigx_dy
        # stress_div_y = dsigy_dx + dsigy_dy
        # q_div         = q_1_dx + q_2_dy
            
        # Compute RHS terms on true interior
        # Kennedy and Gruber (2008) convective term. Skew-symmetric forms for
        # continuity equation and pressure work.
        # Continuity equation
        if self.param.upwind:
            qdot0 = -(div_fx[0] + div_fy[0]) + stab0
            # qdot0 = 0.0 * (-(div_fx[0] + div_fy[0]) + stab0)
        else:
            u_dot_drho = u * drho_dx + v * drho_dy
            div_mass = drhoU_dx + drhoV_dy
            # qdot0 = -0.5 * (
            #     div_mass +
            #     rho * div_vel +
            #     u_dot_drho) + stab0

            qdot0 = 0.0 * (-0.5 * (
                div_mass +
                rho * div_vel +
                u_dot_drho) + stab0)

        # Momentum equation - x

        # visc = stress_div_x
        visc = sigma_11_dx + sigma_12_dy
        if self.param.upwind:
            qdot1 = -(div_fx[1] + div_fy[1]) + visc #+ sourceGauss
            # qdot1 = visc
        else:
            conv = 0.25 * (
                drhoUU_dx + drhoUV_dy +
                rho * (duu_dx + duv_dy) + u * div_mass + u * drhoU_dx + v * drhoU_dy +
                u * u_dot_drho + rhoU * du_dx + rhoV * du_dy + rhoU * div_vel)
            pres  = -dp_dx * self.EOS.P_fac
            # qdot1 = pres - conv + visc + srcU
            qdot1 = pres - visc
            
        # Momentum equation - y
        # visc = stress_div_y
        visc = sigma_12_dx + sigma_22_dy
        if self.param.upwind:
            qdot2 = -(div_fx[2] + div_fy[2]) + visc
            # qdot2 = 0.0 * visc
        else:
            conv = 0.25 * (
                drhoUV_dx + drhoVV_dy +
                rho * (duv_dx + dvv_dy) + v * div_mass + u * drhoV_dx + v * drhoV_dy +
                v * u_dot_drho + rhoU * dv_dx + rhoV * dv_dy + rhoV * div_vel)
            pres  = -dp_dy * self.EOS.P_fac
            # qdot2 = pres - conv + visc
            qdot2 = 0.0 * (pres - visc)
            

        # Momentum equation - z
        qdot3 = 0.0*qdot2
        
        # Total energy equation
        # visc = (u * stress_div_x + v * stress_div_y +
        #         sigma_11 * du_dx + sigma_12 * du_dy +
        #         sigma_12 * dv_dx + sigma_22 * dv_dy)
        # diff = q_div
        
        visc = ( u*( sigma_11_dx + sigma_12_dy ) +
         v*( sigma_12_dx + sigma_22_dy ) +
         sigma_11*du_dx + sigma_12*du_dy +
         sigma_12*dv_dx + sigma_22*dv_dy )
        
        diff  = q_1_dx + q_2_dy
        
        if self.param.upwind:
            # qdot4 = 0.0 * (-(div_fx[3] + div_fy[3]) + visc - diff) #+ stab4
            qdot4 = (-(div_fx[3] + div_fy[3]) + visc - diff)  #+ stab4
        else:
            conv = 0.25 * (
                drhoUE_dx + drhoVE_dy +
                rho * (duE_dx + dvE_dy) + E * div_mass + u * drhoE_dx + v * drhoE_dy +
                E * u_dot_drho + rhoU * dE_dx + rhoV * dE_dy + rhoE * div_vel)
            pres = 0.5 * (
                dpu_dx + dpv_dy +
                p * div_vel + u * dp_dx + v * dp_dy) * self.EOS.P_fac
            # qdot4 = (visc - conv - pres - diff + u * srcU + stab4)
            qdot4 = (visc - pres)

        # Species equations
        srcSC  = self.EOS.get_species_production_rates(rho, self.metrics.full2int(T), SC)
        qdotSC = []
        for isc in range(self.EOS.num_sc):
            conv = ( rhoU * dSC_dx[isc] +
                     rhoV * dSC_dy[isc] +
                     SC[isc] * (drhoU_dx + drhoV_dy) )
            diff = SC_diff_1_dx[isc] + SC_diff_2_dy[isc]
            qdot = diff - conv + srcSC[isc]
            qdotSC.append(qdot)
        

        # Closure model
        # if self.param.Use_Model:
        #     input_dict = q#{'u':u, 'v':v,  'p':p, 'du_dy':du_dy}

        #     qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
        #                  'qdot3':qdot3, 'qdot4':qdot4}
            
        #     model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
        #                                            self.metrics, self.param, self.EOS)
        # else:
        #     model_outputs = None

            
        # Boundary conditions
        # -----------------------------------------------------------------------
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            
            # dirichlet on rho
            qdot0[:,0,:] = 0.0

            # dirichlet on U, V
            # qdot1[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= u[:, 0, :]

            # qdot2[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= v[:, 0, :]
            
            # dirichlet on rhoU, rhoV
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
            
            # Isothermal wall
            if not self.param.adiabatic:
                qdot4[:,0,:] = qdot0[:, 0, :]
                qdot4[:,0,:] *= E[:, 0, :]
            
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5+isc,:,None,None] - rhoSC[isc] )

        # -----------------------------------------------------------------------
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
            
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            # Farfield
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5+isc,:,None,None] - rhoSC[isc] )

        if (self.grid.BC_eta_top=='supersonic' and self.param.jproc==self.param.npy-1):
                # Supersonic outflow boundary - eta top
                qdot0[:,-1,:] = qdot0[:,-2,:]
                qdot1[:,-1,:] = qdot1[:,-2,:]
                qdot2[:,-1,:] = qdot2[:,-2,:]
                qdot4[:,-1,:] = qdot4[:,-2,:]

                # qdot0[:,-1,:] = 0.0
                # qdot1[:,-1,:] = 0.0
                # qdot2[:,-1,:] = 0.0
                # qdot4[:,-1,:] = 0.0

        # -----------------------------------------------------------------------
        # Xi left boundary
        if (hasattr(self.grid, "BC_xi_left") and self.grid.BC_xi_left=='supersonic' and self.param.iproc==0):
            # Supersonic outflow boundary
            qdot0[0,:,:] = qdot0[1,:,:]
            qdot1[0,:,:] = qdot1[1,:,:]
            qdot2[0,:,:] = qdot2[1,:,:]
            qdot4[0,:,:] = qdot4[1,:,:]
                
        elif (not self.grid.periodic_xi and self.param.iproc==0):
            # Farfield
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):
                
                if (not self.grid.BC_xi_left  == 'symmetric'):
                    qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho )
                    qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU )
                    qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV )
                    qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE )
                    # Species
                    for isc in range(self.EOS.num_sc):
                        qdotSC[isc] += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5+isc,None,:,None] - rhoSC[isc] )
                        
                else:
                    

                    # Symmetric boundary 
                    qdot0[0:self.grid.wallPoint,0,:] = qdot0[0:self.grid.wallPoint,1,:]  ## FIX! HARDCODED
                    qdot1[0:self.grid.wallPoint,0,:] = qdot1[0:self.grid.wallPoint,1,:]
                    #qdot2[0:51,0,:] = qdot2[0:51,1,:]
                    qdot4[0:self.grid.wallPoint,0,:] = qdot4[0:self.grid.wallPoint,1,:]
                    
                                
        # -----------------------------------------------------------------------
        # Xi right boundary
        if (hasattr(self.grid, "BC_xi_right") and self.grid.BC_xi_right=='supersonic' and
            self.param.iproc==self.param.npx-1):
            # Supersonic outflow boundary
            qdot0[-1,:,:] = qdot0[-2,:,:]
            qdot1[-1,:,:] = qdot1[-2,:,:]
            qdot2[-1,:,:] = qdot2[-2,:,:]
            qdot4[-1,:,:] = qdot4[-2,:,:]

            # qdot0[-1,:,:] = 0.0
            # qdot1[-1,:,:] = 0.0
            # qdot2[-1,:,:] = 0.0
            # qdot4[-1,:,:] = 0.0


        elif (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            # Farfield
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho )
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5+isc,None,:,None] - rhoSC[isc] )

        # Slip wall boundary 
        # Slip wall boundary 
        slipWall = False
        if slipWall:
            
            
            
            
            # print('Entering slip')
            # Under-relaxation factor
            facUR_U = 1.0
            facUR_T = 1.0

            # Updating the variables
            rho_upd  = rho.detach() + self.param.dt * qdot0.clone()
            rhoU_upd = rhoU.detach() + self.param.dt * qdot1.clone()
            rhoV_upd = rhoV.detach() + self.param.dt * qdot2.clone()
            rhoE_upd = rhoE.detach() + self.param.dt * qdot4.clone()
            q_upd    = q.clone()#torch.zeros(4, self.grid.Nx1, self.grid.Nx2, 1).to(self.param.device)

            q_upd[0][5:-5,5:-5,:] = rho_upd
            q_upd[1][5:-5,5:-5,:] = rhoU_upd
            q_upd[2][5:-5,5:-5,:] = rhoV_upd
            q_upd[3][5:-5,5:-5,:] = rhoE_upd   

            # Recomputing primitives
            u_upd = rhoU_upd/rho_upd
            v_upd = rhoV_upd/rho_upd
            E_upd = rhoE_upd/rho_upd
            

            T_upd, p_upd, e_upd = self.EOS.get_TPE_tensor(q_upd, SC)
            
            T_upd = self.metrics.full2int(T_upd)

            # indexes
            # wm_index_1 = 2
            # wm_index_2 = 3
            # wm_index_3 = 4
            # wm_index_4 = 5
            
            # extracting wall model outputs and reshpaing 
            
            if False:#self.param.Use_Model:
                # wmo_1 = (model_outputs[:,:,:,wm_index_1])
                # wmo_2 = (model_outputs[:,:,:,wm_index_2])
                # wmo_3 = (model_outputs[:,:,:,wm_index_3])
                # wmo_4 = (model_outputs[:,:,:,wm_index_4])
                
                model_inputs = torch.stack(((T_upd/self.EOS.T0).squeeze(), (u_upd/self.EOS.U0).squeeze()),dim=2)
                model_outputs = self.param.model.forward(model_inputs)
    
    
                # Creating masks 
                mask1 = torch.ones_like(T_upd)
                #mask1[self.grid.wallPoint:, 0, :] = 0.
                
                mask2 = torch.zeros_like(T_upd)
                #mask2[self.grid.wallPoint:, 0, :] = 1.
    
                # Defining the accomodation coefficients
                sig = 1.0 * mask1 + model_outputs[:,:,0].unsqueeze(dim=2) * mask2
                alph = 1.0 * mask1 + model_outputs[:,:,1].unsqueeze(dim=2) * mask2
                A    = (2.0/torch.pi) ** 0.5 * mask1 + model_outputs[:,:,2].unsqueeze(dim=2) * ((2.0/torch.pi) ** 0.5 + 1.) * mask2
                Twall = self.EOS.T0 * mask1 + model_outputs[:,:,3].unsqueeze(dim=2) * (1. + 300.) * mask2
                Umul = 1.0 * mask1 + model_outputs[:,:,4].unsqueeze(dim=2) * mask2
                Tmul = 1.0 * mask1 + model_outputs[:,:,5].unsqueeze(dim=2) * mask2
                
                
            else:
                sig = 1.0 
                alph = 1.0
                A    = (2.0/torch.pi) ** 0.5
                Twall = 300#292#self.EOS.T0
                Umul = 1.0
                Tmul = 1.0

            
            # Computing mean free paths
            mu, kappa = self.EOS.get_mu_kappa(T_upd)
            
            mfp    = (mu/rho_upd) * (torch.pi/(2 * self.EOS.Rgas * T_upd)) ** 0.5
            mfp_T  = (2/(self.EOS.gamma + 1)) * (kappa / (rho_upd * self.EOS.cv)) * (torch.pi / (2 * self.EOS.Rgas * T_upd)) ** 0.5

            # mfp = self.get_mfp()

            # Computing slip-velocity
            # Vs = facUR_U * A * (2-sig)/sig * mfp[grid.wallPoint:, 0, 0] * du_dy[grid.wallPoint:, 0, 0] # (u[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # u[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]  = Vs
            u_fac = Umul * A * (2-sig)/sig * mfp * (1 / self.grid.Y[self.grid.wallPoint+1, 1])
            #u_fac = facUR_U * A * (2-sig)/sig * 1 * (1 / self.grid.Y[self.grid.wallPoint:, 1])
            
            # cloning and shifting u values
            u_upd_shift = u_upd.clone()
            u_upd_shift[:, [0,1], :] = u_upd_shift[:, [1,0], :]
            
            Vs    = (u_fac / (1.0 + u_fac)) * u_upd_shift

            mask = torch.ones_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            u_upd = u_upd * mask
            
            mask = torch.zeros_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            Vs = Vs * mask
            
            u_upd  += Vs
            
            # u[self.grid.wallPoint:, 0, 0]  = Vs
            
            # Computing temperature jump 
            # delT = facUR_T * (2 - alph) / alph * mfp_T[grid.wallPoint:, 0, 0] * dT_dy[grid.wallPoint:, 0, 0] # (T[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # T[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]    = EOS.T0 + delT
            T_fac  = Tmul * ((2 - alph) / alph) * mfp_T * (1 / self.grid.Y[self.grid.wallPoint+1, 1])

            # Creating mask
            mask = torch.ones_like(T_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            T_upd = T_upd * mask
            
            mask = torch.zeros_like(self.metrics.full2int(T))
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            T_shifted = self.metrics.full2int(T).clone()
            T_shifted[:, [0,1], :] = T_shifted[:, [1,0], :]
            T_masked =  T_shifted * mask
            T_wall   = Twall * mask
            T_fac     = T_fac * mask

            #T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 0, 0] += (T_fac * T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 1, 0] + Twall) / (1.0 + T_fac)
            T_upd += (T_fac * T_masked + T_wall) / (1.0 + T_fac)

            e_upd = self.EOS.get_internal_energy_TY(T_upd)

            # Creating a mask
            mask = torch.ones_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            rhoU_upd = rhoU_upd * mask
            rhoE_upd = rhoE_upd * mask

            rhoU_upd += Vs * rho_upd
            rhoE_upd += (rho_upd * (e_upd + 1.0 / 2.0 * (u_upd**2 + v_upd**2) ))

            qdot1 = qdot1 * mask
            qdot4 = qdot4 * mask
            
            mask = torch.zeros_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0

            qdot1 += (rhoU_upd * mask - rhoU * mask) / self.param.dt
            qdot4 += (rhoE_upd * mask - rhoE * mask) / self.param.dt 



        if self.jvp:
            # Padding to maintain dimensional consistency 
            R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
            
            qdot_list = [qdot0,qdot1,qdot2,qdot3,qdot4] + qdotSC
            qdot_stack = torch.stack((R(qdot_list[0]), R(qdot_list[1]), R(qdot_list[2]), R(qdot_list[4])), dim=0)
            
            if flag:
                
                xLeftCut  =  self.grid.xIndOptLeft  + 5 
                xRightCut =  self.grid.xIndOptRight - 5 
                yTopCut   =  self.grid.yIndOptTop   - 5 
                yBotCut   =  self.grid.yIndOptBot   + 5 

                # Eta-direction
                qdot_stack[:, :xLeftCut, :, :] = q[:, :xLeftCut, :, :]
                qdot_stack[:, xRightCut:, :, :] = q[:, xRightCut:, :, :]
                
                # Xi-direction
                qdot_stack[:, :yBotCut, :, :] = q[:, :yBotCut, :, :]
                qdot_stack[:, yTopCut:, :, :] = q[:, yTopCut:, :, :]
                
    
            return qdot_stack

        else:
            return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4),dim=0), model_outputs

    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 2D
    #   Collocated visc terms - conservative form
    #   Anisotropic viscocities 
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    #@profile
    def NS_2D_anisotropic_tf(self, q,  flag=False):
        # Extract conserved variables
        
        
        if self.jvp:
            rho  = q[0]
            rhoU = q[1]
            rhoV = q[2]
            rhoW = q[2]*0.0
            rhoE = q[3]
            
            flag = q.requires_grad
        else:
            rho  = q['rho'].var
            rhoU = q['rhoU'].var
            rhoV = q['rhoV'].var
            rhoW = q['rhoW'].var
            rhoE = q['rhoE'].var
            
        rhoSC = []
        for name in self.EOS.sc_names:
            rhoSC.append(q[name].var)

        # Compute primitives including overlaps
        u = rhoU/rho
        v = rhoV/rho
        E = rhoE/rho
        SC = []
        for isc in range(self.EOS.num_sc):
            SC.append(rhoSC[isc] / rho)
            
        # Thermochemistry
        if self.jvp:
            T, p, e = self.EOS.get_TPE_tensor(q, SC)
        else:
            T, p, e = self.EOS.get_TPE(q, SC)
        
        
        # Adiabatic boundary 
        if self.param.adiabatic:
            T[5:-5,5:-5,:][:,1,:] = T[5:-5,5:-5,:][:,0,:] 
        
        # Compute 1st derivatives - true interior
        if not self.param.upwind:
            dp_dx, dp_dy = self.metrics.grad_node( p )[:2]
        
        # Compute 1st derivatives - extended interior
        #drho_dx, drho_dy   = self.metrics.grad_node( rho, compute_extended=True)[:2]
        #drhoU_dx, drhoU_dy = self.metrics.grad_node(rhoU, compute_extended=True )[:2]
        #drhoV_dx, drhoV_dy = self.metrics.grad_node(rhoV, compute_extended=True )[:2]

        du_dx, du_dy = self.metrics.grad_node(u, compute_extended=True)[:2]
        dv_dx, dv_dy = self.metrics.grad_node(v, compute_extended=True)[:2]
        dT_dx, dT_dy = self.metrics.grad_node(T, compute_extended=True)[:2]
        #dE_dx, dE_dy = self.metrics.grad_node(E, compute_extended=True)[:2]

        if not self.param.upwind:
            duu_dx, _ = self.metrics.grad_node(u*u, compute_dy=False)[:2]
            duv_dx, duv_dy = self.metrics.grad_node(u*v)[:2]
            _, dvv_dy = self.metrics.grad_node(v*v, compute_dx=False)[:2]
            drhoE_dx, drhoE_dy = self.metrics.grad_node(rhoE, compute_extended=True)[:2]
            duE_dx, _ = self.metrics.grad_node(u*E, compute_dy=False)[:2]
            _, dvE_dy = self.metrics.grad_node(v*E, compute_dx=False)[:2]
            
            drhoUU_dx, _ = self.metrics.grad_node(rhoU*u, compute_dy=False, compute_extended=True)[:2]
            drhoUV_dx, drhoUV_dy = self.metrics.grad_node(rhoU*v, compute_extended=True)[:2]
            _, drhoVV_dy = self.metrics.grad_node(rhoV*v, compute_dx=False, compute_extended=True)[:2]
            drhoUE_dx, _ = self.metrics.grad_node(rhoU*E, compute_dy=False, compute_extended=True)[:2]
            _, drhoVE_dy = self.metrics.grad_node(rhoV*E, compute_dx=False, compute_extended=True)[:2]
            
            dpu_dx, _ = self.metrics.grad_node(p*u, compute_dy=False, compute_extended=True)[:2]
            _, dpv_dy = self.metrics.grad_node(p*v, compute_dx=False, compute_extended=True)[:2]


        # Velocity and mass divergence
        div_vel = du_dx + dv_dy

        # Transport coefficients
        # mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))



        # NN-augmentation
        if self.param.Use_Model:
            input_dict = q#{'rho':rho, 'rhoU':rhoU, 'rhoV':rhoV, 'rhoE':rhoE}

            #qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
            #             'qdot3':qdot3, 'qdot4':qdot4}
            #model_outputs = self.param.apply_model(self.param.model, input_dict, qdot_dict, self.grid,
            #                                       self.metrics, self.param)
            model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
                                                   self.metrics, self.param, self.EOS)
            
            if (self.param.model_type == 'visc' or self.param.model_type == 'viscNew'):
                #mu_NN = model_outputs[:,:,:,0]
                # print(torch.norm(model_outputs[:,:,:,0]))
                # print(self.grid.lol)
                # kappa_NN = ((self.EOS.base_mu * self.EOS.cp) / (self.EOS.Pr)) * model_outputs[:,:,:,1]

                mu_NN_11 = model_outputs[:,:,:,0]
                mu_NN_12 = model_outputs[:,:,:,1]
                mu_NN_22 = model_outputs[:,:,:,0]

                kappa_NN_1 = model_outputs[:,:,:,2]
                kappa_NN_2 = model_outputs[:,:,:,3]
                
                #print(torch.sum(torch.isnan(mu_NN)) + torch.sum(torch.isnan(mu_NN)))
            

                # masking out outputs 
                # mask = torch.ones_like(mu_NN)
                # mask[:self.grid.xIndOptLeft, :, :] = 0.0
                # mask[self.grid.xIndOptRight:, :, :] = 0.0
                # mask[:, :self.grid.yIndOptTop, :] = 0.0
                # mask[:, self.grid.yIndOptBot:, :] = 0.0

                # mu_NN = mu_NN * mask
                # kappa_NN = kappa_NN * mask


                # Might not work in parallel! Would have to include overlaps into model I think
                mu_NN_11 = self.metrics.expand_overlaps(mu_NN_11)
                mu_NN_12 = self.metrics.expand_overlaps(mu_NN_12)
                mu_NN_22 = self.metrics.expand_overlaps(mu_NN_22)

                kappa_NN_1 = self.metrics.expand_overlaps(kappa_NN_1)
                kappa_NN_2 = self.metrics.expand_overlaps(kappa_NN_2)

                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))             
                #mu =  (mu + 0.0 * mu_NN)#mu + mu_NN #
                # kappa = (kappa  + kappa_NN) #kappa + kappa_NN

                mu_11 =  mu * (1.0 + mu_NN_11)#mu + mu_NN #
                mu_12 =  mu * (1.0 + mu_NN_12)#mu + mu_NN #
                mu_22 =  mu * (1.0 + mu_NN_22)#mu + mu_NN #

                #kappa = (mu * self.EOS.cp / self.EOS.Pr) #* (1.0 + kappa_NN)
                kappa_1 = kappa * (1.0  + kappa_NN_1) #kappa + kappa_NN
                kappa_2 = kappa * (1.0  + kappa_NN_2) #kappa + kappa_NN

                #sourceGauss = model_outputs[:,:,:,0]
                #mu = self.metrics.expand_overlaps(model_outputs[:,:,:,0])

                """ mu = mu_NN
                kappa = kappa_NN """
                model_U = 0.0
            elif self.param.model_type == 'source' or self.param.model_type == 'const':
                model_U = model_outputs
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                
            elif self.param.model_type == 'constant':
                # print('Entering')
                # print(model_outputs)
                mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))
                mu        = mu * model_outputs # self.EOS.base_mu
                kappa     = mu * self.EOS.cp / self.EOS.Pr# model_outputs * kappa
                
                # if flag:
                #     self.constList.append(copy.deepcopy(model_outputs.detach().numpy()))
                #     print(model_outputs)

        else:
            model_outputs = None 
            # Transport coefficients
            mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))

            # anisotropic viscocities
            mu_11 = mu
            mu_12 = mu
            mu_22 = mu

            kappa_1 = kappa
            kappa_2 = kappa

            # # Computing source term
            # x_grid, y_grid = self.grid.get_xy(self.grid.xi_grid, self.grid.eta_grid)
            # X, Y = torch.meshgrid(x_grid, y_grid)
    
            # x0 = 0.25
            # y0 = 0.25
            # sigma_x = 0.1
            # sigma_y = 0.1
            # Amp = 1e0
    
            # sourceGauss = Amp * torch.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2)))
            # sourceGauss = sourceGauss[:,:,None]
            
            # Using only the base value 
            # mu    = self.EOS.base_mu
            # kappa = self.EOS.base_mu / (self.EOS.Pr * self.EOS.Ma**2)
            model_U = 0.0


        # Scalar gradients - extended interior for 2nd derivatives
        dSC_dx = []
        dSC_dy = []
        for isc in range(self.EOS.num_sc):
            _dx, _dy = self.metrics.grad_node( SC[isc], compute_extended=True )[:2]
            dSC_dx.append(_dx)
            dSC_dy.append(_dy)

        # Artificial diffusivity (Kawai & Lele JCP 2008)
        if self.param.artDiss:
            # Exclude from adjoint calculation
            with torch.inference_mode():
                # Strain-rate magnitude
                S = op.strainrate_mag_2D(du_dx ,du_dy ,dv_dx ,dv_dy )

                # Evaulate the artificial transport coefficients
                curl_u = dv_dx - du_dy
                mu_art,beta_art,kappa_art = op.art_diff4_D_2D(self.metrics.full2int(rho),
                                                              drho_dx ,drho_dy ,S,self.tmp_grad,
                                                              div_vel,curl_u,
                                                              self.metrics.full2int(T),
                                                              e,
                                                              self.EOS.get_soundspeed_q(q),
                                                              self.grid,self.metrics,self.param.device)

                mu_eff    = mu + mu_art
                kappa_eff = kappa + kappa_art *self.EOS.Re*self.EOS.Pr
                beta_art  = beta_art *self.EOS.Re
        else:
            mu_eff    = mu
            kappa_eff = kappa
            mu_eff_11 = mu_11
            mu_eff_12 = mu_12
            mu_eff_22 = mu_22
            
            kappa_eff_1 = kappa_1
            kappa_eff_2 = kappa_2
            beta_art  = 0.0

        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                                 self.metrics.ext2int( mu_eff ),
                                                 self.metrics.ext2int( du_dy ),
                                                 None)

        # Viscous stress
        #   Divergence terms are computed on true interior
        div_term_11 = (beta_art - 2.0*mu_eff_11/3.0)*div_vel
        div_term_22 = (beta_art - 2.0*mu_eff_22/3.0)*div_vel

        sigma_11 = 2.0*mu_eff_11*du_dx + div_term_11
        sigma_11_dx,_ = self.metrics.grad_node(sigma_11, extended_input=True, compute_dy=False)[:2]
        
        sigma_22 = 2.0*mu_eff_22*dv_dy + div_term_22
        _,sigma_22_dy = self.metrics.grad_node(sigma_22, extended_input=True, compute_dx=False)[:2]
        
        sigma_12 = mu_eff_12*( du_dy + dv_dx )
        sigma_12_dx,sigma_12_dy = self.metrics.grad_node(sigma_12, extended_input=True)[:2]

        # NOTE: Laplace viscous stress gradients do not currently include
        # artificial diffusivity
        # dmu_eff_dx, dmu_eff_dy = self.metrics.grad_node(mu_eff, extended_input=True)[:2]
        # dk_dx, dk_dy = self.metrics.grad_node(kappa_eff, extended_input=True)[:2]
        # lap_u = self.metrics.lap(u)
        # lap_v = self.metrics.lap(v)
        # lap_T = self.metrics.lap(T)
        if self.param.lapStab:
            stab_strength = 1.0
            lap_rho = self.metrics.lap(rho)
            d2rho_dx2,_ = self.metrics.grad_node(drho_dx, compute_dy=False, extended_input=True)[:2]
            _,d2rho_dy2 = self.metrics.grad_node(drho_dy, compute_dx=False, extended_input=True)[:2]
            stab0 = stab_strength * (lap_rho - d2rho_dx2 - d2rho_dy2) / self.EOS.Re
            lap_E = self.metrics.lap(E)
            d2E_dx2,_ = self.metrics.grad_node(dE_dx, compute_dy=False, extended_input=True)[:2]
            _,d2E_dy2 = self.metrics.grad_node(dE_dy, compute_dx=False, extended_input=True)[:2]
            stab4 = stab_strength * (lap_E - d2E_dx2 - d2E_dy2) / self.EOS.Re / self.EOS.Pr
            if self.param.jproc == 0:
                stab0[:, :3, :] = 0.0
                stab4[:, :3, :] = 0.0
            if self.param.jproc == self.param.npy - 1:
                stab0[:, -4:, :] = 0.0
                stab4[:, -4:, :] = 0.0
        else:
            stab0 = 0.0
            stab4 = 0.0
        #stab0 = 0.0

        ddiv_vel_dx, ddiv_vel_dy = self.metrics.grad_node(div_vel, extended_input=True)[:2]

        # Species diffusion coefficients on extended interior
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.full2ext( SC[isc] )
        DIFF = self.EOS.get_species_diff_coeff(self.metrics.full2ext(T), SC)

        # Species diffusion terms - Fickian diffusion for now
        SC_diff_1_dx = []
        SC_diff_2_dy = []
        rho_ext = self.metrics.full2ext(rho)
        for isc,name in enumerate(self.EOS.sc_names):
            if (name == 'rhoZmix'):
                SC_diff_1_dx.append( 0.0 )
                SC_diff_2_dy.append( 0.0 )

            else:
                SC_diff_1 = rho_ext * DIFF[isc] * dSC_dx[isc]
                _dx, _ = self.metrics.grad_node(SC_diff_1,  extended_input=True, compute_dy=False)[:2]
                SC_diff_1_dx.append(_dx)

                SC_diff_2 = rho_ext * DIFF[isc] * dSC_dy[isc]
                _, _dy = self.metrics.grad_node(SC_diff_2,  extended_input=True, compute_dx=False)[:2]
                SC_diff_2_dy.append(_dy)

        # Heat flux
        q_1 = -kappa_eff_1 * dT_dx
        q_1_dx,_ = self.metrics.grad_node(q_1, extended_input=True, compute_dy=False)[:2]
        
        q_2 = -kappa_eff_2 * dT_dy
        _,q_2_dy = self.metrics.grad_node(q_2, extended_input=True, compute_dx=False)[:2]
        
        # # Coputing stress-divergence terms
        # dsigx_dx, dsigx_dy = self.metrics.grad_node(sigma_11+sigma_12, extended_input=True)[:2]
        # dsigy_dx, dsigy_dy = self.metrics.grad_node(sigma_22+sigma_12, extended_input=True)[:2]
        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        v = self.metrics.full2int( v )
        p = self.metrics.full2int( p )
        E = self.metrics.full2int(E)
        # drho_dx   = self.metrics.ext2int( drho_dx )
        # drho_dy   = self.metrics.ext2int( drho_dy )
        # drhoU_dx  = self.metrics.ext2int( drhoU_dx )
        # drhoU_dy  = self.metrics.ext2int( drhoU_dy )
        # drhoV_dx  = self.metrics.ext2int( drhoV_dx )
        # drhoV_dy  = self.metrics.ext2int( drhoV_dy )
        div_vel   = self.metrics.ext2int( div_vel )
        du_dx     = self.metrics.ext2int( du_dx )
        du_dy     = self.metrics.ext2int( du_dy )
        dv_dx     = self.metrics.ext2int( dv_dx )
        dv_dy     = self.metrics.ext2int( dv_dy )
        # dE_dx     = self.metrics.ext2int( dE_dx )
        # dE_dy     = self.metrics.ext2int( dE_dy )
        sigma_11  = self.metrics.ext2int( sigma_11 )
        sigma_22  = self.metrics.ext2int( sigma_22 )
        sigma_12  = self.metrics.ext2int( sigma_12 )
        if not self.param.upwind:
            drhoUU_dx = self.metrics.ext2int(drhoUU_dx)
            drhoUV_dx = self.metrics.ext2int(drhoUV_dx)
            drhoUV_dy = self.metrics.ext2int(drhoUV_dy)
            drhoVV_dy = self.metrics.ext2int(drhoVV_dy)
            drhoUE_dx = self.metrics.ext2int(drhoUE_dx)
            drhoVE_dy = self.metrics.ext2int(drhoVE_dy)
            dpu_dx    = self.metrics.ext2int(dpu_dx)
            dpv_dy    = self.metrics.ext2int(dpv_dy)
            drhoE_dx = self.metrics.ext2int(drhoE_dx)
            drhoE_dy = self.metrics.ext2int(drhoE_dy)
    
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.ext2int( SC[isc] )
            dSC_dx[isc] = self.metrics.ext2int( dSC_dx[isc] )
            dSC_dy[isc] = self.metrics.ext2int( dSC_dy[isc] )

        mu_eff = self.metrics.ext2int(mu_eff)
        kappa_eff = self.metrics.ext2int(kappa_eff)
        dT_dx = self.metrics.ext2int(dT_dx)
        dT_dy = self.metrics.ext2int(dT_dy)

        if self.param.upwind:
            if self.param.advection_scheme == 'upwind_StegerWarming':
                # Euler fluxes - Steger-Warming
                div_fx, div_fy = self.metrics.Steger_Warming_Fluxes( rho,
                                                                     rhoU,
                                                                     rhoV, 
                                                                     rhoE,
                                                                     self.EOS )
            else:
                # Euler fluxes - Upwind or HLLE
                div_fx, div_fy, div_fz = self.metrics.Euler_Fluxes( rho,
                                                                    rhoU,
                                                                    rhoV,
                                                                    rhoW,
                                                                    rhoE,
                                                                    self.EOS )
            
        
        rho  = self.metrics.full2int( rho )
        rhoU = self.metrics.full2int( rhoU )
        rhoV = self.metrics.full2int( rhoV )
        rhoE = self.metrics.full2int( rhoE )
        for isc in range(self.EOS.num_sc):
            rhoSC[isc] = self.metrics.ext2int( rhoSC[isc] )
            
        # stress_div_x = (
        #     mu_eff * lap_u +
        #     (1/3) * mu_eff * ddiv_vel_dx +
        #     dmu_eff_dx * (du_dx + du_dx) + dmu_eff_dy * (du_dy + dv_dx) -
        #     (2/3) * dmu_eff_dx * div_vel
        # )
        # stress_div_y = (
        #     mu_eff * lap_v +
        #     (1/3) * mu_eff * ddiv_vel_dy +
        #     dmu_eff_dx * (dv_dx + du_dy) + dmu_eff_dy * (dv_dy + dv_dy) -
        #     (2/3) * dmu_eff_dy * div_vel
        # )
        # q_div = - kappa_eff * lap_T - dk_dx * dT_dx - dk_dy * dT_dy
        
        # stress_div_x = dsigx_dx + dsigx_dy
        # stress_div_y = dsigy_dx + dsigy_dy
        # q_div         = q_1_dx + q_2_dy
            
        # Compute RHS terms on true interior
        # Kennedy and Gruber (2008) convective term. Skew-symmetric forms for
        # continuity equation and pressure work.
        # Continuity equation
        if self.param.upwind:
            qdot0 = -(div_fx[0] + div_fy[0]) + stab0
            # qdot0 = 0.0 * (-(div_fx[0] + div_fy[0]) + stab0)
        else:
            u_dot_drho = u * drho_dx + v * drho_dy
            div_mass = drhoU_dx + drhoV_dy
            # qdot0 = -0.5 * (
            #     div_mass +
            #     rho * div_vel +
            #     u_dot_drho) + stab0

            qdot0 = 0.0 * (-0.5 * (
                div_mass +
                rho * div_vel +
                u_dot_drho) + stab0)

        # Momentum equation - x

        # visc = stress_div_x
        visc = sigma_11_dx + sigma_12_dy
        if self.param.upwind:
            qdot1 = -(div_fx[1] + div_fy[1]) + visc #+ sourceGauss
            # qdot1 = visc
        else:
            conv = 0.25 * (
                drhoUU_dx + drhoUV_dy +
                rho * (duu_dx + duv_dy) + u * div_mass + u * drhoU_dx + v * drhoU_dy +
                u * u_dot_drho + rhoU * du_dx + rhoV * du_dy + rhoU * div_vel)
            pres  = -dp_dx * self.EOS.P_fac
            # qdot1 = pres - conv + visc + srcU
            qdot1 = pres - visc
            
        # Momentum equation - y
        # visc = stress_div_y
        visc = sigma_12_dx + sigma_22_dy
        if self.param.upwind:
            qdot2 = -(div_fx[2] + div_fy[2]) + visc
            # qdot2 = 0.0 * visc
        else:
            conv = 0.25 * (
                drhoUV_dx + drhoVV_dy +
                rho * (duv_dx + dvv_dy) + v * div_mass + u * drhoV_dx + v * drhoV_dy +
                v * u_dot_drho + rhoU * dv_dx + rhoV * dv_dy + rhoV * div_vel)
            pres  = -dp_dy * self.EOS.P_fac
            # qdot2 = pres - conv + visc
            qdot2 = 0.0 * (pres - visc)
            

        # Momentum equation - z
        qdot3 = 0.0*qdot2
        
        # Total energy equation
        # visc = (u * stress_div_x + v * stress_div_y +
        #         sigma_11 * du_dx + sigma_12 * du_dy +
        #         sigma_12 * dv_dx + sigma_22 * dv_dy)
        # diff = q_div
        
        visc = ( u*( sigma_11_dx + sigma_12_dy ) +
         v*( sigma_12_dx + sigma_22_dy ) +
         sigma_11*du_dx + sigma_12*du_dy +
         sigma_12*dv_dx + sigma_22*dv_dy )
        
        diff  = q_1_dx + q_2_dy
        
        if self.param.upwind:
            # qdot4 = 0.0 * (-(div_fx[3] + div_fy[3]) + visc - diff) #+ stab4
            qdot4 = (-(div_fx[3] + div_fy[3]) + visc - diff)  #+ stab4
        else:
            conv = 0.25 * (
                drhoUE_dx + drhoVE_dy +
                rho * (duE_dx + dvE_dy) + E * div_mass + u * drhoE_dx + v * drhoE_dy +
                E * u_dot_drho + rhoU * dE_dx + rhoV * dE_dy + rhoE * div_vel)
            pres = 0.5 * (
                dpu_dx + dpv_dy +
                p * div_vel + u * dp_dx + v * dp_dy) * self.EOS.P_fac
            # qdot4 = (visc - conv - pres - diff + u * srcU + stab4)
            qdot4 = (visc - pres)

        # Species equations
        srcSC  = self.EOS.get_species_production_rates(rho, self.metrics.full2int(T), SC)
        qdotSC = []
        for isc in range(self.EOS.num_sc):
            conv = ( rhoU * dSC_dx[isc] +
                     rhoV * dSC_dy[isc] +
                     SC[isc] * (drhoU_dx + drhoV_dy) )
            diff = SC_diff_1_dx[isc] + SC_diff_2_dy[isc]
            qdot = diff - conv + srcSC[isc]
            qdotSC.append(qdot)
        

        # Closure model
        # if self.param.Use_Model:
        #     input_dict = q#{'u':u, 'v':v,  'p':p, 'du_dy':du_dy}

        #     qdot_dict = {'qdot0':qdot0, 'qdot1':qdot1, 'qdot2':qdot2,
        #                  'qdot3':qdot3, 'qdot4':qdot4}
            
        #     model_outputs = self.param.apply_model(self.param.model, input_dict, self.grid,
        #                                            self.metrics, self.param, self.EOS)
        # else:
        #     model_outputs = None

            
        # Boundary conditions
        # -----------------------------------------------------------------------
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            
            # dirichlet on rho
            qdot0[:,0,:] = 0.0

            # dirichlet on U, V
            # qdot1[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= u[:, 0, :]

            # qdot2[:,0,:] = qdot0[:, 0, :]
            # qdot1[:,0,:] *= v[:, 0, :]
            
            # dirichlet on rhoU, rhoV
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
            
            # Isothermal wall
            if not self.param.adiabatic:
                qdot4[:,0,:] = qdot0[:, 0, :]
                qdot4[:,0,:] *= E[:, 0, :]
            
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5+isc,:,None,None] - rhoSC[isc] )

        # -----------------------------------------------------------------------
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
            
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            # Farfield
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5+isc,:,None,None] - rhoSC[isc] )

        if (self.grid.BC_eta_top=='supersonic' and self.param.jproc==self.param.npy-1):
                # Supersonic outflow boundary - eta top
                qdot0[:,-1,:] = qdot0[:,-2,:]
                qdot1[:,-1,:] = qdot1[:,-2,:]
                qdot2[:,-1,:] = qdot2[:,-2,:]
                qdot4[:,-1,:] = qdot4[:,-2,:]

                # qdot0[:,-1,:] = 0.0
                # qdot1[:,-1,:] = 0.0
                # qdot2[:,-1,:] = 0.0
                # qdot4[:,-1,:] = 0.0

        # -----------------------------------------------------------------------
        # Xi left boundary
        if (hasattr(self.grid, "BC_xi_left") and self.grid.BC_xi_left=='supersonic' and self.param.iproc==0):
            # Supersonic outflow boundary
            qdot0[0,:,:] = qdot0[1,:,:]
            qdot1[0,:,:] = qdot1[1,:,:]
            qdot2[0,:,:] = qdot2[1,:,:]
            qdot4[0,:,:] = qdot4[1,:,:]
                
        elif (not self.grid.periodic_xi and self.param.iproc==0):
            # Farfield
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):
                
                if (not self.grid.BC_xi_left  == 'symmetric'):
                    qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho )
                    qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU )
                    qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV )
                    qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE )
                    # Species
                    for isc in range(self.EOS.num_sc):
                        qdotSC[isc] += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5+isc,None,:,None] - rhoSC[isc] )
                        
                else:
                    

                    # Symmetric boundary 
                    qdot0[0:self.grid.wallPoint,0,:] = qdot0[0:self.grid.wallPoint,1,:]  ## FIX! HARDCODED
                    qdot1[0:self.grid.wallPoint,0,:] = qdot1[0:self.grid.wallPoint,1,:]
                    #qdot2[0:51,0,:] = qdot2[0:51,1,:]
                    qdot4[0:self.grid.wallPoint,0,:] = qdot4[0:self.grid.wallPoint,1,:]
                    
                                
        # -----------------------------------------------------------------------
        # Xi right boundary
        if (hasattr(self.grid, "BC_xi_right") and self.grid.BC_xi_right=='supersonic' and
            self.param.iproc==self.param.npx-1):
            # Supersonic outflow boundary
            qdot0[-1,:,:] = qdot0[-2,:,:]
            qdot1[-1,:,:] = qdot1[-2,:,:]
            qdot2[-1,:,:] = qdot2[-2,:,:]
            qdot4[-1,:,:] = qdot4[-2,:,:]

            # qdot0[-1,:,:] = 0.0
            # qdot1[-1,:,:] = 0.0
            # qdot2[-1,:,:] = 0.0
            # qdot4[-1,:,:] = 0.0


        elif (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            # Farfield
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho )
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5+isc,None,:,None] - rhoSC[isc] )

        # Slip wall boundary 
        # Slip wall boundary 
        slipWall = True
        if slipWall:
            
            
            
            
            # print('Entering slip')
            # Under-relaxation factor
            facUR_U = 1.0
            facUR_T = 1.0

            # Updating the variables
            rho_upd  = rho.detach() + self.param.dt * qdot0.clone()
            rhoU_upd = rhoU.detach() + self.param.dt * qdot1.clone()
            rhoV_upd = rhoV.detach() + self.param.dt * qdot2.clone()
            rhoE_upd = rhoE.detach() + self.param.dt * qdot4.clone()
            q_upd    = q.clone()#torch.zeros(4, self.grid.Nx1, self.grid.Nx2, 1).to(self.param.device)

            q_upd[0][5:-5,5:-5,:] = rho_upd
            q_upd[1][5:-5,5:-5,:] = rhoU_upd
            q_upd[2][5:-5,5:-5,:] = rhoV_upd
            q_upd[3][5:-5,5:-5,:] = rhoE_upd   

            # Recomputing primitives
            u_upd = rhoU_upd/rho_upd
            v_upd = rhoV_upd/rho_upd
            E_upd = rhoE_upd/rho_upd
            

            T_upd, p_upd, e_upd = self.EOS.get_TPE_tensor(q_upd, SC)
            
            T_upd = self.metrics.full2int(T_upd)

            # indexes
            # wm_index_1 = 2
            # wm_index_2 = 3
            # wm_index_3 = 4
            # wm_index_4 = 5
            
            # extracting wall model outputs and reshpaing 
            
            if False:#self.param.Use_Model:
                # wmo_1 = (model_outputs[:,:,:,wm_index_1])
                # wmo_2 = (model_outputs[:,:,:,wm_index_2])
                # wmo_3 = (model_outputs[:,:,:,wm_index_3])
                # wmo_4 = (model_outputs[:,:,:,wm_index_4])
                
                model_inputs = torch.stack(((T_upd/self.EOS.T0).squeeze(), (u_upd/self.EOS.U0).squeeze()),dim=2)
                model_outputs = self.param.model.forward(model_inputs)
    
    
                # Creating masks 
                mask1 = torch.ones_like(T_upd)
                #mask1[self.grid.wallPoint:, 0, :] = 0.
                
                mask2 = torch.zeros_like(T_upd)
                #mask2[self.grid.wallPoint:, 0, :] = 1.
    
                # Defining the accomodation coefficients
                sig = 1.0 * mask1 + model_outputs[:,:,0].unsqueeze(dim=2) * mask2
                alph = 1.0 * mask1 + model_outputs[:,:,1].unsqueeze(dim=2) * mask2
                A    = (2.0/torch.pi) ** 0.5 * mask1 + model_outputs[:,:,2].unsqueeze(dim=2) * ((2.0/torch.pi) ** 0.5 + 1.) * mask2
                Twall = self.EOS.T0 * mask1 + model_outputs[:,:,3].unsqueeze(dim=2) * (1. + 300.) * mask2
                Umul = 1.0 * mask1 + model_outputs[:,:,4].unsqueeze(dim=2) * mask2
                Tmul = 1.0 * mask1 + model_outputs[:,:,5].unsqueeze(dim=2) * mask2
                
                
            else:
                sig = 1.0 
                alph = 1.0
                A    = (2.0/torch.pi) ** 0.5
                Twall = 300#292#self.EOS.T0
                Umul = 1.0
                Tmul = 1.0

            
            # Computing mean free paths
            mu, kappa = self.EOS.get_mu_kappa(T_upd)
            
            mfp    = (mu/rho_upd) * (torch.pi/(2 * self.EOS.Rgas * T_upd)) ** 0.5
            mfp_T  = (2/(self.EOS.gamma + 1)) * (kappa / (rho_upd * self.EOS.cv)) * (torch.pi / (2 * self.EOS.Rgas * T_upd)) ** 0.5

            # mfp = self.get_mfp()

            # Computing slip-velocity
            # Vs = facUR_U * A * (2-sig)/sig * mfp[grid.wallPoint:, 0, 0] * du_dy[grid.wallPoint:, 0, 0] # (u[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # u[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]  = Vs
            u_fac = Umul * A * (2-sig)/sig * mfp * (1 / self.grid.Y[self.grid.wallPoint+1, 1])
            #u_fac = facUR_U * A * (2-sig)/sig * 1 * (1 / self.grid.Y[self.grid.wallPoint:, 1])
            
            # cloning and shifting u values
            u_upd_shift = u_upd.clone()
            u_upd_shift[:, [0,1], :] = u_upd_shift[:, [1,0], :]
            
            Vs    = (u_fac / (1.0 + u_fac)) * u_upd_shift

            mask = torch.ones_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            u_upd = u_upd * mask
            
            mask = torch.zeros_like(u_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            Vs = Vs * mask
            
            u_upd  += Vs
            
            # u[self.grid.wallPoint:, 0, 0]  = Vs
            
            # Computing temperature jump 
            # delT = facUR_T * (2 - alph) / alph * mfp_T[grid.wallPoint:, 0, 0] * dT_dy[grid.wallPoint:, 0, 0] # (T[5:-5, 5:-5, :][grid.wallPoint:, 1, 0] )
            # T[5:-5, 5:-5, :][grid.wallPoint:, 0, 0]    = EOS.T0 + delT
            T_fac  = Tmul * ((2 - alph) / alph) * mfp_T * (1 / self.grid.Y[self.grid.wallPoint+1, 1])

            # Creating mask
            mask = torch.ones_like(T_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            T_upd = T_upd * mask
            
            mask = torch.zeros_like(self.metrics.full2int(T))
            mask[self.grid.wallPoint:, 0, 0] = 1.0
            
            T_shifted = self.metrics.full2int(T).clone()
            T_shifted[:, [0,1], :] = T_shifted[:, [1,0], :]
            T_masked =  T_shifted * mask
            T_wall   = Twall * mask
            T_fac     = T_fac * mask

            #T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 0, 0] += (T_fac * T_upd[5:-5, 5:-5, :][self.grid.wallPoint:, 1, 0] + Twall) / (1.0 + T_fac)
            T_upd += (T_fac * T_masked + T_wall) / (1.0 + T_fac)

            e_upd = self.EOS.get_internal_energy_TY(T_upd)

            # Creating a mask
            mask = torch.ones_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 0.0

            rhoU_upd = rhoU_upd * mask
            rhoE_upd = rhoE_upd * mask

            rhoU_upd += Vs * rho_upd
            rhoE_upd += (rho_upd * (e_upd + 1.0 / 2.0 * (u_upd**2 + v_upd**2) ))

            qdot1 = qdot1 * mask
            qdot4 = qdot4 * mask
            
            mask = torch.zeros_like(rhoU_upd)
            mask[self.grid.wallPoint:, 0, 0] = 1.0

            qdot1 += (rhoU_upd * mask - rhoU * mask) / self.param.dt
            qdot4 += (rhoE_upd * mask - rhoE * mask) / self.param.dt 



        if self.jvp:
            # Padding to maintain dimensional consistency 
            R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
            
            qdot_list = [qdot0,qdot1,qdot2,qdot3,qdot4] + qdotSC
            qdot_stack = torch.stack((R(qdot_list[0]), R(qdot_list[1]), R(qdot_list[2]), R(qdot_list[4])), dim=0)
            
            if flag:
                
                xLeftCut  =  self.grid.xIndOptLeft  + 5 
                xRightCut =  self.grid.xIndOptRight - 5 
                yTopCut   =  self.grid.yIndOptTop   - 5 
                yBotCut   =  self.grid.yIndOptBot   + 5 

                # Eta-direction
                qdot_stack[:, :xLeftCut, :, :] = q[:, :xLeftCut, :, :]
                qdot_stack[:, xRightCut:, :, :] = q[:, xRightCut:, :, :]
                
                # Xi-direction
                qdot_stack[:, :yBotCut, :, :] = q[:, :yBotCut, :, :]
                qdot_stack[:, yTopCut:, :, :] = q[:, yTopCut:, :, :]
                
    
            return qdot_stack

        else:
            return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4),dim=0), model_outputs
          
    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 2D - TESTING UPWIND
    # --------------------------------------------------------------
    #@profile
    def NS_2D_Euler(self, q):
        # Extract conserved variables
        if self.jvp:
            rho  = q[0]
            rhoU = q[1]
            rhoV = q[2]
            rhoW = q[2]*0.0
            rhoE = q[3]
        else:
            rho  = q['rho'].var
            rhoU = q['rhoU'].var
            rhoV = q['rhoV'].var
            rhoW = q['rhoW'].var
            rhoE = q['rhoE'].var

        # Compute primitives including overlaps
        u = rhoU/rho
        v = rhoV/rho
        E = rhoE/rho
            
        # Thermochemistry
        if self.jvp:
            T, p, e = self.EOS.get_TPE_tensor(q)
        else:
            T, p, e = self.EOS.get_TPE(q)
            
        # Euler fluxes - Steger-Warming
        div_fx, div_fy = self.metrics.Steger_Warming_Fluxes( rho,
                                                             rhoU,
                                                             rhoV, 
                                                             rhoE,
                                                             self.EOS )
        
        rho  = self.metrics.full2int( rho )
        rhoU = self.metrics.full2int( rhoU )
        rhoV = self.metrics.full2int( rhoV )
        rhoE = self.metrics.full2int( rhoE )
        E = self.metrics.full2int( E )

        # Continuity
        qdot0 = -(div_fx[0] + div_fy[0])
            
        # Momentum equation - x
        qdot1 = -(div_fx[1] + div_fy[1])
            
        # Momentum equation - y
        qdot2 = -(div_fx[2] + div_fy[2])

        # Momentum equation - z
        qdot3 = 0.0*qdot2
        
        # Total energy equation
        qdot4 = -(div_fx[3] + div_fy[3])

            
        # Boundary conditions
        # -----------------------------------------------------------------------
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
            qdot4[:,0,:] = qdot0[:, 0, :]
            qdot4[:,0,:] *= E[:, 0, :]
            
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5+isc,:,None,None] - rhoSC[isc] )

        # -----------------------------------------------------------------------
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
            
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            # Farfield
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho )
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5+isc,:,None,None] - rhoSC[isc] )

        if (self.grid.BC_eta_top=='supersonic' and self.param.jproc==self.param.npy-1):
                # Supersonic outflow boundary - eta top
                qdot0[:,-1,:] = qdot0[:,-2,:]
                qdot1[:,-1,:] = qdot1[:,-2,:]
                qdot2[:,-1,:] = qdot2[:,-2,:]
                qdot4[:,-1,:] = qdot4[:,-2,:]

        # -----------------------------------------------------------------------
        # Xi left boundary
        if (hasattr(self.grid, "BC_xi_left") and self.grid.BC_xi_left=='supersonic' and self.param.iproc==0):
            # Supersonic outflow boundary
            qdot0[0,:,:] = qdot0[1,:,:]
            qdot1[0,:,:] = qdot1[1,:,:]
            qdot2[0,:,:] = qdot2[1,:,:]
            qdot4[0,:,:] = qdot4[1,:,:]
                
        elif (not self.grid.periodic_xi and self.param.iproc==0):
            # Farfield
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):
                qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho )
                qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU )
                qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV )
                qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE )
                # Species
                for isc in range(self.EOS.num_sc):
                    qdotSC[isc] += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5+isc,None,:,None] - rhoSC[isc] )

                # Symmetric boundary 
                #qdot0[0:51,0,:] = qdot0[0:51,1,:]  ## FIX! HARDCODED
                #qdot1[0:51,0,:] = qdot1[0:51,1,:]
                ##qdot2[0:51,0,:] = qdot2[0:51,1,:]
                #qdot4[0:51,0,:] = qdot4[0:51,1,:]
            
        # -----------------------------------------------------------------------
        # Xi right boundary
        if (hasattr(self.grid, "BC_xi_right") and self.grid.BC_xi_right=='supersonic' and
            self.param.iproc==self.param.npx-1):
            # Supersonic outflow boundary
            qdot0[-1,:,:] = qdot0[-2,:,:]
            qdot1[-1,:,:] = qdot1[-2,:,:]
            qdot2[-1,:,:] = qdot2[-2,:,:]
            qdot4[-1,:,:] = qdot4[-2,:,:]
            
        elif (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            # Farfield
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho )
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5+isc,None,:,None] - rhoSC[isc] )

        if self.jvp:
            # Padding to maintain dimensional consistency 
            R = lambda L: torch.nn.functional.pad(L,(0,0,5,5,5,5),"constant",0)
            
            qdot_list = [qdot0,qdot1,qdot2,qdot3,qdot4] + qdotSC
            return torch.stack((R(qdot_list[0]), R(qdot_list[1]), R(qdot_list[2]), R(qdot_list[4])), dim=0)

        else:
            return torch.stack((qdot0,qdot1,qdot2,qdot3,qdot4),dim=0), None



    # --------------------------------------------------------------
    # Navier-Stokes RHS function - 3D
    #   Collocated visc terms - conservative form
    #   2nd derivatives obtained by repeated application of 1st derivatives
    # --------------------------------------------------------------
    def NS_3D(self, q):
        # Extract conserved variables
        rho  = q['rho']
        rhoU = q['rhoU']
        rhoV = q['rhoV']
        rhoW = q['rhoW']
        rhoE = q['rhoE']
        rhoSC = []
        for name in self.EOS.sc_names:
            rhoSC.append(q[name])

        # Compute primitives including overlaps
        u = rhoU.var/rho.var
        v = rhoV.var/rho.var
        w = rhoW.var/rho.var
        E = rhoE.var/rho.var
        SC = []
        for isc in range(self.EOS.num_sc):
            SC.append(rhoSC[isc].var / rho.var)
            
        # Thermochemistry
        T, p, e = self.EOS.get_TPE(q, SC)

        # Compute 1st derivatives - true interior
        drhoUU_dx,_,_ = self.metrics.grad_node(rhoU.var*u, compute_dy=False, compute_dz=False)
        _,drhoVV_dy,_ = self.metrics.grad_node(rhoV.var*v, compute_dx=False, compute_dz=False)
        _,_,drhoWW_dz = self.metrics.grad_node(rhoW.var*w, compute_dx=False, compute_dy=False)
        drhoUV_dx, drhoUV_dy, _ = self.metrics.grad_node(rhoU.var*v, compute_dz=False)
        drhoUW_dx, _, drhoUW_dz = self.metrics.grad_node(rhoU.var*w, compute_dy=False)
        _, drhoVW_dy, drhoVW_dz = self.metrics.grad_node(rhoV.var*w, compute_dx=False)

        drhoUE_dx,_,_ = self.metrics.grad_node(rhoU.var*E, compute_dy=False, compute_dz=False)
        _,drhoVE_dy,_ = self.metrics.grad_node(rhoV.var*E, compute_dx=False, compute_dz=False)
        _,_,drhoWE_dz = self.metrics.grad_node(rhoW.var*E, compute_dx=False, compute_dy=False)

        duu_dx, _, _ = self.metrics.grad_node(u*u, compute_dy=False, compute_dz=False)
        _, dvv_dy, _ = self.metrics.grad_node(v*v, compute_dx=False, compute_dz=False)
        _, _, dww_dz = self.metrics.grad_node(w*w, compute_dx=False, compute_dy=False)
        duv_dx, duv_dy, _ = self.metrics.grad_node(u*v, compute_dz=False)
        duw_dx, _, duw_dz = self.metrics.grad_node(u*w, compute_dy=False)
        _, dvw_dy, dvw_dz = self.metrics.grad_node(v*w, compute_dx=False)

        duE_dx, _, _ = self.metrics.grad_node(u*E, compute_dy=False, compute_dz=False)
        _, dvE_dy, _ = self.metrics.grad_node(v*E, compute_dx=False, compute_dz=False)
        _, _, dwE_dz = self.metrics.grad_node(w*E, compute_dx=False, compute_dy=False)

        dpu_dx, _, _ = self.metrics.grad_node(p*u, compute_dy=False, compute_dz=False)
        _, dpv_dy, _ = self.metrics.grad_node(p*v, compute_dx=False, compute_dz=False)
        _, _, dpw_dz = self.metrics.grad_node(p*w, compute_dx=False, compute_dy=False)

        dp_dx,dp_dy,dp_dz          = self.metrics.grad_node( p )
        dE_dx,dE_dy,dE_dz          = self.metrics.grad_node( E, compute_extended=True )

        # Compute 1st derivatives - extended interior for 2nd derivatives
        drho_dx,drho_dy,drho_dz     = self.metrics.grad_node(rho.var,  compute_extended=True )
        drhoU_dx,drhoU_dy,drhoU_dz  = self.metrics.grad_node(rhoU.var, compute_extended=True )
        drhoV_dx,drhoV_dy,drhoV_dz  = self.metrics.grad_node(rhoV.var, compute_extended=True )
        drhoW_dx,drhoW_dy,drhoW_dz  = self.metrics.grad_node(rhoW.var, compute_extended=True )
        dT_dx,dT_dy,dT_dz           = self.metrics.grad_node(T, compute_extended=True)
        drhoE_dx, drhoE_dy, drhoE_dz = self.metrics.grad_node(rhoE.var, compute_extended=True)

        # Gradients - extended interior
        du_dx, du_dy, du_dz = self.metrics.grad_node(u, compute_extended=True)
        dv_dx, dv_dy, dv_dz = self.metrics.grad_node(v, compute_extended=True)
        dw_dx, dw_dy, dw_dz = self.metrics.grad_node(w, compute_extended=True)

        # Velocity and mass divergence
        div_vel = du_dx + dv_dy + dw_dz
    
        # Transport coefficients
        mu, kappa = self.EOS.get_mu_kappa(self.metrics.full2ext(T))

        # Scalar gradients - extended interior for 2nd derivatives
        dSC_dx = []
        dSC_dy = []
        dSC_dz = []
        for isc in range(self.EOS.num_sc):
            _dx, _dy, _dz = self.metrics.grad_node( SC[isc], compute_extended=True )
            dSC_dx.append(_dx)
            dSC_dy.append(_dy)
            dSC_dz.append(_dz)

        # Artificial diffusivity (Kawai & Lele JCP 2008)
        if self.param.artDiss:
            # Exclude from adjoint calculation
            with torch.inference_mode():
                # Strain-rate magnitude
                S = op.strainrate_mag_3D( du_dx,du_dy,du_dz,
                                          dv_dx,dv_dy,dv_dz,
                                          dw_dx,dw_dy,dw_dz )

                # Magnitude of the curl of velocity
                curl_u = op.vec_mag_3D( dw_dy - dv_dz,
                                        du_dz - dw_dx,
                                        dv_dx - du_dy )

                # Get the coefficients
                mu_art,beta_art,kappa_art = op.art_diff4_D_3D(rho,drho_dx,drho_dy,drho_dz,S,self.tmp_grad,
                                                              div_vel,curl_u,T,e,
                                                              self.EOS.get_soundspeed_q(q),
                                                              self.grid,self.metrics,self.param.device)

                mu_eff    = mu + mu_art *self.EOS.Re ## BE CAREFUL HERE - FOR NONDIM ONLY
                kappa_eff = kappa + kappa_art *self.EOS.Re*self.EOS.Pr
                beta_art  = beta_art *self.EOS.Re
        else:
            mu_eff    = mu
            kappa_eff = kappa
            beta_art  = 0.0

        
        # Body forces
        srcU,srcW = self.param.bodyforce.compute(q, self.param.dt,
                                                 self.metrics.ext2int( mu_eff ),
                                                 self.metrics.ext2int( du_dy ),
                                                 self.metrics.ext2int( dw_dy ))

        # Viscous stresses
        div_term = (beta_art - 2.0*mu_eff/3.0)*div_vel
        sigma_colon_g = (
                (2.0*mu_eff * du_dx + div_term) * du_dx +   # sigma_11 * du_dx
                (2.0*mu_eff * dv_dy + div_term) * dv_dy +   # sigma_22 * dv_dy
                (2.0*mu_eff * dw_dz + div_term) * dw_dz +   # sigma_33 * dw_dz
                mu_eff * (du_dy + dv_dx)**2 +               # sigma_12 * (du_dy + dv_dx)
                mu_eff * (du_dz + dw_dx)**2 +               # sigma_13 * (du_dz + dw_dx)
                mu_eff * (dv_dz + dw_dy)**2                 # sigma_23 * (dv_dz + dw_dy)
        )

        # NOTE: Laplace viscous stress gradients do not currently include
        # artificial diffusivity
        ddiv_vel_dx, ddiv_vel_dy, ddiv_vel_dz = self.metrics.grad_node(div_vel, extended_input=True)
        dmu_eff_dx, dmu_eff_dy, dmu_eff_dz = self.metrics.grad_node(mu_eff, extended_input=True)
        dk_dx, dk_dy, dk_dz = self.metrics.grad_node(kappa_eff, extended_input=True)
        lap_u = self.metrics.lap(u)
        lap_v = self.metrics.lap(v)
        lap_w = self.metrics.lap(w)
        lap_T = self.metrics.lap(T)
        if self.param.lapStab:
            stab_strength = 0.1 # Should be in range [0, 1] ish.
            # lap_rho = self.metrics.lap(rho.var)
            # d2rho_dx2,_,_ = self.metrics.grad_node(drho_dx, compute_dy=False, compute_dz=False, extended_input=True)
            # _,d2rho_dy2,_ = self.metrics.grad_node(drho_dy, compute_dx=False, compute_dz=False, extended_input=True)
            # _,_,d2rho_dz2 = self.metrics.grad_node(drho_dz, compute_dx=False, compute_dy=False, extended_input=True)
            # stab0 = stab_strength * (lap_rho - d2rho_dx2 - d2rho_dy2 - d2rho_dz2) / self.EOS.Re
            lap_E = self.metrics.lap(E)
            d2E_dx2,_,_ = self.metrics.grad_node(dE_dx, compute_dy=False, compute_dz=False, extended_input=True)
            _,d2E_dy2,_ = self.metrics.grad_node(dE_dy, compute_dx=False, compute_dz=False, extended_input=True)
            _,_,d2E_dz2 = self.metrics.grad_node(dE_dz, compute_dx=False, compute_dy=False, extended_input=True)
            stab4 =  stab_strength * (lap_E - d2E_dx2 - d2E_dy2 - d2E_dz2) / self.EOS.Re / self.EOS.Pr
            if self.param.jproc == 0:
                # stab0[:, :3, :] = 0.0
                stab4[:, :3, :] = 0.0
            if self.param.jproc == self.param.npy - 1:
                # stab0[:, -4:, :] = 0.0
                stab4[:, -4:, :] = 0.0
        else:
            # stab0 = 0.0
            stab4 = 0.0
        stab0 = 0.0
        
        # Species diffusion coefficients on extended interior
        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.full2ext( SC[isc] )
        DIFF = self.EOS.get_species_diff_coeff(self.metrics.full2ext(T), SC)

        # Species diffusion terms - Fickian diffusion for now
        SC_diff_1_dx = []
        SC_diff_2_dy = []
        SC_diff_3_dz = []
        rho_ext = self.metrics.full2ext(rho.var)
        for isc,name in enumerate(self.EOS.sc_names):
            if (name == 'rhoZmix'):
                SC_diff_1_dx.append( 0.0 )
                SC_diff_2_dy.append( 0.0 )
                SC_diff_2_dz.append( 0.0 )
            else:
                SC_diff_1 = rho_ext * DIFF[isc] * dSC_dx[isc]
                _dx, _, _ = self.metrics.grad_node(SC_diff_1,  extended_input=True, compute_dy=False, compute_dz=False)
                SC_diff_1_dx.append(_dx)

                SC_diff_2 = rho_ext * DIFF[isc] * dSC_dy[isc]
                _, _dy, _ = self.metrics.grad_node(SC_diff_2,  extended_input=True, compute_dx=False, compute_dz=False)
                SC_diff_2_dy.append(_dy)

                SC_diff_3 = rho_ext * DIFF[isc] * dSC_dz[isc]
                _, _, _dz = self.metrics.grad_node(SC_diff_2,  extended_input=True, compute_dx=False, compute_dy=False)
                SC_diff_3_dz.append(_dz)
        
        
        # Truncate extended interior to true interior
        u = self.metrics.full2int( u )
        v = self.metrics.full2int( v )
        w = self.metrics.full2int( w )
        p = self.metrics.full2int( p )
        E = self.metrics.full2int(E)
        drho_dx  = self.metrics.ext2int( drho_dx )
        drho_dy  = self.metrics.ext2int( drho_dy )
        drho_dz  = self.metrics.ext2int( drho_dz )
        drhoU_dx = self.metrics.ext2int( drhoU_dx )
        drhoU_dy = self.metrics.ext2int( drhoU_dy )
        drhoU_dz = self.metrics.ext2int( drhoU_dz )
        drhoV_dx = self.metrics.ext2int( drhoV_dx )
        drhoV_dy = self.metrics.ext2int( drhoV_dy )
        drhoV_dz = self.metrics.ext2int( drhoV_dz )
        drhoW_dx = self.metrics.ext2int( drhoW_dx )
        drhoW_dy = self.metrics.ext2int( drhoW_dy )
        drhoW_dz = self.metrics.ext2int( drhoW_dz )
        drhoE_dx = self.metrics.ext2int( drhoE_dx )
        drhoE_dy = self.metrics.ext2int( drhoE_dy )
        drhoE_dz = self.metrics.ext2int( drhoE_dz )
        div_vel  = self.metrics.ext2int( div_vel )
        du_dx    = self.metrics.ext2int( du_dx )
        du_dy    = self.metrics.ext2int( du_dy )
        du_dz    = self.metrics.ext2int( du_dz )
        dv_dx    = self.metrics.ext2int( dv_dx )
        dv_dy    = self.metrics.ext2int( dv_dy )
        dv_dz    = self.metrics.ext2int( dv_dz )
        dw_dx    = self.metrics.ext2int( dw_dx )
        dw_dy    = self.metrics.ext2int( dw_dy )
        dw_dz    = self.metrics.ext2int( dw_dz )
        mu_eff   = self.metrics.ext2int( mu_eff )
        kappa_eff   = self.metrics.ext2int( kappa_eff )
        dT_dx = self.metrics.ext2int( dT_dx )
        dT_dy = self.metrics.ext2int( dT_dy )
        dT_dz = self.metrics.ext2int( dT_dz )
        dE_dx = self.metrics.ext2int( dE_dx )
        dE_dy = self.metrics.ext2int( dE_dy )
        dE_dz = self.metrics.ext2int( dE_dz )
        sigma_colon_g = self.metrics.ext2int( sigma_colon_g )

        for isc in range(self.EOS.num_sc):
            SC[isc] = self.metrics.ext2int( SC[isc] )
            dSC_dx[isc] = self.metrics.ext2int( dSC_dx[isc] )
            dSC_dy[isc] = self.metrics.ext2int( dSC_dy[isc] )
            dSC_dz[isc] = self.metrics.ext2int( dSC_dz[isc] )

        stress_div_x = (
            mu_eff * lap_u +
            (1/3) * mu_eff * ddiv_vel_dx +
            dmu_eff_dx * (du_dx + du_dx) + dmu_eff_dy * (du_dy + dv_dx) + dmu_eff_dz * (du_dz + dw_dx) -
            (2/3) * dmu_eff_dx * div_vel
        )
        stress_div_y = (
            mu_eff * lap_v +
            (1/3) * mu_eff * ddiv_vel_dy +
            dmu_eff_dx * (dv_dx + du_dy) + dmu_eff_dy * (dv_dy + dv_dy) + dmu_eff_dz * (dv_dz + dw_dy) -
            (2/3) * dmu_eff_dy * div_vel
        )
        stress_div_z = (
            mu_eff * lap_w +
            (1/3) * mu_eff * ddiv_vel_dz +
            dmu_eff_dx * (dw_dx + du_dz) + dmu_eff_dy * (dw_dy + dv_dz) + dmu_eff_dz * (dw_dz + dw_dz) -
            (2/3) * dmu_eff_dz * div_vel
        )
        q_div = - (kappa_eff * lap_T + dk_dx * dT_dx + dk_dy * dT_dy + dk_dz * dT_dz)

        # Compute RHS terms on true interior
        # Kennedy and gruber (2008) convective term; KE conserving variant with
        # alpha = beta = 0.25 (Kuya). Skew-symmetric forms for continuity
        # equation and pressure work.
        div_mass = drhoU_dx + drhoV_dy + drhoW_dz
        u_dot_drho = u * drho_dx + v * drho_dy + w * drho_dz

        # Continuity equation
        qdot0 = -0.5 * (
            div_mass +
            rho.interior() * div_vel + u_dot_drho) + stab0
        
        # Momentum equation - x
        conv  = 0.25 * (
            drhoUU_dx + drhoUV_dy + drhoUW_dz +
            rho.interior() * (duu_dx + duv_dy + duw_dz) + u * div_mass + u * drhoU_dx + v * drhoU_dy + w * drhoU_dz +
            u * u_dot_drho + rhoU.interior() * du_dx + rhoV.interior() * du_dy  + rhoW.interior() * du_dz + rhoU.interior() * div_vel
        )
        pres  = -dp_dx * self.EOS.P_fac
        visc  = stress_div_x
        qdot1 = pres - conv + visc + srcU
        
        # Momentum equation - y
        conv  = 0.25 * (
            drhoUV_dx + drhoVV_dy + drhoVW_dz +
            rho.interior() * (duv_dx + dvv_dy + dvw_dz) + v * div_mass + u * drhoV_dx + v * drhoV_dy + w * drhoV_dz +
            v * u_dot_drho + rhoU.interior() * dv_dx + rhoV.interior() * dv_dy + rhoW.interior() * dv_dz + rhoV.interior() * div_vel
        )
        pres  = -dp_dy * self.EOS.P_fac
        visc  = stress_div_y
        qdot2 = pres - conv + visc

        # Momentum equation -z
        conv = 0.25 * (
            drhoUW_dx + drhoVW_dy + drhoWW_dz +
            rho.interior() * (duw_dx + dvw_dy + dww_dz) + w * div_mass + u * drhoW_dx + v * drhoW_dy + w * drhoW_dz +
            w * u_dot_drho + rhoU.interior() * dw_dx + rhoV.interior() * dw_dy + rhoW.interior() * dw_dz + rhoW.interior() * div_vel
        )
        pres  = -dp_dz * self.EOS.P_fac
        visc  = stress_div_z
        qdot3 = pres - conv + visc + srcW
        
        # Total energy equation
        conv = 0.25 * (
            drhoUE_dx + drhoVE_dy + drhoWE_dz +
            rho.interior() * (duE_dx + dvE_dy + dwE_dz) + E * div_mass + u * drhoE_dx + v * drhoE_dy + w * drhoE_dz +
            E * u_dot_drho + rhoU.interior() * dE_dx + rhoV.interior() * dE_dy + rhoW.interior() * dE_dz + rhoE.interior() * div_vel
        )
        pres = 0.5 * self.EOS.P_fac *(
            dpu_dx + dpv_dy + dpw_dz +
            p * div_vel + u * dp_dx + v * dp_dy + w * dp_dz)
        visc = (
            u * stress_div_x + v * stress_div_y + w * stress_div_z +
            sigma_colon_g
        )
        diff  = q_div
        src_pg = w*srcW + u*srcU
        qdot4 = visc - conv - pres - diff + src_pg + stab4

        # Species equations
        srcSC  = self.EOS.get_species_production_rates(rho.interior(), self.metrics.full2int(T), SC)
        qdotSC = []
        for isc in range(self.EOS.num_sc):
            conv = ( rhoU.interior() * dSC_dx[isc] +
                     rhoV.interior() * dSC_dy[isc] +
                     rhoW.interior() * dSC_dz[isc] +
                     SC[isc] * (drhoU_dx + drhoV_dy + drhoW_dz) )
            diff = SC_diff_1_dx[isc] + SC_diff_2_dy[isc] + SC_diff_3_dz[isc]
            qdot = diff - conv + srcSC[isc]
            qdotSC.append(qdot)

        # Closure model needs putting before restrict to interior. Copy from
        # airfoil branch when needed.

        # Boundary conditions
        # Dirichlet BCs on -/+ eta
        # Eta bottom boundary (e.g. cylinder surface)
        if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
            qdot1[:,0,:] = 0.0
            qdot2[:,0,:] = 0.0
            qdot3[:,0,:] = 0.0
            qdot4[:, 0, :] = qdot0[:, 0, :] * E[:, 0, :]
        if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
            # Only treat rho and rhoE as Dirichlet if applying absorbing layer
            qdot0[:,0,:] = 0.0
            qdot4[:, 0, :] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[0,:,None,None] - rho.interior())
            qdot1 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[1,:,None,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[2,:,None,None] - rhoV.interior() )
            qdot3 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[3,:,None,None] - rhoW.interior() )
            qdot4 += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[4,:,None,None] - rhoE.interior() )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_bot[:,:,None] * ( self.param.Q_BC_bot[5+isc,:,None,None] - rhoSC[isc].interior() )
        
        # Eta top boundary
        if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
            qdot1[:,-1,:] = 0.0
            qdot2[:,-1,:] = 0.0
            qdot3[:,-1,:] = 0.0
        if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
            qdot0[:,-1,:] = 0.0
            qdot4[:,-1,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[0,:,None,None] - rho.interior())
            qdot1 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[1,:,None,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[2,:,None,None] - rhoV.interior() )
            qdot3 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[3,:,None,None] - rhoW.interior() )
            qdot4 += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[4,:,None,None] - rhoE.interior() )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_top[:,:,None] * ( self.param.Q_BC_top[5+isc,:,None,None] - rhoSC[isc].interior() )

        # Apply farfield to non-periodic Xi boundaries
        if (not self.grid.periodic_xi and self.param.iproc==0):
            qdot0[0,:,:] = 0.0
            qdot1[0,:,:] = 0.0
            qdot2[0,:,:] = 0.0
            qdot3[0,:,:] = 0.0
            qdot4[0,:,:] = 0.0
            # Source terms
            if (self.grid.sigma_BC_left != None):
                # For spatial jets, inflow is constant, so this attribute won't be present
                qdot0 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[0,None,:,None] - rho.interior())
                qdot1 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[1,None,:,None] - rhoU.interior() )
                qdot2 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[2,None,:,None] - rhoV.interior() )
                qdot3 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[3,None,:,None] - rhoW.interior() )
                qdot4 += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[4,None,:,None] - rhoE.interior() )
                # Species
                for isc in range(self.EOS.num_sc):
                    qdotSC[isc] += self.grid.sigma_BC_left[:,:,None] * ( self.param.Q_BC_left[5+isc,None,:,None] - rhoSC[isc].interior() )
            
        if (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
            qdot0[-1,:,:] = 0.0
            qdot1[-1,:,:] = 0.0
            qdot2[-1,:,:] = 0.0
            qdot3[-1,:,:] = 0.0
            qdot4[-1,:,:] = 0.0
            # Source terms
            qdot0 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[0,None,:,None] - rho.interior())
            qdot1 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[1,None,:,None] - rhoU.interior() )
            qdot2 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[2,None,:,None] - rhoV.interior() )
            qdot3 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[3,None,:,None] - rhoW.interior() )
            qdot4 += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[4,None,:,None] - rhoE.interior() )
            # Species
            for isc in range(self.EOS.num_sc):
                qdotSC[isc] += self.grid.sigma_BC_right[:,:,None] * ( self.param.Q_BC_right[5+isc,None,:,None] - rhoSC[isc].interior() )
                

        qdot_list = [qdot0,qdot1,qdot2,qdot3,qdot4] + qdotSC
        return torch.stack(qdot_list, dim=0), None # None here should be model outputs!
