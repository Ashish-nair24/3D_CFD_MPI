"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Adjoint.py

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


class Adjoint_RHS:
    def __init__(self, comms, grid, metrics, param, rhs, EOS):
        
        self.comms = comms
        self.grid = grid
        self.metrics = metrics
        self.param = param
        self.rhs = rhs
        self.EOS = EOS
        
        return

    def RHS(self, q, q_a, Q_T):

        # Start tracking operations
        q.requires_grad(True)

        # Evaluate the forward RHS
        k_fwd, model_outputs = self.rhs( q )

        # ---------------------------------------------------------------
        # Adjoint RHS
        rho_A  = q_a['rho_A']
        rhoU_A = q_a['rhoU_A']
        rhoV_A = q_a['rhoV_A']
        rhoW_A = q_a['rhoW_A']
        rhoE_A = q_a['rhoE_A']

        ## dq_A/dt = q_A * dF/dq + dJ/dq
        # CHECK -1 IN PARAMETER UPDATE
        g = ( torch.sum( k_fwd[0,:,:,:] *  rho_A.interior() ) +
              torch.sum( k_fwd[1,:,:,:] * rhoU_A.interior() ) +
              torch.sum( k_fwd[2,:,:,:] * rhoV_A.interior() ) +
              torch.sum( k_fwd[3,:,:,:] * rhoW_A.interior() ) +
              torch.sum( k_fwd[4,:,:,:] * rhoE_A.interior() ) )
        g.backward(retain_graph=True)

        # Evaluate dJ/dq
        J,_,_,_,_ = self.param.loss(self.comms, self.grid, self.param, self.metrics, self.EOS, q, Q_T, model_outputs) #/ self.param.dt
        J.backward()

        q_rho_A  = self.metrics.full2int(  q['rho'].var.grad.data.detach() )
        q_rhoU_A = self.metrics.full2int( q['rhoU'].var.grad.data.detach() )
        q_rhoE_A = self.metrics.full2int( q['rhoE'].var.grad.data.detach() )
        # 2D / 3D
        if (self.grid.ndim > 1):
            q_rhoV_A = self.metrics.full2int( q['rhoV'].var.grad.data.detach() )
        else:
            q_rhoV_A = 0.0 * q_rhoU_A
        # 3D
        if (self.grid.ndim > 2):
            q_rhoW_A = self.metrics.full2int( q['rhoW'].var.grad.data.detach() )
        else:
            q_rhoW_A = 0.0 * q_rhoU_A

        # Stop tracking operations
        q.requires_grad(False)

        # # Boundary conditions
        # if (not self.grid.periodic_xi and self.param.iproc==0):
        #     q_rho_A[ 0,:,:] = 0.0
        #     q_rhoU_A[0,:,:] = 0.0
        #     q_rhoV_A[0,:,:] = 0.0
        #     q_rhoW_A[0,:,:] = 0.0
        #     q_rhoE_A[0,:,:] = 0.0
        # if (not self.grid.periodic_xi and self.param.iproc==self.param.npx-1):
        #     q_rho_A[ -1,:,:] = 0.0
        #     q_rhoU_A[-1,:,:] = 0.0
        #     q_rhoV_A[-1,:,:] = 0.0
        #     q_rhoW_A[-1,:,:] = 0.0
        #     q_rhoE_A[-1,:,:] = 0.0
        # if (not self.grid.BC_eta_bot=='periodic' and self.param.jproc==0):
        #     q_rhoU_A[:,0,:] = 0.0
        #     q_rhoV_A[:,0,:] = 0.0
        #     q_rhoW_A[:,0,:] = 0.0
        # if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
        #     q_rho_A[ :,0,:] = 0.0
        #     q_rhoE_A[:,0,:] = 0.0
        # if (not self.grid.BC_eta_top=='periodic' and self.param.jproc==self.param.npy-1):
        #     q_rhoU_A[:,-1,:] = 0.0
        #     q_rhoV_A[:,-1,:] = 0.0
        #     q_rhoW_A[:,-1,:] = 0.0
        # if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
        #     q_rho_A[ :,-1,:] = 0.0
        #     q_rhoE_A[:,-1,:] = 0.0

        k_A = torch.stack((q_rho_A, q_rhoU_A, q_rhoV_A, q_rhoW_A, q_rhoE_A ),dim=0)

        return k_fwd.detach(), k_A.detach()


    # Function to calculate Loss derivative for Adjoint RHS
    def Loss_der(self, q, Q_T):
        
        # free-stream constants
        # rho_inf =  4.948292209051355e-06
        # u_inf = 1756.1372531052273
        # v_inf = 0.876647
        # T_inf = 300
        # P_inf = 300
        # Re = 132.04039497031783 
        # Ma = 5.939404
        # gamma = 1.667
        
        # q_rho_A = (1/(self.grid.Nx1*self.grid.Nx2)) * (1/self.EOS.rho0**2) * (q['rho'].interior()- Q_T['rho'].interior())
        # q_rhoU_A = (1/(self.grid.Nx1*self.grid.Nx2))  * (1/self.EOS.U0**2) * ((q['rhoU'].interior() / q['rho'].interior()) - Q_T['U'].interior()) * ()
        # q_rhoV_A = 0.0 * (1/(self.grid.Nx1*self.grid.Nx2)) * (q['rho'].interior() - Q_T['rho'].interior())
        # q_rhoE_A = 0.0 * (1/(self.grid.Nx1*self.grid.Nx2)) * (q['rho'].interior() - Q_T['rho'].interior())
        
        q.requires_grad(True)
        
        J,_,_,_,_ = self.param.loss(self.comms, self.grid, self.param, self.metrics, self.EOS,q, Q_T, None) 
        J.backward()
 
        q_rho_A  = self.metrics.full2int(  q['rho'].var.grad.data.detach() )
        q_rhoU_A = self.metrics.full2int( q['rhoU'].var.grad.data.detach() )#0.0 * self.metrics.full2int(  q['rho'].var.grad.data.detach() ) #
        q_rhoV_A = self.metrics.full2int(  q['rhoV'].var.grad.data.detach() )#self.metrics.full2int( q['rhoV'].var.grad.data.detach() )
        q_rhoE_A = self.metrics.full2int(  q['rhoE'].var.grad.data.detach() )#self.metrics.full2int( q['rhoE'].var.grad.data.detach() ) 


        # Clearing gradients
        q.requires_grad(False)

        q['rho'].var.grad = None
        q['rhoU'].var.grad = None
        q['rhoV'].var.grad = None
        q['rhoE'].var.grad = None

    
        # if (self.param.lossVar== 'all'):
        #     q_rho_A  = self.metrics.full2int(  q['rho'].var.grad.data.detach() )
        #     q_rhoU_A = self.metrics.full2int( q['rhoU'].var.grad.data.detach() )
        #     q_rhoE_A = self.metrics.full2int( 0.0 * q['rho'].var.grad.data.detach() ) 
        #     #q_rho_A = 0.0 * q_rhoU_A
        #     #q_rhoE_A = 0.0 * q_rhoU_A
        #     if (self.grid.ndim > 1):
        #         q_rhoV_A = self.metrics.full2int( 0.0 * q['rho'].var.grad.data.detach() )
        #         #q_rhoV_A = 0.0 * q_rhoU_A
        #     else:
        #         q_rhoV_A = 0.0 * q_rho_A
            
        #     if (self.grid.ndim > 2):
        #         q_rhoW_A = self.metrics.full2int( q['rhoW'].var.grad.data.detach() )
        #     else: 
        #         q_rhoW_A = 0.0 * q_rhoU_A
            
        #     q.requires_grad(False)
            
        #     # # Boundary conditions
        #     # if (self.param.solver_mode != 'unsteady'):
        #     #     if (self.grid.BC_xi_left=='farfield' and self.param.iproc==0):
        #     #         q_rho_A[ 0,:,:] = 0.0     
        #     #         q_rhoU_A[0,:,:] = 0.0    
        #     #         q_rhoV_A[0,:,:] = 0.0    
        #     #         q_rhoW_A[0,:,:] = 0.0    
        #     #         q_rhoE_A[0,:,:] = 0.0    
        #     #     if (self.grid.BC_xi_right=='farfield' and self.param.iproc==self.param.npx-1):
        #     #         q_rho_A[ -1,:,:] = 0.0     
        #     #         q_rhoU_A[-1,:,:] = 0.0    
        #     #         q_rhoV_A[-1,:,:] = 0.0    
        #     #         q_rhoW_A[-1,:,:] = 0.0    
        #     #         q_rhoE_A[-1,:,:] = 0.0    
                
        #     #     # Adiabatic walls only!
        #     #     if (self.grid.BC_eta_bot == 'wall' and self.grid.__class__.__name__ == 'flat_plate' and self.param.jproc==0):
        #     #         q_rhoU_A[self.grid.cut_ind:,0,:] = 0.0
        #     #         q_rhoV_A[self.grid.cut_ind:,0,:] = 0.0
        #     #     elif (self.grid.BC_eta_bot=='wall'  and self.param.jproc==0):   
        #     #         q_rhoU_A[:,0,:] = 0.0   
        #     #         q_rhoV_A[:,0,:] = 0.0 
        #     #     if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):
        #     #         q_rho_A[ :,0,:] = 0.0
        #     #         q_rhoU_A[:,0,:] = 0.0
        #     #         q_rhoV_A[:,0,:] = 0.0
        #     #         q_rhoW_A[:,0,:] = 0.0
        #     #         q_rhoE_A[:,0,:] = 0.0
        #     #     """ if (self.grid.BC_eta_top=='wall' or self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):      
        #     #         q_rho_A[:,-1,:] = 0.0 
        #     #         q_rhoE_A[:,-1,:] = 0.0   """
        #     #     if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):
        #     #         q_rho_A[ :,-1,:] = 0.0
        #     #         q_rhoU_A[ :,-1,:] = 0.0
        #     #         q_rhoV_A[ :,-1,:] = 0.0
        #     #         q_rhoW_A[ :,-1,:] = 0.0
        #     #         q_rhoE_A[:,-1,:] = 0.0
            
        
        # # elif (self.param.lossVar == 'U'):  
        # #     zero = torch.tensor((0.0,), dtype=torch.float64).to(self.param.device)
            
        # #     q_rhoU_A  = self.metrics.full2int( q['rhoU'].var.grad.data.detach() )
        # #     q_rho_A = 0.0 * q_rhoU_A
        # #     q_rhoV_A = 0.0 * q_rhoU_A
        # #     q_rhoW_A = 0.0 * q_rhoU_A
        # #     q_rhoE_A = 0.0 * q_rhoU_A
 
        # #     q.requires_grad(False)   
        # #     # if (self.param.solver_mode != 'unsteady'):     
        # #     #     # Boundary conditions                                                        
        # #     #     if (self.grid.BC_xi_left =='farfield' and self.param.iproc==0):
        # #     #         q_rhoU_A[0,:,:] = 0.0    
        # #     #     if (self.grid.BC_xi_right =='farfield' and self.param.iproc==self.param.npx-1):
        # #     #         q_rhoU_A[-1,:,:] = 0.0    
        # #     #     if (self.grid.BC_eta_bot=='farfield' and self.param.jproc==0):   
        # #     #         q_rhoU_A[:,0,:] = 0.0   
        # #     #     if (self.grid.BC_eta_top=='farfield' and self.param.jproc==self.param.npy-1):      
        # #     #         q_rhoU_A[:,-1,:] = 0.0 

        # #     #     if (self.grid.BC_eta_bot == 'wall' and self.grid.__class__.__name__ == 'flat_plate' and self.param.jproc==0):
        # #     #         q_rhoU_A[self.grid.cut_ind:,0,:] = 0.0
        # #     #     elif (self.grid.BC_eta_bot=='wall'  and self.param.jproc==0):   
        # #     #         q_rhoU_A[:,0,:] = 0.0   
          
        k_A = torch.stack((q_rho_A, q_rhoU_A, q_rhoV_A, q_rhoE_A ),dim=0)
        
        
        # Setting BCs
        # xLeftCut  =  self.grid.xIndOptLeft   
        # xRightCut =  self.grid.xIndOptRight  
        # yTopCut   =  self.grid.yIndOptTop    
        # yBotCut   =  self.grid.yIndOptBot    
        
        
        # # Xi-direction
        # k_A[:, :xLeftCut, :, :] = 0.0
        # k_A[:, xRightCut:, :, :] = 0.0
        
        # # Eta-direction
        # k_A[:, :, :yBotCut, :] = 0.0
        # k_A[:, :, yTopCut:, :] = 0.0
       
        
        
        #k_A = q_rhoU_A[None,:,:,:]
        return k_A.detach()
