"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Bodyforce.py

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

class Bodyforce:
    def __init__(self,cfg,comms,decomp,grid,Q):

        # Save pointer to comms object
        self.comms  = comms

        # Needed from decomp
        self.jproc = decomp.jproc; self.npy = decomp.npy

        # Configuration options
        # Channel
        self.channel = False

        if hasattr(cfg,'IC_opt'):
            if (cfg.IC_opt=='channel'):
                self.channel = True
            
                # Grid spacings
                if (decomp.nx > 1):
                    self.Dx = grid.Dx[:,:,None]
                    
                self.ndim = grid.ndim
                if (self.ndim==3):
                    # 3D
                    self.Dz = grid.Dz
                    # Cell volume
                    self.Dv = grid.Dx[:,:,None] * grid.Dy[:,:,None] * grid.Dz
                    # Domain volume
                    self.V = grid.Lx1 * grid.Lx2 * grid.Lx3
                elif (self.ndim==2):
                    # 2D
                    self.Dz = 1.0
                    # Cell volume
                    self.Dv = grid.Dx[:,:,None] * grid.Dy[:,:,None]
                    # Domain volume
                    self.V = grid.Lx1 * grid.Lx2
                else:
                    # 1D
                    self.Dx = 1.0
                    self.Dz = 1.0
                    # Cell volume
                    self.Dv = grid.Dy[:,:,None]
                    # Domain volume
                    self.V = grid.Lx2
                    
                # Compute initial mass flow rates
                self.ubulk,self.wbulk = self.compute_massflowrate(Q)
                #if (comms.rank==0): print("bulk",self.ubulk,self.wbulk)
         
        return


    def compute_friction(self,mu,du_dy,dw_dy):

        tmp_x = 0.0
        tmp_z = 0.0
        
        # Lower wall
        if (self.jproc==0):
            mu_loc = mu[:,0,:]

            if (self.ndim==1):
                tmp_x += torch.sum( self.Dz * self.Dx * mu_loc * du_dy[:,0,:] )
            else:
                tmp_x += torch.sum( self.Dz * self.Dx[:,0,:] * mu_loc * du_dy[:,0,:] )
            if (dw_dy is not None):
                tmp_z += torch.sum( self.Dz * self.Dx[:,0,:] * mu_loc * dw_dy[:,0,:] )

        # Upper wall
        if (self.jproc==self.npy-1):
            mu_loc = mu[:,-1,:]

            if (self.ndim==1):
                tmp_x -= torch.sum( self.Dz * self.Dx * mu_loc * du_dy[:,-1,:] )
            else:
                tmp_x -= torch.sum( self.Dz * self.Dx[:,-1,:] * mu_loc * du_dy[:,-1,:] )
            if (dw_dy is not None):
                tmp_z -= torch.sum( self.Dz * self.Dx[:,-1,:] * mu_loc * dw_dy[:,-1,:] )

        # Sum over the procs
        #(Fx becomes -ve so does temp x at the start for tanh and uniform is +ve)
        Fx = self.comms.parallel_sum( tmp_x ) / self.V
        
        if (dw_dy is not None):
            Fz = self.comms.parallel_sum( tmp_z ) / self.V
        else:
            Fz = 0.0
            
        return Fx,Fz


    def compute_massflowrate(self,Q):
        
        tmp_x = torch.sum( Q['rhoU'].interior() * self.Dv )
        tmp_z = torch.sum( Q['rhoW'].interior() * self.Dv )
        
        umean = self.comms.parallel_sum( tmp_x ) / self.V
        wmean = self.comms.parallel_sum( tmp_z ) / self.V
       
        return umean,wmean


    def bodyforce_channel(self,Q,dt,mu,du_dy,dw_dy):

        Fx,Fz = self.compute_friction(mu,du_dy,dw_dy) #mu is taken care of here
        umean,wmean = self.compute_massflowrate(Q)

        alpha = 1.0
        srcU = alpha * (self.ubulk - umean)/dt + Fx
        srcW = alpha * (self.wbulk - wmean)/dt + Fz
        #if(self.comms.rank==0): print("bulk velocity: ",self.ubulk-umean,self.wbulk - wmean)
        return srcU,srcW


    def compute(self,Q,dt,mu,du_dy,dw_dy):
        srcU = 0.0
        srcW = 0.0

        if (self.channel):
            srcU,srcW = self.bodyforce_channel(Q,dt,mu,du_dy,dw_dy)

        return srcU,srcW
