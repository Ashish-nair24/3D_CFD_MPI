"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Parallel.py

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
#import cupy as cp

   
# ----------------------------------------------------
# Parallel communication functions
# ----------------------------------------------------
class Comms:
    def __init__(self):
        if (not MPI.Is_initialized()):
            MPI.Init()
        
        # Get MPI decomposition info
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def finalize(self):
        MPI.Finalize()

    def parallel_sum(self, sendBuf, comm=None):
        if comm is None:
            comm = self.comm
        if (self.size>1):
            if hasattr(sendBuf,'__cuda_array_interface__'):
                device = sendBuf.device
                cuda = True
            else:
                cuda = False
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
                if cuda: sendBuf = cp.asarray(sendBuf)
                else:    sendBuf = np.asarray(sendBuf)
            else:
                sendLen = 1
                if cuda:
                    buf = torch.tensor((sendBuf,),device=device)
                    sendBuf = cp.asarray(buf)
                else:
                    sendBuf = np.asarray((sendBuf,))
            if cuda: recvBuf = cp.zeros_like(sendBuf)
            else:    recvBuf = np.zeros_like(sendBuf)
            comm.Allreduce(sendBuf,recvBuf,op=MPI.SUM)
            if (sendLen==1):
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)[0]
                else:
                    out = recvBuf[0]
            else:
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)
                else:
                    out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    def parallel_max(self, sendBuf, comm=None):
        if comm is None:
            comm = self.comm
        if (self.size>1):
            if hasattr(sendBuf,'__cuda_array_interface__'):
                device = sendBuf.device
                cuda = True
            else:
                cuda = False
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
                if cuda: sendBuf = cp.asarray(sendBuf)
                else:    sendBuf = np.asarray(sendBuf)
            else:
                sendLen = 1
                if cuda:
                    buf = torch.tensor((sendBuf,),device=device)
                    sendBuf = cp.asarray(buf)
                else:
                    sendBuf = np.asarray((sendBuf,))
            if cuda: recvBuf = cp.zeros_like(sendBuf)
            else:    recvBuf = np.zeros_like(sendBuf)
            comm.Allreduce(sendBuf,recvBuf,op=MPI.MAX)
            if (sendLen==1):
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)[0]
                else:
                    out = recvBuf[0]
            else:
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)
                else:
                    out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    def parallel_min(self, sendBuf, comm=None):
        if comm is None:
            comm = self.comm
        if (self.size>1):
            if hasattr(sendBuf,'__cuda_array_interface__'):
                device = sendBuf.device
                cuda = True
            else:
                cuda = False
            if (hasattr(sendBuf,"__size__")):
                sendLen = len(sendBuf)
                if cuda: sendBuf = cp.asarray(sendBuf)
                else:    sendBuf = np.asarray(sendBuf)
            else:
                sendLen = 1
                if cuda:
                    buf = torch.tensor((sendBuf,),device=device)
                    sendBuf = cp.asarray(buf)
                else:
                    sendBuf = np.asarray((sendBuf,))
            if cuda: recvBuf = cp.zeros_like(sendBuf)
            else:    recvBuf = np.zeros_like(sendBuf)
            comm.Allreduce(sendBuf,recvBuf,op=MPI.MIN)
            if (sendLen==1):
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)[0]
                else:
                    out = recvBuf[0]
            else:
                if cuda:
                    out = torch.as_tensor(recvBuf,device=device)
                else:
                    out = recvBuf
        else:
            # Serial computation; nothing to do
            out = sendBuf
        return out

    
# ----------------------------------------------------
# MPI decomposition
# ----------------------------------------------------
class Decomp:
    def __init__(self,cfg,grid,WP,WP_np):
        self.WP = WP
        self.WP_np = WP_np

        # Offloading settings
        self.device = cfg.device
        
        # ---------------------------------------
        # MPI communicators
        
        # Get the global communicator
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Cartesian task decomposition
        self.npx = cfg.nproc_x
        self.npy = cfg.nproc_y
        self.npz = cfg.nproc_z

        # Minibatch setup
        # Number of MPI tasks in the minibatch dimension
        if hasattr(cfg, 'nproc_minibatch'):
            self.npm = cfg.nproc_minibatch
        else:
            self.npm = 1
        # Number of minibatch simulations per MPI task
        if hasattr(cfg, 'N_minibatch_per_task'):
            self.nm_per_task = cfg.N_minibatch_per_task
        else:
            # Defaults to 1 minibatch per MPI task
            self.nm_per_task = 1

        # Overall MPI task decomposition
        nproc_decomp = [self.npx,self.npy,self.npz,self.npm]
        
        # Check the decomp
        if (self.size!=self.npx*self.npy*self.npz*self.npm):
            raise Exception('\nNumber of MPI tasks does not match the specified domain decomposition\n')

        # Cartesian communicator to determine coordinates
        self.isper = ( grid.periodic_xi,
                       grid.periodic_eta,
                       True,
                       False )
        self.cartComm = self.comm.Create_cart(nproc_decomp,periods=self.isper,
                                              reorder=True)
        
        # Proc's location in the cartesian communicator
        self.iproc,self.jproc,self.kproc,self.mproc = self.cartComm.Get_coords(self.rank)

        # Create line communicators
        # Along x
        dir = [True,False,False,False]
        self.cartCommX = self.cartComm.Sub(dir)
        #self.irank_x   = self.cartCommX.Get_rank()
        # Along y
        dir = [False,True,False,False]
        self.cartCommY = self.cartComm.Sub(dir)
        #self.irank_y   = self.cartCommY.Get_rank()
        # Along z
        dir = [False,False,True,False]
        self.cartCommZ = self.cartComm.Sub(dir)
        #self.irank_z   = self.cartCommZ.Get_rank()
        # Along minibatch dimension
        dir = [False,False,False,True]
        self.cartCommM = self.cartComm.Sub(dir)

        # Create plane communicators
        # Along xy
        dir = [True,True,False,False]
        self.cartCommXY = self.cartComm.Sub(dir)
        # Along xz
        dir = [True,False,True,False]
        self.cartCommXZ = self.cartComm.Sub(dir)
        # Along yz
        dir = [False,True,True,False]
        self.cartCommYZ = self.cartComm.Sub(dir)

        # Along xyz
        dir = [True,True,True,False]
        self.cartCommXYZ = self.cartComm.Sub(dir)

        
        # ---------------------------------------
        # Domain decomposition
        
        # Grid size
        self.nx = grid.Nx1
        self.ny = grid.Nx2
        self.nz = grid.Nx3
        self.N = self.nx*self.ny*self.nz

        # Global number of minibatch simulations
        self.nm = self.npm * self.nm_per_task

        # Overlap size
        # 3 for derivatives, 4 for filtering
        # 6 for "extended interior" derivatives
        self.nover = 5

        # Global indexing
        self.nxo  = self.nx+2*self.nover
        self.nyo  = self.ny+2*self.nover
        self.nzo  = self.nz+2*self.nover
        self.imino = 0
        self.jmino = 0
        self.kmino = 0
        self.imaxo = self.imino+self.nxo-1
        self.jmaxo = self.jmino+self.nyo-1
        self.kmaxo = self.kmino+self.nzo-1
        self.imin  = self.imino+self.nover
        self.jmin  = self.jmino+self.nover
        self.kmin  = self.kmino+self.nover
        self.imax  = self.imin+self.nx-1
        self.jmax  = self.jmin+self.ny-1
        self.kmax  = self.kmin+self.nz-1
        
        # Decomposition:
        #   imin_loc, imax_loc, etc. are local positions
        #   in global grid but do NOT include overlap cells
        
        imin = 0
        q = int(self.nx/self.npx)
        r = int(np.mod(self.nx,self.npx))
        if ((self.iproc+1)<=r):
            self.nx_   = q+1
            self.imin_loc = imin + self.iproc*(q+1)
        else:
            self.nx_   = q
            self.imin_loc = imin + r*(q+1) + (self.iproc-r)*q
        self.imax_loc = self.imin_loc + self.nx_ - 1
        
        # y-deomposition
        jmin = 0
        q = int(self.ny/self.npy)
        r = int(np.mod(self.ny,self.npy))
        if ((self.jproc+1)<=r):
            self.ny_   = q+1
            self.jmin_loc = jmin + self.jproc*(q+1)
        else:
            self.ny_   = q
            self.jmin_loc = jmin + r*(q+1) + (self.jproc-r)*q
        self.jmax_loc = self.jmin_loc + self.ny_ - 1
        
        # z-decomposition
        kmin = 0
        q = int(self.nz/self.npz)
        r = int(np.mod(self.nz,self.npz))
        if ((self.kproc+1)<=r):
            self.nz_   = q+1
            self.kmin_loc = kmin + self.kproc*(q+1)
        else:
            self.nz_   = q
            self.kmin_loc = kmin + r*(q+1) + (self.kproc-r)*q
        self.kmax_loc = self.kmin_loc + self.nz_ - 1

        # Minibatch decomposition
        q = int(self.nm/self.npm)
        r = int(np.mod(self.nm,self.npm))
        if ((self.mproc+1)<=r):
            self.nm_ = q+1
            self.mmin_loc = 0 + self.mproc*(q+1)
        else:
            self.nm_ = q
            self.mmin_loc = 0 + r*(q+1) + (self.mproc-r)*q
        self.mmax_loc = self.mmin_loc + self.nm_ - 1

        # print('rank={}\t nm_={}\t mmin_={}\t mmax_={}'.format(self.rank,self.nm_,self.mmin_loc,self.mmax_loc))

        #print("rank={}\t imin_={}\timax_={}\tnx_={}\t jmin_={}\tjmax_={}\tny_={}\t kmin_={}\tkmax_={}\tnz_={}"
        #      .format(self.rank,
        #              self.imin_loc,self.imax_loc,self.nx_,
        #              self.jmin_loc,self.jmax_loc,self.ny_,
        #              self.kmin_loc,self.kmax_loc,self.nz_))

        # Local indexing including overlaps
        self.nxo_  = self.nx_+2*self.nover
        self.nyo_  = self.ny_+2*self.nover
        self.nzo_  = self.nz_+2*self.nover
        self.imino_ = 0
        self.jmino_ = 0
        self.kmino_ = 0
        self.imaxo_ = self.imino_+self.nxo_-1
        self.jmaxo_ = self.jmino_+self.nyo_-1
        self.kmaxo_ = self.kmino_+self.nzo_-1

        # Local indexing for interior only
        self.imin_ = self.imino_+self.nover
        self.jmin_ = self.jmino_+self.nover
        self.kmin_ = self.kmino_+self.nover
        self.imax_ = self.imin_+self.nx_-1
        self.jmax_ = self.jmin_+self.ny_-1
        self.kmax_ = self.kmin_+self.nz_-1

        # Minibatch indexing
        self.mmin_ = 0
        self.mmax_ = self.nm_-1

        # Eliminate z-overlaps if problem is 2D
        if (self.nz==1):
            self.nzo    = 1
            self.nz_    = 1
            self.nzo_   = 1
            self.kmaxo  = 0
            self.kmino_ = 0
            self.kmaxo_ = 0
            self.kmin_  = 0
            self.kmax_  = 0

        # Eliminate x-overlaps if problem is 1D
        if (self.nx==1):
            self.nxo    = 1
            self.nx_    = 1
            self.nxo_   = 1
            self.imaxo  = 0
            self.imino_ = 0
            self.imaxo_ = 0
            self.imin_  = 0
            self.imax_  = 0


    # ------------------------------------------------
    # Destructor
    def __del__(self):
        self.cartCommX.Free()
        self.cartCommY.Free()
        self.cartCommZ.Free()
        self.cartCommXY.Free()
        self.cartCommXZ.Free()
        self.cartCommYZ.Free()
        self.cartComm.Free()

        
    # ------------------------------------------------
    # Communicate overlap cells for generic state data
    def communicate_border(self,A):
        n1 = self.nxo_
        n2 = self.nyo_
        n3 = self.nzo_
        no = self.nover

        # x
        if (self.nx > 1):
            self.communicate_border_x(A,n1,n2,n3,no)
        # y
        self.communicate_border_y(A,n1,n2,n3,no)
        # z
        if (self.nz > 1):
            self.communicate_border_z(A,n1,n2,n3,no)
        
    # ------------------------------------------------
    # Communicate overlap cells for 2D data
    def communicate_border_2D(self,A):
        n1 = self.nxo_
        n2 = self.nyo_
        n3 = 1
        no = self.nover

        # x
        if (self.nx > 1):
            self.communicate_border_x(A,n1,n2,n3,no)
        # y
        self.communicate_border_y(A,n1,n2,n3,no)
        
    def communicate_border_2D_ADJOINT(self,A):
        n1 = self.nxo_
        n2 = self.nyo_
        n3 = 1
        no = self.nover

        # x
        if (self.nx > 1):
            self.communicate_border_x_ADJOINT(A,n1,n2,n3,no)
        # y
        self.communicate_border_y_ADJOINT(A,n1,n2,n3,no)        
        
            
    # --------------------------------------------
    # Communicate overlap cells in the x-direction
    def communicate_border_x(self,A,n1,n2,n3,no):

        device = A.device
        # Left buffer
        sendbuf = A.detach()[no:2*no,:,:].to(torch.device('cpu')).numpy()
        recvbuf = np.empty([no,n2,n3],dtype=self.WP_np)
        #sendbuf = cp.asarray(A[no:2*no,:,:])
        #recvbuf = cp.empty_like(sendbuf)
        
        # Send left buffer to left neighbor
        #assert hasattr(sendbuf, '__cuda_array_interface__')
        isource,idest = self.cartComm.Shift(0,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received left buffer to the right overlap cells
        if (isource!=MPI.PROC_NULL):
            #buf = np.asarray(recvbuf.get())
            #A[n1-no:n1,:,:].copy_(torch.from_numpy(buf).to(self.device))
            A[n1-no:n1,:,:].copy_(torch.from_numpy(recvbuf).to(self.device))
            #A[n1-no:n1,:,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Right buffer
        sendbuf = A.detach()[n1-2*no:n1-no,:,:].to(torch.device('cpu')).numpy()
        recvbuf = np.empty_like(sendbuf)
        #sendbuf = cp.asarray(A[n1-2*no:n1-no,:,:])
        
        # Send right buffer to right neighbor
        isource,idest = self.cartComm.Shift(0,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received right buffer to the left overlap cells
        if (isource!=MPI.PROC_NULL):
            A[0:no,:,:].copy_(torch.from_numpy(recvbuf).to(self.device))
            #A[0:no,:,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Clean up
        del recvbuf
        
        
    # --------------------------------------------
    # Communicate overlap cells in the x-direction for adjoint
    def communicate_border_x_ADJOINT(self,A,n1,n2,n3,no):

        device = A.device
        # Left buffer
        sendbuf = A[0:1*no,:,:].to(torch.device('cpu')).numpy()
        recvbuf = np.empty([no,n2,n3],dtype=self.WP_np)
        #sendbuf = cp.asarray(A[no:2*no,:,:])
        #recvbuf = cp.empty_like(sendbuf)
        
        # Send left buffer to left neighbor
        #assert hasattr(sendbuf, '__cuda_array_interface__')
        isource,idest = self.cartComm.Shift(0,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received left buffer to the right overlap cells
        if (isource!=MPI.PROC_NULL):
            A[-2*no:-1*no,:,:] += torch.from_numpy(recvbuf).to(self.device)
            

        # Right buffer
        sendbuf = A[-1*no:,:,:].to(torch.device('cpu')).numpy()
        recvbuf = np.empty_like(sendbuf)
        #sendbuf = cp.asarray(A[n1-2*no:n1-no,:,:])
        
        # Send right buffer to right neighbor
        isource,idest = self.cartComm.Shift(0,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received right buffer to the left overlap cells
        if (isource!=MPI.PROC_NULL):
            #A[0:no,:,:].copy_(torch.from_numpy(recvbuf).to(self.device))
            A[1*no:2*no,:,:] += torch.from_numpy(recvbuf).to(self.device)

        # Clean up
        del recvbuf        
        
        
    # --------------------------------------------
    # Communicate overlap cells in the y-direction
    def communicate_border_y(self,A,n1,n2,n3,no):

        # Initialize the buffers
        # We need to allocate and copy since y is not contiguous in memory
        sendbuf = np.empty([n1,no,n3],dtype=self.WP_np)
        recvbuf = np.empty([n1,no,n3],dtype=self.WP_np)
        #sendbuf = cp.empty([n1,no,n3],dtype=self.WP_np)
        #recvbuf = cp.empty([n1,no,n3],dtype=self.WP_np)
        icount = no*n1*n3
        device = A.device
        
        # Lower buffer
        sendbuf = np.copy(A.detach()[:,no:2*no,:].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,no:2*no,:]))
        
        # Send lower buffer to lower neighbor
        isource,idest = self.cartComm.Shift(1,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received lower buffer to the upper overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,n2-no:n2,:] = torch.from_numpy(recvbuf).to(self.device)
            #A[:,n2-no:n2,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Upper buffer
        sendbuf = np.copy(A.detach()[:,n2-2*no:n2-no,:].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,n2-2*no:n2-no,:]))
        
        # Send upper buffer to upper neighbor
        isource,idest = self.cartComm.Shift(1,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received upper buffer to the lower overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,0:no,:] = torch.from_numpy(recvbuf).to(self.device)
            #A[:,0:no,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Clean up
        del sendbuf
        del recvbuf
        
    # --------------------------------------------
    # Communicate overlap cells in the y-direction
    def communicate_border_y_ADJOINT(self,A,n1,n2,n3,no):

        # Initialize the buffers
        # We need to allocate and copy since y is not contiguous in memory
        sendbuf = np.empty([n1,no,n3],dtype=self.WP_np)
        recvbuf = np.empty([n1,no,n3],dtype=self.WP_np)
        #sendbuf = cp.empty([n1,no,n3],dtype=self.WP_np)
        #recvbuf = cp.empty([n1,no,n3],dtype=self.WP_np)
        icount = no*n1*n3
        device = A.device
        
        # Lower buffer
        sendbuf = np.copy(A[:,0*no:1*no,:].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,no:2*no,:]))
        
        # Send lower buffer to lower neighbor
        isource,idest = self.cartComm.Shift(1,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received lower buffer to the upper overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,-2*no:-1*no,:] += torch.from_numpy(recvbuf).to(self.device)
            #A[:,n2-no:n2,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Upper buffer
        sendbuf = np.copy(A[:,-1*no:,:].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,n2-2*no:n2-no,:]))
        
        # Send upper buffer to upper neighbor
        isource,idest = self.cartComm.Shift(1,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received upper buffer to the lower overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,1*no:2*no,:] += torch.from_numpy(recvbuf).to(self.device)
            #A[:,0:no,:].copy_(torch.as_tensor(recvbuf,device=device))

        # Clean up
        del sendbuf
        del recvbuf
        
        
        
    # --------------------------------------------
    # Communicate overlap cells in the z-direction
    def communicate_border_z(self,A,n1,n2,n3,no):

        # Initialize the buffers
        # We need to allocate and copy since z is not contiguous in memory
        sendbuf = np.empty([n1,n2,no],dtype=self.WP_np)
        recvbuf = np.empty([n1,n2,no],dtype=self.WP_np)
        #recvbuf = cp.empty([n1,n2,no],dtype=self.WP_np)
        icount = no*n2*n3
        device = A.device
        
        # Front buffer
        sendbuf = np.copy(A[:,:,no:2*no].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,:,no:2*no]))
        
        # Send front buffer to front neighbor
        isource,idest = self.cartComm.Shift(2,-1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received front buffer to the back overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,:,n3-no:n3] = torch.from_numpy(recvbuf).to(self.device)
            #A[:,:,n3-no:n3].copy_(torch.as_tensor(recvbuf,device=device))

        # Back buffer
        sendbuf = np.copy(A[:,:,n3-2*no:n3-no].to(torch.device('cpu')).numpy())
        #sendbuf = cp.copy(cp.asarray(A[:,:,n3-2*no:n3-no]))
        
        # Send back buffer to back neighbor
        isource,idest = self.cartComm.Shift(2,+1)
        self.cartComm.Sendrecv(sendbuf,idest,0,recvbuf,isource,0)

        # Copy the received back buffer to the front overlap cells
        if (isource!=MPI.PROC_NULL):
            A[:,:,0:no] = torch.from_numpy(recvbuf).to(self.device)
            #A[:,:,0:no].copy_(torch.as_tensor(recvbuf,device=device))

        # Clean up
        del sendbuf
        del recvbuf
