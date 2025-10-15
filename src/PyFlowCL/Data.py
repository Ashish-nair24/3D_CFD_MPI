"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Data.py

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
import h5py

# --------------------------------------------------------------
# Base class for PyFlowCL state variables
# --------------------------------------------------------------
class PCL_Var:
    def __init__(self, decomp, name, force_2D=False):
        
        # Data sizes
        nx_  = decomp.nx_
        ny_  = decomp.ny_
        nz_  = decomp.nz_
        nxo_ = decomp.nxo_
        nyo_ = decomp.nyo_
        nzo_ = decomp.nzo_
        self.nover  = decomp.nover
        self.imin_  = decomp.imin_;  self.imax_  = decomp.imax_
        self.jmin_  = decomp.jmin_;  self.jmax_  = decomp.jmax_
        self.kmin_  = decomp.kmin_;  self.kmax_  = decomp.kmax_
        self.imino_ = decomp.imino_; self.imaxo_ = decomp.imaxo_
        self.jmino_ = decomp.jmino_; self.jmaxo_ = decomp.jmaxo_
        self.kmino_ = decomp.kmino_; self.kmaxo_ = decomp.kmaxo_

        if force_2D:
            nz_ = 1; nzo_ = 1
            self.kmin_  = 0; self.kmax_  = 0
            self.kmino_ = 0; self.kmaxo_ = 0
            self.is_2D = True
        else:
            self.is_2D = False
            
        # Allocate data array
        if (name=='rho' or name=='rhoE' or name=='e'):
            # Initialize to 1.0 to avoid divide by zero
            self.var = torch.ones(nxo_,nyo_,nzo_, dtype=decomp.WP, requires_grad=False).to(decomp.device)
        elif (name=='rhoY_N2' or name=='Y_N2'):
            # Initialize overlaps to pure N2 - needed for get_T() on full domain
            self.var = torch.ones(nxo_,nyo_,nzo_, dtype=decomp.WP, requires_grad=False).to(decomp.device)
            self.copy_(torch.tensor(0.0))
        else:
            # All other variables initizlied to 0.0
            self.var = torch.zeros(nxo_,nyo_,nzo_, dtype=decomp.WP, requires_grad=False).to(decomp.device)

        # Save a pointer to the decomposition object
        self.decomp = decomp

    def copy(self,data):
        # Copy data to the interior and update border
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(data)
        self.update_border()
    

    def add(self,data):
        # Add data to interior and update border
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].add_(data)
        self.update_border()        
        

    def sub(self,data):
        # Add data to interior and update border
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].sub_(data)
        self.update_border()

    def mul(self,data):
        # Add data to interior and update border
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].mul_(data)
        self.update_border()

    # "Underscore" routines do not call update_border()
    def copy_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].copy_(data)

    def add_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].add_(data)

    def sub_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].sub_(data)

    def mul_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].mul_(data)
    
    def add_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].add_(data)

    def sub_(self,data):
        self.var[self.imin_:self.imax_+1,
                 self.jmin_:self.jmax_+1,
                 self.kmin_:self.kmax_+1].sub_(data)

    def interior(self):
        # Return only data in the subdomain interior
        return self.var[self.imin_:self.imax_+1,
                        self.jmin_:self.jmax_+1,
                        self.kmin_:self.kmax_+1]
    
    def copy_full(self,data):
        self.var.copy_(data)
        
    def update_border(self):
        # Update the overlap cells
        if self.is_2D:
            self.decomp.communicate_border_2D(self.var)
        else:
            self.decomp.communicate_border(self.var)

    def update_interior_ADJOINT(self):
        # Update the overlap cells
        self.decomp.communicate_border_2D_ADJOINT(self.var)        

        
# --------------------------------------------------------------
# Collection of Navier-Stokes state variables
#   Derived from the Python dict type
# --------------------------------------------------------------
class State(dict):
    def __init__(self,names,decomp,*arg,**kw):
        super(State, self).__init__(*arg,**kw)
        self.names = names

        # Allocate state memory
        for name in self.names:
            self[name] = PCL_Var(decomp,name) 

    def copy(self,data):
        # Copy data to the interior for all state vars
        for ivar,name in enumerate(self.names):
            self[name].copy(data[ivar,...])

    def copy_sum(self,Q,data):
        # Copy Q+data to the interior for all state vars
        for ivar,name in enumerate(self.names):
            self[name].copy(Q[name].interior() + data[ivar,...])

    def add(self,data):
        # Add data to the interior for all state vars
        for ivar,name in enumerate(self.names):
            self[name].add(data[ivar,...])

    def deepcopy(self,state_in):
        # Deepcopy without updating ghost cells
        for name in self.names:
            self[name].copy_full(state_in[name].var.detach())

    def requires_grad(self,logical):
        if logical:
            # If true, set requires_grad = True
            for name in self.names:
                self[name].var.requires_grad = logical
        else:
            # If false, detach from the current graph
            # Automatically sets requires_grad = False
            for name in self.names:
                self[name].var = self[name].var.detach()     
            
        

# --------------------------------------------------------------
# Read data -- HDF5 -- Interface function
# --------------------------------------------------------------
def read_data(dfName,cfg,decomp,Q):
    if (not hasattr(cfg,"restart_2D_to_3D")):
        cfg.restart_2D_to_3D = False
        
    try:
        if (cfg.grid.ndim<=2 or (cfg.grid.ndim==3 and cfg.restart_2D_to_3D)):
            return read_hdf5_2D(dfName,decomp,Q)
        elif (cfg.grid.ndim==3):
            return read_hdf5_3D(dfName,decomp,Q)
    except:
        if (decomp.rank==0): print('ERROR: Data.read_hdf5: Could not read restart file.')
        return
        
        
# --------------------------------------------------------------
# Read data -- HDF5 -- 2D -- parallel
# --------------------------------------------------------------
def read_hdf5_2D(dfName,decomp,Q):

    # Sizes
    nx  = decomp.nx;  ny  = decomp.ny
    nx_ = decomp.nx_; ny_ = decomp.ny_
    imin_ = decomp.imin_loc; imax_ = decomp.imax_loc+1
    jmin_ = decomp.jmin_loc; jmax_ = decomp.jmax_loc+1

    args   = (dfName, 'r')
    kwargs = {}
    use_MPI = False
    
    if (h5py.h5fd.MPIO > 0):
        # Use HDF5 MPI support if available
        kwargs = {'driver' : 'mpio', 'comm' : decomp.comm}
        use_MPI = True
    elif (decomp.size > 1):
        raise Exception('Data.py: HDF5 MPI support is required for nproc>1')

    with h5py.File(*args,**kwargs) as f:
        # Time info
        Nstart = np.array(f['Ntime'])[0]
        tstart = np.array(f['time'])[0]
        dt     = np.array(f['dt'])[0]
        # Not reading grid info
      
        # Conserved quantities
        Q_tmp = np.empty((nx_,ny_,1), dtype=decomp.WP_np)
        for name in Q.names:
            
            f[name].read_direct(Q_tmp[:,:,0], np.s_[imin_:imax_,jmin_:jmax_])
            
            # Copy to state buffer
            Q[name].copy( torch.from_numpy(Q_tmp).to(decomp.device) )
                     
    return Nstart,tstart,dt
    
    
    
# --------------------------------------------------------------
# Read data -- HDF5 -- 3D -- parallel
# --------------------------------------------------------------
def read_hdf5_3D(dfName,decomp,Q):
    # Sizes
    nx  = decomp.nx;  ny  = decomp.ny;  nz  = decomp.nz
    nx_ = decomp.nx_; ny_ = decomp.ny_; nz_ = decomp.nz_
    imin_ = decomp.imin_loc; imax_ = decomp.imax_loc+1
    jmin_ = decomp.jmin_loc; jmax_ = decomp.jmax_loc+1
    kmin_ = decomp.kmin_loc; kmax_ = decomp.kmax_loc+1

    args   = (dfName, 'r')
    kwargs = {}
    use_MPI = False
    
    if (h5py.h5fd.MPIO > 0):
        # Use HDF5 MPI support if available
        kwargs = {'driver' : 'mpio', 'comm' : decomp.comm}
        use_MPI = True
    elif (decomp.size > 1):
        raise Exception('Data.py: HDF5 MPI support is required for nproc>1')

    with h5py.File(*args,**kwargs) as f:
        # Time info
        Nstart = np.array(f['Ntime'])[0]
        tstart = np.array(f['time'])[0]
        dt     = np.array(f['dt'])[0]
        # Not reading grid info
        #
        # Conserved quantities
        Q_tmp = np.empty((nx_,ny_,nz_), dtype=decomp.WP_np)

        for name in Q.names:
            f[name].read_direct(Q_tmp, np.s_[imin_:imax_,jmin_:jmax_,kmin_:kmax_])
            
            # Copy to state buffer
            Q[name].copy( torch.from_numpy(Q_tmp).to(decomp.device) )
                     
    return Nstart,tstart,dt


# --------------------------------------------------------------
# Read grid -- HDF5 -- Interface function
# --------------------------------------------------------------
def read_grid(dfName, cfg, decomp, ndim):
    if (not hasattr(cfg,"restart_2D_to_3D")):
        cfg.restart_2D_to_3D = False
        
    # when to use read_grid_2D or read_grid_3D
    #   ndim == 2: 
    #       we are only working in 2D, so we use read_grid_2D
    #   cfg.restart_2D_to_3D:
    #       the grid is stored in 2D and is read as such using read_grid_2D; 
    #       later extruded into the third dimension
    #   cfg.dfName_read is None:
    #       this means we are not restarting, and we currently assume that all
    #       stored grids are 2D, so we use read_grid_2D; if future grids are
    #       generated in 3D and stored as such, this code will need to be
    #       updated
    #   ndim == 3:
    #       the above has not been used, we must be restarting from a 3D 
    #       simulation where the grid is stored in 3D
    if (ndim == 2) or cfg.restart_2D_to_3D or (cfg.dfName_read is None):
        return read_grid_2D(dfName, decomp)
    elif ndim == 3:
        return read_grid_3D(dfName, decomp)
    else:
        raise NotImplementedError
    

# --------------------------------------------------------------
# Read grid -- HDF5 -- 2D
# --------------------------------------------------------------
def read_grid_2D(dfName, decomp):      
    nx_ = decomp.nx_
    ny_ = decomp.ny_
    nz_ = decomp.nz_
    
    imin_ = decomp.imin_loc
    imax_ = decomp.imax_loc+1
    
    jmin_ = decomp.jmin_loc
    jmax_ = decomp.jmax_loc+1
    
    kmin_ = decomp.kmin_loc
    kmax_ = decomp.kmax_loc+1
    
    selection = np.s_[imin_:imax_, jmin_:jmax_]
    
    X = np.empty((nx_, ny_), dtype=decomp.WP_np)
    Y = np.empty((nx_, ny_), dtype=decomp.WP_np)
    
    args = (dfName, "r")
    kwargs = {}
    with h5py.File(*args,**kwargs) as f:
        f["Grid0/X"].read_direct(X, selection)
        f["Grid0/Y"].read_direct(Y, selection)
    
    return (torch.from_numpy(X).to(decomp.device), 
            torch.from_numpy(Y).to(decomp.device)
            )

# --------------------------------------------------------------
# Read data -- HDF5 -- 3D
# --------------------------------------------------------------
def read_grid_3D(dfName, decomp):    
    nx_ = decomp.nx_
    ny_ = decomp.ny_
    nz_ = decomp.nz_
    
    imin_ = decomp.imin_loc
    imax_ = decomp.imax_loc+1
    
    jmin_ = decomp.jmin_loc
    jmax_ = decomp.jmax_loc+1
    
    kmin_ = decomp.kmin_loc
    kmax_ = decomp.kmax_loc+1
    
    selection = np.s_[imin_:imax_, jmin_:jmax_, 0]
    
    X = np.empty((nx_, ny_), dtype=decomp.WP_np)
    Y = np.empty((nx_, ny_), dtype=decomp.WP_np)
    
    args = (dfName, "r")
    kwargs = {}
    with h5py.File(*args,**kwargs) as f:
        f["Grid0/X"].read_direct(X, selection)
        f["Grid0/Y"].read_direct(Y, selection)
    
    return (torch.from_numpy(X).to(decomp.device), 
            torch.from_numpy(Y).to(decomp.device)
            )


# --------------------------------------------------------------
# Write data -- Interface function
# --------------------------------------------------------------
def write_data(cfg,grid,decomp,q_cpu,names,n,t,dt,adjoint=False):
    try:
        if (cfg.grid.ndim==2):
            write_hdf5_2D(cfg,grid,decomp,q_cpu,names,n,t,dt,adjoint)
            write_xdmf_2D(cfg,grid,decomp,names,n,t,adjoint)
            return
        elif (cfg.grid.ndim==3):
            write_hdf5_3D(cfg,grid,decomp,q_cpu,names,n,t,dt,adjoint)
            write_xdmf_3D(cfg,grid,decomp,names,n,t,adjoint)
            return
    except:
        if (decomp.rank==0): print('ERROR: Data.write_data: Could not write restart file.')
        return
        
    
# --------------------------------------------------------------
# Write data -- HDF5 -- 2D
# --------------------------------------------------------------
def write_hdf5_2D(cfg,grid,decomp,q_cpu,names,n,t,dt,adjoint):
    # Grid sizes
    nx  = decomp.nx;  ny  = decomp.ny
    nx_ = decomp.nx_; ny_ = decomp.ny_
    imin_ = decomp.imin_loc; imax_ = decomp.imax_loc+1
    jmin_ = decomp.jmin_loc; jmax_ = decomp.jmax_loc+1

    if names[0] == "u_mean":
        dfName = cfg.outDir+"/Stat_PyFlowCL_{:09d}.h5".format(n)
    elif adjoint:
        dfName = cfg.outDir+"/A_PyFlowCL_{:09d}.h5".format(n)
    else:
        dfName = cfg.outDir+"/PyFlowCL_{:09d}.h5".format(n)
        
    args   = (dfName, 'w')
    kwargs = {}
    use_MPI = False
    
    if (h5py.h5fd.MPIO > 0):
        # Use HDF5 MPI support if available
        kwargs = {'driver' : 'mpio', 'comm' : decomp.comm}
        use_MPI = True
    elif (decomp.size > 1):
        raise Exception('Data.py: HDF5 MPI support is required for nproc>1')

    with h5py.File(*args,**kwargs) as f:
        # Time info
        f.create_dataset("Ntime", (1,), dtype='i', data=n)
        f.create_dataset("time",  (1,), dtype=decomp.WP_np, data=t)
        f.create_dataset("dt",    (1,), dtype=decomp.WP_np, data=dt)

        # Grid
        dset = f.create_dataset("Grid0/X", (nx,ny), dtype=decomp.WP_np)
        if use_MPI:
            with dset.collective:
                dset[imin_:imax_,jmin_:jmax_] = grid.X_cpu.numpy()
        else:
            dset[imin_:imax_,jmin_:jmax_] = grid.X_cpu.numpy()

        dset = f.create_dataset("Grid0/Y", (nx,ny), dtype=decomp.WP_np)
        if use_MPI:
            with dset.collective:
                dset[imin_:imax_,jmin_:jmax_] = grid.Y_cpu.numpy()
        else:
            dset[imin_:imax_,jmin_:jmax_] = grid.Y_cpu.numpy()

        # Data
        for name in names:
            dset = f.create_dataset(name, (nx,ny), dtype=decomp.WP_np)
            if use_MPI:
                with dset.collective:
                    dset[imin_:imax_,jmin_:jmax_] = q_cpu[names.index(name),:,:,0]
            else:
                dset[imin_:imax_,jmin_:jmax_] = q_cpu[names.index(name),:,:,0]

    return

    
# --------------------------------------------------------------
# Write data -- HDF5 -- 3D
# --------------------------------------------------------------
def write_hdf5_3D(cfg,grid,decomp,q_cpu,names,n,t,dt,adjoint):
    # Grid sizes
    nx  = decomp.nx;  ny  = decomp.ny;  nz  = decomp.nz
    nx_ = decomp.nx_; ny_ = decomp.ny_; nz_ = decomp.nz_
    imin_ = decomp.imin_loc; imax_ = decomp.imax_loc+1
    jmin_ = decomp.jmin_loc; jmax_ = decomp.jmax_loc+1
    kmin_ = decomp.kmin_loc; kmax_ = decomp.kmax_loc+1
    
    if adjoint:
        dfName = cfg.outDir+"/A_PyFlowCL_{:09d}.h5".format(n)
    else: 
        dfName = cfg.outDir+"/PyFlowCL_{:09d}.h5".format(n)
        
    args   = (dfName, 'w')
    kwargs = {}
    use_MPI = False
    
    if (h5py.h5fd.MPIO > 0):
        # Use HDF5 MPI support if available
        kwargs = {'driver' : 'mpio', 'comm' : decomp.comm}
        use_MPI = True
    elif (decomp.size > 1):
        raise Exception('Data.py: HDF5 MPI support is required for nproc>1')

    with h5py.File(*args,**kwargs) as f:
        # Time info
        f.create_dataset("Ntime", (1,), dtype='i', data=n)
        f.create_dataset("time",  (1,), dtype=decomp.WP_np, data=t)
        f.create_dataset("dt",    (1,), dtype=decomp.WP_np, data=dt)
        
        # Grid
        dset = f.create_dataset("Grid0/X", (nx,ny,nz), dtype=decomp.WP_np)
        if use_MPI:
            with dset.collective:
                dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.X_cpu
        else:
            dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.X_cpu

        dset = f.create_dataset("Grid0/Y", (nx,ny,nz), dtype=decomp.WP_np)
        if use_MPI:
            with dset.collective:
                dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.Y_cpu
        else:
            dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.Y_cpu

        dset = f.create_dataset("Grid0/Z", (nx,ny,nz), dtype=decomp.WP_np)
        if use_MPI:
            with dset.collective:
                dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.Z_cpu
        else:
            dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = grid.Z_cpu
            
        # Data
        for name in names:
            dset = f.create_dataset(name, (nx,ny,nz), dtype=decomp.WP_np)
            if use_MPI:
                with dset.collective:
                    dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = q_cpu[names.index(name),...]
            else:
                dset[imin_:imax_,jmin_:jmax_,kmin_:kmax_] = q_cpu[names.index(name),...]

    return

    
# --------------------------------------------------------------
# Write XDMF -- 2D
# --------------------------------------------------------------
def write_xdmf_2D(cfg,grid,decomp,names,n,t,adjoint):
    # Only root process writes XDMF
    if (decomp.rank>0): return
    
    # Grid sizes
    Nx = decomp.nx
    Ny = decomp.ny
    N = Nx*Ny

    if names[0] == "u_mean":
        dfName = "Stat_PyFlowCL_{:09d}.h5:/".format(n)
        fName  = cfg.outDir+"/Stat_PyFlowCL_{:09d}.xmf".format(n)
    
    elif adjoint:
        dfName = "A_PyFlowCL_{:09d}.h5:/".format(n)
        fName  = cfg.outDir+"/A_PyFlowCL_{:09d}.xmf".format(n)
    else:
        dfName = "PyFlowCL_{:09d}.h5:/".format(n)        
        fName  = cfg.outDir+"/PyFlowCL_{:09d}.xmf".format(n)
    
    with open(fName, 'w') as f:
        # Header for xml file
        f.write('''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="2.0">
        <Domain>
        ''')
        
        # Grid info
        f.write('''
           <Grid Name="Grid0" GridType="Uniform"> # 
              <Topology TopologyType="2DSMesh" Dimensions="%d %d"/>
              <Geometry GeometryType="X_Y">
                 <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">
                    %s
                 </DataItem>
                 <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">
                    %s
                 </DataItem>
              </Geometry>
              <Time Value="%8.5e" />
        '''%(Nx,Ny, Nx,Ny, dfName+"Grid0/X", Nx,Ny, dfName+"Grid0/Y", t))

        # Scalar attributes
        for name in names:
            f.write('''\n
            <Attribute Name="%s" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF"> 
            %s
            </DataItem>
            </Attribute>\n
            '''%(name, Nx, Ny, dfName+name))

        # End the xml file
        f.write('''
            </Grid>
        </Domain>
        </Xdmf>
        ''')

    return


    
# --------------------------------------------------------------
# Write XDMF -- 3D
# --------------------------------------------------------------
def write_xdmf_3D(cfg,grid,decomp,names,n,t,adjoint):
    # Only root process writes XDMF
    if (decomp.rank>0): return
    
    # Grid sizes
    Nx = decomp.nx
    Ny = decomp.ny
    Nz = decomp.nz
    N = Nx*Ny*Nz

    if adjoint:
        dfName = "A_PyFlowCL_{:09d}.h5:/".format(n)
        fName  = cfg.outDir+"/PyFlowCL_A_{:09d}.xmf".format(n)
        
    else :
        dfName = "PyFlowCL_{:09d}.h5:/".format(n)
        fName  = cfg.outDir+"/PyFlowCL_{:09d}.xmf".format(n)

    with open(fName, 'w') as f:
        # Header for xml file
        f.write('''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="2.0">
        <Domain>
        ''')
        
        # Grid info
        f.write('''
           <Grid Name="Grid0" GridType="Uniform"> # 
              <Topology TopologyType="3DSMesh" Dimensions="%d %d %d"/>
              <Geometry GeometryType="X_Y_Z">
                 <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">
                    %s
                 </DataItem>
                 <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">
                    %s
                 </DataItem>
                 <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">
                    %s
                 </DataItem>
              </Geometry>
              <Time Value="%8.5e" />
        '''%(Nx,Ny,Nz,
             Nx,Ny,Nz, dfName+"Grid0/X",
             Nx,Ny,Nz, dfName+"Grid0/Y",
             Nx,Ny,Nz, dfName+"Grid0/Z", t))

        # Scalar attributes
        for name in names:
            f.write('''\n
            <Attribute Name="%s" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF"> 
            %s
            </DataItem>
            </Attribute>\n
            '''%(name, Nx, Ny, Nz, dfName+name))

        # End the xml file
        f.write('''
            </Grid>
        </Domain>
        </Xdmf>
        ''')

    return
    
