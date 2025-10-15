# ------------------------------------------------------------------------
#
# @file dataReader.py
# Functions to interact with NGA-format binary files
# @author Jonathan F. MacArt
#
# The MIT License (MIT)
# Copyright (c) 2022 University of Notre Dame
#
# Permission is hereby granted, free of charge, to any person 
# obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without 
# restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.
# 
# ------------------------------------------------------------------------

import sys
import numpy as np
import struct
import array as arr
import torch

from mpi4py import MPI


# --------------------------------------------------------
# Read grid data from an NGA config file
# --------------------------------------------------------    
def readNGAconfig(fName):

    with open(fName, 'rb') as f:
        # Read the geometry name
        strLen = 64
        f_Data = f.read(strLen)
        nameStr = struct.unpack(strLen*"c",f_Data)
        configName = ''
        for s in nameStr:
            if (s.isspace())==False:
                configName += s.decode('UTF-8')
                    
        # Read data sizes
        f_Data = f.read(28)
        icyl   = struct.unpack('<i',f_Data[ 0: 4])[0]
        xper   = struct.unpack('<i',f_Data[ 4: 8])[0]
        yper   = struct.unpack('<i',f_Data[ 8:12])[0]
        zper   = struct.unpack('<i',f_Data[12:16])[0]
        nx     = struct.unpack('<i',f_Data[16:20])[0]
        ny     = struct.unpack('<i',f_Data[20:24])[0]
        nz     = struct.unpack('<i',f_Data[24:28])[0]
        
        # Read the grid field
        xGrid = arr.array('d')
        xGrid.fromfile(f, nx+1)
        
        yGrid = arr.array('d')
        yGrid.fromfile(f, ny+1)
        
        zGrid = arr.array('d')
        zGrid.fromfile(f, nz+1)

        # Read the mask - NOT IMPLEMENTED
        
        # Print some file info
        if (MPI.COMM_WORLD.Get_rank()==0):
            print(' ')
            print(' --> Importing grid data from NGA config file')
            print('   Config file name:   {}'.format(fName))
            print('   Configuration name: {}'.format(configName))
            print('   Cylindrical:        {}'.format(icyl))
            print('   xper,yper,zper:     {},{},{}'.format(xper,yper,zper))
            print('   Grid size:         nx={}, ny={}, nz={}'.format(nx,ny,nz))
            print('       xmin, xmax:    {}, {}'.format(min(xGrid),max(xGrid)))
            print('       ymin, ymax:    {}, {}'.format(min(yGrid),max(yGrid)))
            print('       zmin, zmax:    {}, {}'.format(min(zGrid),max(zGrid)))

        # Return the grid data info
        xGridOut = np.frombuffer(xGrid,dtype='f8')
        yGridOut = np.frombuffer(yGrid,dtype='f8')
        zGridOut = np.frombuffer(zGrid,dtype='f8')
        
        return(xGridOut,yGridOut,zGridOut,xper,yper,zper)
    


# --------------------------------------------------------
# Write grid data to an NGA config file
# --------------------------------------------------------    
def writeNGAconfig(fName,configName,xgrid,ygrid,zgrid,xper,yper,zper):

    with open(fName, 'wb') as f:
        # Geometry name
        strLen = 64
        configName = "{:<64}".format(configName)
        for i in range(strLen):
            f.write(struct.pack("c",configName[i].encode('UTF-8')))
                    
        # Write data sizes
        f.write(struct.pack('<i',0))
        f.write(struct.pack('<i',xper))
        f.write(struct.pack('<i',yper))
        f.write(struct.pack('<i',zper))
        f.write(struct.pack('<i',len(xgrid)-1))
        f.write(struct.pack('<i',len(ygrid)-1))
        f.write(struct.pack('<i',len(zgrid)-1))
        
        # Write the grid field
        outData = arr.array('d',xgrid)
        outData.tofile(f)
        
        outData = arr.array('d',ygrid)
        outData.tofile(f)
        
        outData = arr.array('d',zgrid)
        outData.tofile(f)
    


# --------------------------------------------------------
# Read state data from a RESTART format data file
# --------------------------------------------------------    
def readNGArestart(fName,headerOnly=False,printOut=True):

    with open(fName, 'rb') as f:
        # Read data sizes
        f_Data = f.read(16)
        nx     = struct.unpack('<i',f_Data[ 0: 4])[0]
        ny     = struct.unpack('<i',f_Data[ 4: 8])[0]
        nz     = struct.unpack('<i',f_Data[ 8:12])[0]
        nvar   = struct.unpack('<i',f_Data[12:16])[0]
        
        # Read timestep size and simulation time
        f_Data = f.read(16)
        dt     = struct.unpack('<d',f_Data[0:8])[0]
        time   = struct.unpack('<d',f_Data[8:16])[0]
        
        # Read names of the variables in the file
        f_Data = f.read(8*nvar)
        names  = ['']*nvar
        for ivar in range(nvar):
            nameStr = struct.unpack("cccccccc",f_Data[ivar*8:(ivar+1)*8])
            for s in nameStr:
                if (s.isspace())==False:
                    names[ivar] += s.decode('UTF-8')
        
        # Print some file info
        if (MPI.COMM_WORLD.Get_rank()==0 and printOut):
            print(' ')
            print(' --> Importing state data from NGA restart file')
            print('   Data file name:    {}'.format(fName))
            print('   Data file at time: {:10.4E}'.format(time))
            print('   Timestep size:     {:10.4E}'.format(dt))
            print('   Number of vars:    {}'.format(nvar))
            print('   Variables in file: {}'.format(names))
            
        if (not headerOnly):
            # Read data arrays in serial and return the output
            nread = nvar
            #nread = min([31,nvar])
            data   = np.empty([nx,ny,nz,nread])
            for ivar in range(nread):
                inData = arr.array('d')
                inData.fromfile(f, nx*ny*nz)
                data[:,:,:,ivar] = np.frombuffer(inData,dtype='f8').reshape((nx,ny,nz),order='F')
                if (printOut):
                    print('   --> Done reading {}'.format(names[ivar]))

            return(nx,ny,nz,names,time,dt,data)

        else:
            # Just return the grid info and variable names
            # Need to call readNGA_parallel to get data
            return(nx,ny,nz,names,time)



# --------------------------------------------------------
# Write the grid and data to a RESTART format data file
#    Serial operations only; useful for writing the header
#    for files of all sizes and data for small files
# --------------------------------------------------------
def writeNGArestart(fName,Data,headerOnly=True):

    with open(fName, 'wb') as f:
        # Write data sizes
        f.write(struct.pack('<i',Data.nx))
        f.write(struct.pack('<i',Data.ny))
        f.write(struct.pack('<i',Data.nz))
        f.write(struct.pack('<i',Data.nvar))

        # Write timestep size and simulation time
        f.write(struct.pack('<d',Data.dt))
        f.write(struct.pack('<d',Data.time))

        # Write names of the variables in the file
        for name in Data.names:
            outNameStr = name.ljust(8)
            for s in outNameStr:
                f.write(struct.pack("c",s.encode('UTF-8')))

        if (not headerOnly):
            # Write the data arrays
            #  --> NOTE: assumes "Data" is an instance of class state.data_all_CPU
            for ivar in range(Data.nvar):
                outData = arr.array('d',Data.data[ivar].reshape(Data.nx*Data.ny*Data.nz,order='F'))
                outData.tofile(f)
                #print('   --> Wrote {}'.format(Data.names[ivar]))

            print('  --> Wrote data file')

    return
    


# --------------------------------------------------------
# Read data from a restart format file in parallel
#   NOTE: names_all is names from data file header
# --------------------------------------------------------
def readNGArestart_parallel(fName,decomp,names_all,Q):

    # Indices to read
    ivar_read_start = 0
    nvar_read = len(Q.names)
    ## WARNING: This only reads the first nvar_read elements of Q, not necessarily the elements of Q.names

    # Get MPI decomposition info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI data sizes
    nvar_all = len(names_all)
    nx_MOK = decomp.nx
    ny_MOK = decomp.ny
    nz_MOK = decomp.nz
    NVARS_MOK = nvar_all
    WP_MOK  = 8
    str_MOK = 8

    # Cartesian decomposition
    gsizes = [decomp.nx,  decomp.ny,  decomp.nz ]
    lsizes = [decomp.nx_, decomp.ny_, decomp.nz_]
    start  = [decomp.imin_loc, decomp.jmin_loc, decomp.kmin_loc]

    #print("gsizes=[{}, {}, {}]".format(data.nx,data.ny,data.nz))
    #print("lsizes=[{}, {}, {}]".format(data.nx_,data.ny_,data.nz_))
    #print("start =[{}, {}, {}]".format(data.imin_,data.jmin_,data.kmin_))

    # Open the file
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(comm,fName,amode)
    
    # Create the subarray
    subData = MPI.DOUBLE.Create_subarray(
        gsizes, lsizes, start, order=MPI.ORDER_FORTRAN)
    subData.Commit()

    # Allocate the buffer to read
    readBuffer = np.empty(decomp.nx_*decomp.ny_*decomp.nz_, dtype='float64')
        
    # Decide how much data to read
    #  --> Be careful with options 2-4, as data.ivar is automatically
    #      reset to zero. Useful for reading one variable at a time, but
    #      potentially catastrophic if the entire dataset is necessary.
    if (ivar_read_start==None and nvar_read==None):
        # Read all the data
        ivar_start = 0
        ivar_end   = nvar
    elif (ivar_read_start==None):
        # Read a fixed number of fields starting from zero
        ivar_start = 0
        ivar_end   = nvar_read
    elif (nvar_read==None):
        # Read all the data starting from ivar_read_start
        ivar_start = ivar_read_start
        ivar_end   = nvar_all
    else:
        # Read just the specified data range
        ivar_start = ivar_read_start
        ivar_end   = ivar_read_start+nvar_read

    for ivar in range(ivar_start,ivar_end):
        # Set the file view
        var_MOK = ivar
        disp = ( 4*4 + NVARS_MOK*str_MOK + 2*WP_MOK
                 + nx_MOK*ny_MOK*nz_MOK*var_MOK*WP_MOK )
        fh.Set_view(disp, filetype=subData)
        
        # Read the file
        fh.Read_all(readBuffer)

        #print("irank={} ivar={} min={} max={}".format(rank,ivar,np.min(readBuffer),np.max(readBuffer)))

        # Copy the variable to the data buffer
        Q[names_all[ivar]].copy( torch.DoubleTensor( readBuffer.reshape((decomp.nx_,
                                                                         decomp.ny_,
                                                                         decomp.nz_),
                                                                        order='F') ) )
        
        #print("irank={} read ivar={}".format(rank,ivar))

    # Close and return
    subData.Free()
    del subData
    del readBuffer
    fh.Close()

    return  


# --------------------------------------------------------
# Write data to a restart format file in parallel
# --------------------------------------------------------
def writeNGArestart_parallel(fName,data): #,ivar_write_start=None,nvar_write=None):

    # Indices to read
    ivar_write_start = data.ivar_read_start
    nvar_write = data.nvar_read

    # Get MPI decomposition info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI data sizes
    nx_MOK = data.nx
    ny_MOK = data.ny
    nz_MOK = data.nz
    NVARS_MOK = data.nvar
    WP_MOK  = 8
    str_MOK = 8

    # Cartesian decomposition
    gsizes = [data.nx,  data.ny,  data.nz ]
    lsizes = [data.nx_, data.ny_, data.nz_]
    start  = [data.imin_loc, data.jmin_loc, data.kmin_loc]

    # Open the file
    amode = MPI.MODE_WRONLY
    fh = MPI.File.Open(comm,fName,amode)
    
    # Create the subarray
    subData = MPI.DOUBLE.Create_subarray(
        gsizes, lsizes, start, order=MPI.ORDER_FORTRAN)
    subData.Commit()

    # Allocate the buffer to write
    writeBuffer = np.empty(data.nx_*data.ny_*data.nz_, dtype='float64')

    # Decide how much data to write
    if (ivar_write_start==None and nvar_write==None):
        # Write all the data
        ivar_start = 0
        ivar_end   = data.nvar
    elif (ivar_write_start==None):
        # Write a fixed number of fields starting from zero
        ivar_start = 0
        ivar_end   = nvar_write
    elif (nvar_write==None):
        # Write all the data starting from ivar_write_start
        ivar_start = ivar_write_start
        ivar_end   = data.nvar
    else:
        # Write just the specified data range
        ivar_start = ivar_write_start
        ivar_end   = ivar_write_start+nvar_write

    # Offset for position in data.data
    ivar_offs = ivar_start
    
    for ivar in range(ivar_start,ivar_end):
        # Set the file view
        var_MOK = ivar
        disp = ( 4*4 + NVARS_MOK*str_MOK + 2*WP_MOK
                 + nx_MOK*ny_MOK*nz_MOK*var_MOK*WP_MOK )
        fh.Set_view(disp, filetype=subData)

        # Fill the buffer
        writeBuffer = data.read(ivar-ivar_offs).reshape(
            data.nx_*data.ny_*data.nz_,order='F').astype('float64')

        #print("irank={} ivar={} min={} max={}".format(
        #    rank,ivar,np.min(writeBuffer),np.max(writeBuffer)))
        
        # Write to the file
        fh.Write_all(writeBuffer)
        
        #print("irank={} wrote ivar={}".format(rank,ivar))

    # Close and return
    subData.Free()
    del subData
    del writeBuffer
    fh.Close()

    return




# --------------------------------------------------------
# Read the grid and data from a VOLUME format data file
# --------------------------------------------------------
def readNGA(fName,readData=True):

    f = open(fName, 'rb')
    
    # Read data sizes
    f_Data = f.read(20)
    ntime  = struct.unpack('<i',f_Data[ 0: 4])[0]
    nx     = struct.unpack('<i',f_Data[ 4: 8])[0]
    ny     = struct.unpack('<i',f_Data[ 8:12])[0]
    nz     = struct.unpack('<i',f_Data[12:16])[0]
    nvar   = struct.unpack('<i',f_Data[16:20])[0]
    
    # Read grid coordinates in x, y, z
    xGrid = arr.array('d')
    xGrid.fromfile(f, nx)
    
    yGrid = arr.array('d')
    yGrid.fromfile(f, ny)
    
    zGrid = arr.array('d')
    zGrid.fromfile(f, nz)
    
    # Read names of the variables in the file
    f_Data = f.read(8*nvar)
    names  = ['']*nvar
    for ivar in range(nvar):
        nameStr = struct.unpack("cccccccc",f_Data[ivar*8:(ivar+1)*8])
        for s in nameStr:
            if (s.isspace())==False:
                names[ivar] += s.decode('UTF-8')
    
    # Read timestep size and simulation time
    f_Data = f.read(16)
    dt     = struct.unpack('<d',f_Data[0:8])[0]
    time   = struct.unpack('<d',f_Data[8:16])[0]
        
    # Print some file info
    if (0):
        print(' ')
        print(fName)
        print('   Grid size:         nx={}, ny={}, nz={}'.format(nx,ny,nz))
        print('       xmin, xmax:    {}, {}'.format(min(xGrid),max(xGrid)))
        print('       ymin, ymax:    {}, {}'.format(min(yGrid),max(yGrid)))
        print('       zmin, zmax:    {}, {}'.format(min(zGrid),max(zGrid)))
        print('   Data file at time: {}'.format(time))
        print('   Timestep size:     {}'.format(dt))
        print('   Number of vars:    {}'.format(nvar))
        print('   Variables in file: {}'.format(names))
    
    xGridOut = np.frombuffer(xGrid,dtype='f8')
    yGridOut = np.frombuffer(yGrid,dtype='f8')
    zGridOut = np.frombuffer(zGrid,dtype='f8')

    if readData:
        # Read data arrays in serial and return the output
        nread = nvar
        #nread = min([31,nvar])
        data   = np.empty([nx,ny,nz,nread])
        for ivar in range(nread):
            inData = arr.array('d')
            inData.fromfile(f, nx*ny*nz)
            data[:,:,:,ivar] = np.frombuffer(inData,dtype='f8').reshape((nx,ny,nz),order='F')
            #print('   --> Done reading {}'.format(names[ivar]))
            
        #print(data.shape)
        #plotData(data[:,:,int(nz/2),0],"test")

        return(xGridOut,yGridOut,zGridOut,names,time,data)

    else:
        # Just return the grid info and variable names
        # Need to call readNGA_parallel to get data
        return(xGridOut,yGridOut,zGridOut,names,time)



# --------------------------------------------------------
# Write the grid and data to a VOLUME format data file
#    Serial operations only; useful for writing the header
#    for files of all sizes and data for small files
# --------------------------------------------------------
def writeNGA(fName,Data,headerOnly=True):

    with open(fName, 'wb') as f:
        # Write data sizes
        ntime = 1
        f.write(struct.pack('<i',ntime))
        f.write(struct.pack('<i',Data.nx))
        f.write(struct.pack('<i',Data.ny))
        f.write(struct.pack('<i',Data.nz))
        f.write(struct.pack('<i',Data.nvar))

        # Write x,y,z grid coordinates
        xGrid = arr.array('d',Data.xGridIn)
        yGrid = arr.array('d',Data.yGridIn)
        zGrid = arr.array('d',Data.zGridIn)
        xGrid.tofile(f)
        yGrid.tofile(f)
        zGrid.tofile(f)

        # Write names of the variables in the file
        for name in Data.names:
            outNameStr = name.ljust(8)
            for s in outNameStr:
                f.write(struct.pack("c",s.encode('UTF-8')))

        # Write timestep size and simulation time
        dt = 0.0
        f.write(struct.pack('<d',dt))
        f.write(struct.pack('<d',Data.time))

        if (not headerOnly):
            # Write the data arrays as well
            for ivar in range(Data.nvar):
                outData = arr.array('d',Data.data[:,:,:,ivar].reshape(Data.nx*Data.ny*Data.nz,order='F'))
                outData.tofile(f)
                print('   --> Wrote {}'.format(Data.names[ivar]))

    return
    


# --------------------------------------------------------
# Read data from a VOLUME format file in parallel
# --------------------------------------------------------
def readNGA_parallel(fName,data,ivar_read_start=None,nvar_read=None):

    # Get MPI decomposition info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI data sizes
    nx_MOK = data.nx
    ny_MOK = data.ny
    nz_MOK = data.nz
    NVARS_MOK = data.nvar
    WP_MOK  = 8
    str_MOK = 8

    # Cartesian decomposition
    gsizes = [data.nx,  data.ny,  data.nz ]
    lsizes = [data.nx_, data.ny_, data.nz_]
    start  = [data.imin_, data.jmin_, data.kmin_]

    #print("gsizes=[{}, {}, {}]".format(data.nx,data.ny,data.nz))
    #print("lsizes=[{}, {}, {}]".format(data.nx_,data.ny_,data.nz_))
    #print("start =[{}, {}, {}]".format(data.imin_,data.jmin_,data.kmin_))

    # Open the file
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(comm,fName,amode)
    
    # Create the subarray
    subData = MPI.DOUBLE.Create_subarray(
        gsizes, lsizes, start, order=MPI.ORDER_FORTRAN)
    subData.Commit()

    # Allocate the buffer to read
    readBuffer = np.empty(data.nx_*data.ny_*data.nz_, dtype='float64')

    # Reset the data structure's variable counter to zero
    data.ivar = 0
        
    # Decide how much data to read
    #  --> Be careful with options 2-4, as data.ivar is automatically
    #      reset to zero. Useful to reading one variable at a time, but
    #      potentially catastrophic if the entire dataset is necessary.
    if (ivar_read_start==None and nvar_read==None):
        # Read all the data
        ivar_start = 0
        ivar_end   = data.nvar
    elif (ivar_read_start==None):
        # Read a fixed number of fields starting from zero
        ivar_start = 0
        ivar_end   = nvar_read
    elif (nvar_read==None):
        # Read all the data starting from ivar_read_start
        ivar_start = ivar_read_start
        ivar_end   = data.nvar
    else:
        # Read just the specified data range
        ivar_start = ivar_read_start
        ivar_end   = ivar_read_start+nvar_read

    for ivar in range(ivar_start,ivar_end):
        # Set the file view
        var_MOK = ivar
        disp = ( 5*4 + (nx_MOK+ny_MOK+nz_MOK)*WP_MOK + NVARS_MOK*str_MOK + 2*WP_MOK
                 + nx_MOK*ny_MOK*nz_MOK*var_MOK*WP_MOK )
        fh.Set_view(disp, filetype=subData)
        
        # Read the file
        fh.Read_all(readBuffer)

        # Append variable to the data buffer
        data.add(readBuffer.reshape((data.nx_,data.ny_,data.nz_),order='F'))
        
        #print("irank={} read ivar={}".format(rank,ivar))

    # Close and return
    subData.Free()
    del subData
    del readBuffer
    fh.Close()

    return  


# --------------------------------------------------------
# Write data to a VOLUME format file in parallel
# --------------------------------------------------------
def writeNGA_parallel(fName,data,ivar_write_start=None,nvar_write=None):

    # Get MPI decomposition info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # MPI data sizes
    nx_MOK = data.nx
    ny_MOK = data.ny
    nz_MOK = data.nz
    NVARS_MOK = data.nvar
    WP_MOK  = 8
    str_MOK = 8

    # Cartesian decomposition
    gsizes = [data.nx,  data.ny,  data.nz ]
    lsizes = [data.nx_, data.ny_, data.nz_]
    start  = [data.imin_, data.jmin_, data.kmin_]

    #print("gsizes=[{}, {}, {}]".format(data.nx,data.ny,data.nz))
    #print("lsizes=[{}, {}, {}]".format(data.nx_,data.ny_,data.nz_))
    #print("start =[{}, {}, {}]".format(data.imin_,data.jmin_,data.kmin_))

    # Open the file
    amode = MPI.MODE_WRONLY
    fh = MPI.File.Open(comm,fName,amode)
    
    # Create the subarray
    subData = MPI.DOUBLE.Create_subarray(
        gsizes, lsizes, start, order=MPI.ORDER_FORTRAN)
    subData.Commit()

    # Allocate the buffer to write
    writeBuffer = np.empty(data.nx_*data.ny_*data.nz_, dtype='float64')

    # Decide how much data to write
    if (ivar_write_start==None and nvar_write==None):
        # Write all the data
        ivar_start = 0
        ivar_end   = data.nvar
    elif (ivar_write_start==None):
        # Write a fixed number of fields starting from zero
        ivar_start = 0
        ivar_end   = nvar_write
    elif (nvar_write==None):
        # Write all the data starting from ivar_write_start
        ivar_start = ivar_write_start
        ivar_end   = data.nvar
    else:
        # Write just the specified data range
        ivar_start = ivar_write_start
        ivar_end   = ivar_write_start+nvar_write

    # Offset for position in data.data
    ivar_offs = ivar_start
    
    for ivar in range(ivar_start,ivar_end):
        # Set the file view
        var_MOK = ivar
        disp = ( 5*4 + (nx_MOK+ny_MOK+nz_MOK)*WP_MOK + NVARS_MOK*str_MOK + 2*WP_MOK
                 + nx_MOK*ny_MOK*nz_MOK*var_MOK*WP_MOK )
        fh.Set_view(disp, filetype=subData)

        # Fill the buffer
        writeBuffer = data.data[:,:,:,ivar-ivar_offs].reshape(data.nx_*data.ny_*data.nz_,order='F')
        
        # Write to the file
        fh.Write_all(writeBuffer)
        
        print("irank={} wrote ivar={}".format(rank,ivar))

    # Close and return
    subData.Free()
    del subData
    del writeBuffer
    fh.Close()

    return

