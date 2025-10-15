"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Conversion.py
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
import h5py
from . import dataReader


# --------------------------------------------------------------
# Read NGA binary restart data and convert to PyFlowCL HDF5
# --------------------------------------------------------------
def NGA_to_PyFlow(dfNameRead,dfNameWrite,Ma,rho_in=1.0,gamma=1.4):

    # Read NGA data
    nx,ny,nz,names_in,time,dt,data_in = dataReader.readNGArestart(dfNameRead)

    # Convert to PyFlowCL conserved variables
    data_out  = []
    names_out = []
    # density
    if ('rho' in names_in):
        rho = data_in[:,:,:,names_in.index('RHO')]
    else:
        rho = rho_in * np.ones_like(data_in[:,:,:,names_in.index('U')])
    data_out.append(rho); names_out.append('rho')
        
    # velocity -> momentum
    rhoU = rho * data_in[:,:,:,names_in.index('U')]
    rhoV = rho * data_in[:,:,:,names_in.index('V')]
    rhoW = rho * data_in[:,:,:,names_in.index('W')]
    data_out.append(rhoU); names_out.append('rhoU')
    data_out.append(rhoV); names_out.append('rhoV')
    data_out.append(rhoW); names_out.append('rhoW')


    # Internal energy from pressure
    #   p_infty = 1/gamma
    p = 1.0 / gamma
    e = p / ((gamma-1.0) * rho * Ma**2)
    
    # Total energy
    rhoE = rho*e + 0.5*(rhoU**2 + rhoV**2 + rhoW**2)/rho
    data_out.append(rhoE); names_out.append('rhoE')

    # Scalars -- address when needed

    # Write to HDF5
    with h5py.File(dfNameWrite,'w') as f:
        # Time info
        f.create_dataset("Ntime", (1,), dtype='i', data=0)
        f.create_dataset("time",  (1,), dtype=np.float64, data=time)
        f.create_dataset("dt",    (1,), dtype=np.float64, data=dt)
        
        # Not writing grid -- not needed for restarts

        # Data
        for name,data in zip(names_out,data_out):
            dset = f.create_dataset(name, (nx,ny,nz), dtype=np.float64, data=data)

            
# --------------------------------------------------------------
# Change Mach number via internal energy
# --------------------------------------------------------------
def change_Mach(dfNameRead,dfNameWrite,Ma_in,Ma_out,gamma=1.4):

    # Read data
    data = dict()
    with h5py.File(dfNameRead,'r') as f:
        for key in f.keys():
            if 'Res' not in key and 'Grid' not in key:
                data[key] = f[key][:]

    rho = data['rho']
    e   = data['e']
    
    # Pressure
    p = (gamma - 1.0) * rho * e * Ma_in**2

    # New internal energy
    e_new =  p / ((gamma-1.0) * rho * Ma_out**2)
    data['e'] = e_new

    # Write data
    with h5py.File(dfNameWrite, 'w') as f:

        # Not writing grid

        # Data
        for key in data.keys():
            dset = f.create_dataset(key, data[key].shape, dtype=data[key].dtype,
                                    data=data[key])

            
# --------------------------------------------------------------
# Adjust the Mach number mid-simulation
#   --> Only for dimensionless calorically perfect gas EOS
# --------------------------------------------------------------
def change_Mach(q, Ma_in, Ma_out, EOS):
    if EOS.dimensional:
        raise Exception('Conversion.change_Mach: Cannot handle dimensional EOS')

    rho = q['rho'].interior()
    u   = q['rhoU'].interior()
    v   = q['rhoV'].interior()
    w   = q['rhoW'].interior()

    # Kinetic energy
    ek  = 0.5*(u**2 + v**2 + w**2)
    
    # Internal energy
    e   = q['rhoE'].interior()/rho - ek
    
    # Old pressure
    p = (EOS.gamma - 1.0) * rho * e * Ma_in**2

    # New internal energy
    e_new =  p / ((EOS.gamma-1.0) * rho * Ma_out**2)

    # New total energy
    q['rhoE'].copy(rho*(e_new + ek))

    return
