"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file CUDA_Util.py

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

# Find the first available CUDA device and return its Pytorch handle
def get_device():
    if torch.cuda.is_available():
        NGPU = torch.cuda.device_count()
        # Import the CUDA driver
        try:
            import pynvml
        except ModuleNotFoundError:
            print("pynvml module not found, please install pynvml")
            return torch.device('cuda:0')
        try:
            pynvml.nvmlInit()
        except NVMLError_DriverNotLoaded:
            print("cuda driver can't be loaded, is cuda enabled?")
            return torch.device('cuda:0')
        # Loop over the GPUs
        for iGPU in range(NGPU):
            handle = pynvml.nvmlDeviceGetHandleByIndex(iGPU)
            procs  = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if (len(procs)==0):
                device = 'cuda:{}'.format(iGPU)
                print('Using device '+device)
                return torch.device(device)
        # No available GPUs
        print('WARNING: All GPUs busy! Using CPU ... performance might suffer!')

    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    #     print('Using device '+device)
    #     return torch.device(device)
        
    else:
        # CUDA not available
        print('CUDA not available; using CPU')

    # CUDA not available or no free GPUs
    return torch.device('cpu')
