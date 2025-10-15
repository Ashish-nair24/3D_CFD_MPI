"""
J. Jacobowitz
MacRTL Research Group Notre Dame
Created: 23 May 2023
Last Edited: 23 May 2023

Reads in simulation data for post-processing.
"""
import sys
import os
import glob
import numpy as np
import torch
import inspect

# Add PyFlowCL src to Python path
sys.path.append(os.environ["PYFLOW_SRC"])
from PyFlowCL.PyFlowCL import Param
from PyFlowCL.Library import Parallel
from PyFlowCL import Grid, Data, Metrics, Thermochemistry

class PostProcessingHelper:
    def __init__(self,
                 parent_path: str,
                 outDir: str,
                 driver: str,
                 Nx1: int,
                 Nx2: int,
                 Nx3: int,
                 driver_params: dict):
        self.parent_path = parent_path
        sys.path.append(self.parent_path)
        self.outDir = outDir
        self.driver = driver
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        self.Nx3 = Nx3
        self.driver_params = driver_params

        self.ndim = (self.Nx1 > 0) + (self.Nx2 > 0) + (self.Nx3 > 0)

        # determines Nsteps using the last saved output file
        h5files = glob.glob(f"{self.parent_path}/{self.outDir}/*.h5")
        h5files.sort(reverse=True)
        self.dfName_read = h5files[0]

        Nsteps = int(self.dfName_read.split("_")[-1].strip(".h5"))
        dt = 1      # value doesn't matter

        mod = __import__(driver.strip(".py"))

        self.cfg = mod.inputConfigClass(self.Nx1,
                                        self.Nx2,
                                        self.Nx3,
                                        dt,
                                        Nsteps,
                                        **self.driver_params)
        
        if self.cfg.nproc_x*self.cfg.nproc_y*self.cfg.nproc_z > 1:
            print("This code is untested for use in parallel. If you try it, "
                  "please fix it so it works and parallel and remove this "
                  "warning.")

        # remove the directory created during the class initialization
        # but this seems to remove the folder in the directory it's run in
        # so this should probably be removed
        # os.system(f"rm -rf {self.outDir}")

        self.grid = Grid.enforce_periodic(self.cfg)

        # Set working precision
        self.WP = torch.float64
        self.WP_np = np.float64
        if hasattr(self.cfg, "WP"):
            self.WP = self.cfg.WP
        if hasattr(self.cfg, "WP_np"):
            self.WP_np = self.cfg.WP_np

        self.comms = Parallel.Comms()
        self.decomp = Parallel.Decomp(self.cfg, self.grid, self.WP, self.WP_np)
        self.metrics = Metrics.central_4th_periodicRectZ(self.grid,
                                                         self.decomp)
        Grid.initialize_transforms(self.cfg,
                                   self.grid,
                                   self.decomp,
                                   self.metrics)

        # Initialize thermochemical equations of state
        if hasattr(self.cfg, "EOS_Name"):
            self.EOS = None
            for name, obj in inspect.getmembers(Thermochemistry,
                                                inspect.isclass):
                if name == self.cfg.EOS_Name:
                    self.EOS = obj(self.cfg)
            if self.EOS is None:
                raise Exception(
                    "PyFlowCL.py: EOS Name not found in Thermochemistry.py")
        else:
            # Default EOS is dimensionless calorically perfect gas
            self.EOS = Thermochemistry.Perfect_Gas_Nondim(self.cfg)
            if (self.comms.rank == 0):
                print("Defaulting to dimensionless CPG EOS")

        # Extract parameters from the input config
        self.param = Param(self.cfg,
                           self.decomp,
                           self.EOS,
                           self.WP,
                           self.WP_np)

        if self.param.RANS:
            aux_names = ["rhok", "rhoeps"]
        else:
            aux_names = []

        
        names = (["rho", "U", "V", "W", "e"] 
                 + self.EOS.sc_names_prim 
                 + aux_names)        
        self.Q = Data.State(names, self.decomp)
        
        # load data file
        Data.read_data(self.dfName_read, self.cfg, self.decomp, self.Q)

        # read primitives from the data file
        self.rho = self.Q["rho"].interior()
        self.u = self.Q["U"].interior()
        self.v = self.Q["V"].interior()
        self.w = self.Q["W"].interior()

        # convert primitives to conserved
        for name in names:
            if (name == "e"):
                # Convert internal energy to total energy
                self.Q["rhoE"] = self.Q.pop("e")
                self.Q["rhoE"].mul(self.Q["rho"].interior())
                ke = 0.5*(self.Q["rhoU"].interior()**2 +
                          self.Q["rhoV"].interior()**2 +
                          self.Q["rhoW"].interior()**2)/self.Q["rho"].interior()
                self.Q["rhoE"].add(ke)
            elif (name != "rho"):
                self.Q["rho"+name] = self.Q.pop(name)
                self.Q["rho"+name].mul(self.Q["rho"].interior())

        names = (["rho", "rhoU", "rhoV", "rhoW", "rhoE"] 
                 + self.EOS.sc_names 
                 + aux_names)
        self.Q.names = names
        
        self.T, self.p, self.e = self.EOS.get_TPE(self.Q, interior=True)
        
        # some useful slice objects
        self.top_indx = np.s_[:, -1]
        self.bottom_indx = np.s_[:, 0]
        self.left_indx = np.s_[0, :]
        self.right_indx = np.s_[-1, :]
        self.i = self.j = np.s_[1:-1]           # interior (individual)
        self.ij = np.s_[self.i, self.j]         # interior (combined)
