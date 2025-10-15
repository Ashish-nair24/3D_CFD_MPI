#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:52:49 2023

@author: anair
"""

import sys
import os, glob
import torch
import numpy as np

# Add PyFlowCL src to Python path
sys.path.append('../../../src')

from PyFlowCL import PyFlowCL_aniso_tf, Grid, Data, Model
from PyFlowCL.Monitor import Monitor, Statistics
from PyFlowCL.Library import CUDA_Util

# ----------------------------------------------------
# User-specified parameters
# ----------------------------------------------------
class inputConfigClass:
    def __init__(self,Nx1,Nx2,dt,Nsteps,resume):

        # Grid size
        self.Nx1 = Nx1  # xi
        self.Nx2 = Nx2  # eta
        self.Nx3 = 1  # z

        # Parallel decomposition
        self.nproc_x = 1
        self.nproc_y = 1
        self.nproc_z = 1

        gridType = 'flat_plate'

        # Initial conditions
        self.IC_opt='flat_plate'


        # Specifying EOS 
        self.EOS_Name = 'Perfect_Gas_Dim'


        # Thermodynamic reference states
        self.gamma = 5/3
        self.Rgas  = 208.13                # kJ/kg-K
        self.Pr    = 2/3               

        # M5
        # self.U0    = 1757.18    # m/s
        # self.T0    = 300                   # K
        # self.rho0  = 1.32525e-05 # kg/m^3
        # self.rhoU0 = self.rho0 * self.U0

        # M2
        # self.U0    = 703.838    # m/s
        # self.T0    = 300                   # K
        # self.rho0  = 1.65431e-05 # kg/m^3
        # self.rhoU0 = self.rho0 * self.U0
        # self.cv = self.Rgas / (self.gamma - 1.0)
        # self.e0    = self.cv * self.T0
        # self.rhoE0 = (self.rho0 * self.e0 + 0.5*(self.rhoU0**2)/self.rho0) 

        # # M3
        # self.U0    = 1054.5   # m/s
        # self.T0    = 300                   # K
        # self.rho0  = 1.6518e-05 # kg/m^3
        
        # M7 - High density 
        # self.U0    = 2457.16   # m/s
        # self.T0    = 300                   # K
        # self.rho0  = 9.89013e-05 # kg/m^3

        # M7 - Low density 
        self.U0    = 2457.16   # m/s
        self.T0    = 300                   # K
        self.rho0  = 1.6518e-05 # kg/m^3
        #self.p0    = 3.73

        # M10
        # self.U0    = 3513.75    # m/s
        # self.T0    = 300                   # K
        # self.rho0  = 1.66372e-05 # kg/m^3

        # M-10 paper recreate
        # self.U0    = 2633    # m/s
        # self.T0    = 200                   # K
        # self.rho0  = 2.4e-05 # kg/m^3

        # self.rhoU0 = self.rho0 * self.U0
        # self.cv    = self.Rgas / (self.gamma - 1.0)
        # self.e0    = self.cv * self.T0
        # self.E0    = (self.e0 + 0.5 * (self.U0**2)) 
        # self.rhoE0 = (self.rho0 * self.e0 + 0.5*(self.rhoU0**2)/self.rho0) 
        # self.a0    = np.sqrt(self.gamma * self.Rgas * self.T0)


        # Viscocity estimate 
        m = 6.6335209e-26                 # kg/molecule
        k = 1.38054e-23                   # J/K
        self.Tref = 1000#273.15#                    # K
        dref = 4.04e-10                  # m
        self.omega = 0.734                     
        
        # multiplication factor 
        mulFac = 1.0# 200/500

        self.mu = 50.7e-6#mulFac * ((15 * np.sqrt(2 * np.pi * m * k * self.Tref)) / (2 * np.pi * (dref)**2 * (5 - 2*self.omega) * (7 - 2*self.omega)))#2.117e-5#mulFac * ((15 * np.sqrt(2 * np.pi * m * k * self.Tref)) / (2 * np.pi * (dref)**2 * (5 - 2*self.omega) * (7 - 2*self.omega))) # 2.117e-5#

        self.L0 = 1.0

        # wall-start
        self.wallLs = 0.55 #0.0#     # distance at which wall starts 

        # Thermodynamic parameters
        # self.gamma = 1.4

        # StegerWarming 
        self.advection_scheme = 'upwind_StegerWarming'

        # Low-pass filtering frequency
        self.Nsteps_filter = None
        # self.explicit_filter = True
        # self.max_CFL = 0.85

        # Avection scheme
        #self.advection_scheme = 'upwind_1st'
        
        # Artificial diffusivity (shock capturing)
        self.artDiss = False
        
        # Adiabatic BC
        self.Adiabatic = False

        # Absorbing boundary condition
        self.BC_thickness = 0.1
        self.BC_strength  = 0.0
        self.BC_order     = 3

        # Stopping condition
        self.dt     = dt #/ 10
        self.Nsteps = Nsteps
        self.N_monitor = 1e3 #5

        # Output options
        #self.outDir = '/data/wallModels/M7_scalar/Output_flat_plate_2D_Nx1_{}_Nx2_{}_M7_train_sc_wm_gpuTest_newInps'.format(self.Nx1,self.Nx2)
        self.outDir = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_{}_Nx2_{}_M7_train_aniso_tf'.format(self.Nx1,self.Nx2)
        #self.outDir = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_{}_Nx2_{}_M7_scalar_train_psuedo_try'.format(self.Nx1,self.Nx2)
        os.system('mkdir -p '+self.outDir)

        # Restart file
        if resume:
            # h5files = glob.glob(f"{self.outDir}/*.h5")
            # h5files.sort(reverse=True)
            # self.dfName_read = h5files[0]
            #self.dfName_read = '/data/wallModels/M7_scalar/Output_flat_plate_2D_Nx1_{}_Nx2_{}_M7_train_sc_wm_gpuTest_newInps/PyFlowCL_000647000.h5'.format(self.Nx1,self.Nx2)
            #self.dfName_read = '/data/PyFlow/outOfSamp/M3/Output_flat_plate_2D_Nx1_{}_Nx2_{}_M3_infer_slip_noModel/PyFlowCL_000209000.h5'.format(self.Nx1,self.Nx2)
            #self.dfName_read  = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_withWall_restTrained/PyFlowCL_004624000.h5'
            #self.dfName_read  = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_withWall/PyFlowCL_003217000.h5'
            #self.dfName_read  = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_aniso_2_rest/PyFlowCL_000843000.h5'
            self.dfName_read  = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M3_dist_wall_scalar_trans_model_aPost_BCFix_aniso_infer/PyFlowCL_000099000.h5'
            
        else:
            self.dfName_read = None

        # Compute device
        
        #self.device = torch.device('cpu')
        self.device = CUDA_Util.get_device()
        self.WP = torch.float64
        self.WP_np = np.float64
        
        # Solver Model
        self.solver_mode = 'steady_Newton'#'steady_adjoint_RK4'#; self.dt *= 5
        #fac = 100
        #self.dt /= fac
        #self.N_monitor *= 10

        # Training and model parameters
        self.Train = True
        self.Nsteps_Optim = int(6.5e4)

        # Implicit adjoint solver 
        self.Nsteps_imp_adj = 1000000

        # Explicit adjoint solver 
        self.Nsteps_exp_adj = 3500


        # Type of loss
        self.lossVar= 'U'
        self.model_type = 'viscNew'
        self.theta_init = 1.0

        self.H  = 26

        self.LR      = 5.0e-4
        self.C_out   = 1e0#1.0e0


        # Parameters for inference
        self.Use_Model  = True
        self.Load_Model = False
        #self.modelName_read = 'slipRHSCheck/Output_flat_plate_2D_Nx1_{}_Nx2_{}_testSlip_edConsts_M5DirichTrainOldInps'.format(self.Nx1,self.Nx2) + '/saveMod_Init' #  'saveMod' 
        self.modelName_read = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_aniso_2_rest' + '/saveMod_11'
        self.optimizerName_read = '/data/PyFlow/viscModels/Scalar/Output_flat_plate_2D_Nx1_256_Nx2_256_M7_train_aniso_2_rest' + '/saveOpt_11' #   'saveOpt'


        # Saving model     
        self.Save_Model = True
        self.restNum = 0
        self.modelName_save = self.outDir + '/saveMod'
        self.optimizerName_save = self.outDir + '/saveOpt'

        # Domain to optimize over 
        # self.xLeftOpt  = 0.45
        # self.xRightOpt = 0.9
        # self.yBotOpt   = 0.0
        # self.yTopOpt   = 0.35
        
        self.xLeftOpt  = 0.0
        self.xRightOpt = 1.5
        self.yBotOpt   = 0.0
        self.yTopOpt   = 0.75



        # --------------------------------------------------------------
        # Grids
        if (gridType == 'flat_plate'):
            self.Lx1 = 1.5
            self.Lx2 = 0.5
            self.Lx3 = 0.0
            self.grid = Grid.flat_plate(self.device,self.Nx1,self.Nx2,self.Nx3,
                                        self.Lx1,self.Lx2,self.Lx3,self.wallLs,stretching=True,sx=1.0,sy=1.0, xLeftOpt=self.xLeftOpt, xRightOpt=self.xRightOpt, yBotOpt=self.yBotOpt, yTopOpt=self.yTopOpt)
            
        elif (gridType == 'flat_plate_2'):
            self.Lx1 = 1.5
            self.Lx2 = 0.5
            self.Lx3 = 0.0
            self.grid = Grid.flat_plate_2(self.device,self.Nx1,self.Nx2,self.Nx3,
                                          'wall', 'supersonic', 'supersonic', 'supersonic',
                                          cut_ind=32 )

        else:
            raise Exception('Grid type '+gridType+' not recognized')



    # Function to read target data 
    def load_target_data(self, EOS, names, decomp, cfg, grid, param, comms):


        # DSMC DATA
        # DSMCfilePath = 'M7Tests/dsmcDat/M7_256_256_higherDensity.csv'
        DSMCfilePath = '../dsmcDat/lowDensity/filtered/M7_256_256_lowerDensity_filt.npy'
        
        # Extracting full array 
        # fullArray = np.genfromtxt(DSMCfilePath, delimiter=',', dtype=None, encoding=None)
        # fullArray = np.array(fullArray[1:,:], dtype=np.float64)    
        
        fullArray = (np.load(DSMCfilePath))

        # free-stream constants
        # rho_inf =  4.948292209051355e-06
        # u_inf = 1756.1372531052273
        # T_inf = 300
        # # P_inf = 300
        # # Re = 132.04039497031783 
        # # Ma = 5.939404
        # # gamma = 1.667

        # Extracting and normalizing DSMC data
        # rho_T = np.reshape(fullArray[:,1], (self.Nx1,self.Nx2))
        # U_T   = np.reshape(fullArray[:,2], (self.Nx1,self.Nx2))
        # V_T   = np.reshape(fullArray[:,3], (self.Nx1,self.Nx2)) 
        # T_T = np.reshape(fullArray[:,8], (self.Nx1,self.Nx2)) 
        
        rho_T = fullArray[:,:,0]
        U_T   = fullArray[:,:,1]
        V_T   = fullArray[:,:,2]
        T_T   = fullArray[:,:,3]

        # Adding 3rd dimension
        rho_T = rho_T[:,:,None]
        U_T = U_T[:,:,None]
        V_T = V_T[:,:,None]
        T_T = T_T[:,:,None]

        # # # computing internal energy 
        # # e_T = EOS.get_internal_energy_TY(T_T)

        # # # Computing conserved quantities
        # # rhoU_T = rho_T * U_T
        # # rhoV_T = rho_T * V_T
        # # rhoE_T = e_T + 0.5*(rhoU_T**2 + rhoV_T**2)

        # Defining target variable
        names = ['rho','U','V','T']
        Q_T = Data.State(names, decomp)

        for var in names:
            Q_T[var] = Data.PCL_Var(decomp, var)

        # # Copying DSMC data into Data.State
        Q_T['rho'].copy(torch.Tensor(rho_T).type(self.WP))
        Q_T['U'].copy(torch.Tensor(U_T).type(self.WP))
        Q_T['V'].copy(torch.Tensor(V_T).type(self.WP))
        Q_T['T'].copy(torch.Tensor(T_T).type(self.WP))


        # Writing target data for check
        # monitor = Monitor.PCL_Monitor(cfg,grid,param,comms,decomp)
        # time = monitor.step(cfg,grid,param,EOS,comms,decomp,0,0,1e-3,Q_T,None,None,0,F_res=None)

        # print('Saved Target Data')

        # # Load from a .h5 file 
        # # Restart file contains primitives
        # targetDataPath = 'singleParamTests/Output_flat_plate_2D_Nx1_{}_Nx2_{}_InitConvergemulFac5/PyFlowCL_000000009.h5'.format(self.Nx1, self.Nx2) 
        # # targetDataPath = 'targetData.h5'
        # names = ['rho','U','V','W','T','e']
        # Q_T = Data.State(names, decomp)
        
        # # Load restart file
        # Nstart, t, dt_tmp = Data.read_data(targetDataPath, self, decomp, Q_T)

        # # Convert primitives to conserved
        # for name in names:
        #     if (name=='e'):
        #         # Convert internal energy to total energy
        #         Q_T['rhoE'] = Q_T.pop('e')
        #         Q_T['rhoE'].mul( Q_T['rho'].interior() )
        #         Q_T['rhoE'].add( 0.5*( Q_T['rhoU'].interior()**2 +
        #                                 Q_T['rhoV'].interior()**2 +
        #                                 Q_T['rhoW'].interior()**2 ) / Q_T['rho'].interior() )
        #     elif (name!='rho'):
        #         Q_T['rho'+name] = Q_T.pop(name)
        #         Q_T['rho'+name].mul( Q_T['rho'].interior() )
                
        # names = ['rho','rhoU','rhoV','rhoW','rhoE'] + EOS.sc_names 
        # Q_T.names = names


        # Writing target data for check
        # monitor = Monitor.PCL_Monitor(cfg,grid,param,comms,decomp)
        # time = monitor.step(cfg,grid,param,EOS,comms,decomp,0,0,1e-3,Q_T,None,None,0,F_res=None)

        # Q_T = names.name


        return Q_T

    # ------------------------------
    #   Function to specify model
    # ------------------------------

    def define_model(self):

        # Number of hidden units
        H = self.H

        #number of model inputs
        num_inputs = 8
        # Output factor
        C_out =  self.C_out
        if self.model_type == 'source':
            # Number of model outputs
            num_outputs = 1
            model = Model.NeuralNetworkModel_ELU(H, num_inputs, num_outputs, C_out)
        elif self.model_type == 'visc':
            # Number of model outputs 
            num_outputs = 2
            model = Model.NeuralNetworkModel_ELU(H, num_inputs, num_outputs, C_out)
            
        elif self.model_type == 'viscNew':
            
            num_outputs = 5
            num_inputs_1 = 4
            num_inputs_2 = 4
            model = Model.NeuralNetworkModel_tanh_aniso_tf(H, num_inputs_1, num_inputs_2, num_outputs, C_out)
            
            
        elif self.model_type == 'Q_apriori_Wall':
            
            model = Model.MLP(2, [32, 32, 32], 6)

            
        elif self.model_type == 'constant':
            
            num_outputs   = 1
            initial_theta = torch.tensor(self.theta_init)
            perturb       = 0.0
            model = Model.SingleLayerConstModel(perturb, num_outputs, initial_theta)

        # Names for saved model and optimizer

        # self.modelName_save = self.modelName_read
        # self.optimizerName_save = self.optimizerName_read
        # self.schedulerName_save = self.schedulerName_read

        return model

    def apply_model(self, model, input_dict, grid, metrics, param, EOS ):
        # if self.model_type == 'visc':
        #     tol_mu = 0.0
        #     epsilon = 1e-05
        # elif self.model_type == 'source':
        #     tol_mu = 0.0
        #     epsilon = 0.0
        # #tol_kappa = tol_mu
        # tol_kappa = tol_mu / (self.Pr * self.Ma**2)

        rho        = input_dict[0]
        rhoU       = input_dict[1]
        rhoV       = input_dict[2]
        rhoE       = input_dict[3]
        
        # rho        = input_dict['rho'].var
        # rhoU       = input_dict['rhoU'].var
        # rhoV       = input_dict['rhoV'].var
        # rhoE       = input_dict['rhoE'].var
        

        drho_dx, drho_dy   = metrics.grad_node(rho)[:2]
        # drhoU_dx, drhoU_dy = metrics.grad_node(rhoU)[:2]
        # drhoV_dx, drhoV_dy = metrics.grad_node(rhoV)[:2]
        # drhoE_dx, drhoE_dy = metrics.grad_node(rhoE)[:2]

        rho  = metrics.full2int(rho)
        rhoU = metrics.full2int(rhoU)
        rhoV = metrics.full2int(rhoV)
        rhoE = metrics.full2int(rhoE)

        T, p, e = EOS.get_TPE_tensor(input_dict, interior=True)
        #T, p, e = EOS.get_TPE(input_dict, interior=True)
        
        u = (rhoU / rho)
        v = (rhoV / rho)
        T = metrics.expand_overlaps(T)
        p = metrics.expand_overlaps(p)
        u = metrics.expand_overlaps(u)
        v = metrics.expand_overlaps(v)

        # Velocity gradients - extended interior

        du_dx, du_dy      = metrics.grad_node(u, extended_input= True)[:2]
        dv_dx, dv_dy      = metrics.grad_node(v, extended_input= True)[:2]

        dT_dx,dT_dy       = metrics.grad_node( T, extended_input= True )[:2]
        dp_dx,dp_dy       = metrics.grad_node( p, extended_input= True )[:2]
        u = metrics.ext2int(u)
        #print(torch.amax(u))
        v = metrics.ext2int(v)
        p = metrics.ext2int(p)
        T = metrics.ext2int(T)
        a = torch.sqrt(self.gamma * self.Rgas * T)
        
        # Ma_loc = u / torch.sqrt(p / rho) * self.Ma
        #torch.set_printoptions(profile='full')
        #print(e[6,:])

        # Length scale based on density gradient 
        cs = torch.sqrt(self.gamma * self.Rgas * T) 
        muLoc = self.mu * (T/self.Tref)**self.omega
        lambLoc = (16/5) * (self.gamma/(2 * torch.pi))**(1/5) * (muLoc/(rho * cs))

        # Computing free stream mean-free path
        lambFree = torch.amin(lambLoc) * torch.ones_like(lambLoc)

        # Computing local Mach number 
        MacLoc = u / cs


        # Gradient magnitudes 
        eps = 1e-20
        drho_r  = torch.sqrt(drho_dx**2 + drho_dy**2 + eps) 
        # drhoU_r = torch.sqrt(drhoU_dx**2 + drhoU_dy**2)
        # drhoV_r = torch.sqrt(drhoV_dx**2 + drhoV_dy**2) 
        # drhoE_r = torch.sqrt(drhoE_dx**2 + drhoE_dy**2) 
        dp_r    = torch.sqrt(dp_dx**2 + dp_dy**2 + eps) 
        dT_r    = torch.sqrt(dT_dx**2 + dT_dy**2 + eps) 
        # du_r    = torch.sqrt(du_dx**2 + du_dy**2) 
        # dv_r    = torch.sqrt(dv_dx**2 + dv_dy**2) 

        # Computing symmetric stress tenson 
        e = torch.zeros((self.Nx1,self.Nx2,2,2)).to(self.device)
        e[:,:,0,0] = du_dx[:,:,0] / a[:,:,0]
        e[:,:,0,1] = 0.5 * (du_dy[:,:,0] + dv_dx[:,:,0]) / a[:,:,0]
        e[:,:,1,0] = 0.5 * (du_dy[:,:,0] + dv_dx[:,:,0]) / a[:,:,0]
        e[:,:,1,1] = dv_dy[:,:,0] / a[:,:,0]


        # Computing local Kn number
        gradT = T / dT_r 
        KnLoc = lambLoc / gradT

        # Normalizing by local MFP
        drhoBar = (lambFree * drho_r) / rho
        dpBar   = (lambFree * dp_r) / p
        dTBar   = (lambFree * dT_r) / T
        deBar   = torch.zeros_like(e).to(self.device)
        deBar[:,:,0,0] = lambFree[:,:,0] * e[:,:,0,0]
        deBar[:,:,0,1] = lambFree[:,:,0] * e[:,:,0,1]
        deBar[:,:,1,0] = lambFree[:,:,0] * e[:,:,1,0]
        deBar[:,:,1,1] = lambFree[:,:,0] * e[:,:,1,1] 

        # computing eigen values
        deBarEigs = torch.linalg.eigvals(deBar).real   
        #print(deBarEigs.real.dtype)
        #print(self.lol)     

        # rho = rho / 5
        T   = T / 656.2000#500
        rho = rho/ 2.6119e-05
        p   = p / 3.5079
        
        # drho_r =  drho_r / 500
        # du_r    = du_r / 100
        # dv_r    = dv_r / 50
        # dT_r    = dT_r / 1200
        # Ma_loc  = Ma_loc / 15

        # # Normalizing
        # drho_r  = drho_r / torch.amax(drho_r)
        # du_r    = du_r / torch.amax(du_r)
        # dv_r    = dv_r / torch.amax(dv_r)
        # dT_r    = dT_r / torch.amax(dT_r)

        # # Normalizing individual direction gradients
        # drho_dx  = drho_dx / torch.amax(drho_dx)
        # drho_dy  = drho_dy / torch.amax(drho_dy)
        # du_dx     = du_dx / torch.amax(du_dx)
        # #print(torch.amax(torch.abs(du_dx)))
        # du_dy     = du_dy / torch.amax(du_dy)
        # #print(torch.amax(torch.abs(du_dy)))
        # #print(self.lol)
        # dv_dx    = dv_dx / torch.amax(dv_dx)
        # dv_dy    = dv_dy / torch.amax(dv_dy)
        # dT_dx    = dT_dx / 45726.8622#torch.amax(dT_dx)
        # dT_dy    = dT_dy / 11529.5786#torch.amax(dT_dy)


        # print(torch.max(drho_r), torch.max(du_r), torch.max(dv_r), torch.max(dT_r))        

        """ drho_r_max = torch.max(torch.abs(drho_r))
        du_r_max = torch.max(torch.abs(du_r))
        dv_r_max = torch.max(torch.abs(dv_r))
        dT_r_max = torch.max(torch.abs(dT_r))
        rho_max = torch.max(torch.abs(rho))
        T_max = torch.max(torch.abs(T))
        Ma_loc_max = torch.max(torch.abs(Ma_loc))

        print(rho_max, T_max, drho_r_max, du_r_max, dv_r_max, dT_r_max, Ma_loc_max) """
        # model_inputs = torch.stack( ( rho, T,
        #                             drho_r, du_r, dv_r,
        #                             dT_r,
        #                             Ma_loc
        #                               ), dim = -1)


        # model_inputs = torch.stack( (drho_r, du_r, dv_r,
        #                             dT_r), dim = -1)

        #model_inputs = torch.stack( (T, dT_dx, dT_dy), dim = -1)
        
        #model_inputs = torch.stack( (drhoBar, dpBar, dTBar, deBarEigs[:,:,0,None], deBarEigs[:,:,1,None], rho, p, T), dim = -1)

        model_inputs_1 = torch.stack( (drhoBar, dTBar, deBarEigs[:,:,0,None], deBarEigs[:,:,1,None]), dim = -1)
        model_inputs_2 = torch.stack( (rho, T, MacLoc, KnLoc), dim = -1) 

        #model_inputs = T
        #model_inputs_xy = model_inputs[:,:,:,None]

        # model_inputs_xy = torch.stack( (drho_dx, drho_dy, du_dx, du_dy, dv_dx, dv_dy,
        #                             dT_dx, dT_dy), dim = -1)

        # model_inputs_xplusy  = torch.concat((model_inputs_xy[1:, :, :, :], model_inputs_xy[:1, :, :, :]), dim=0)
        # model_inputs_xminusy = torch.concat((model_inputs_xy[:-1, :, :, :], model_inputs_xy[-1:, :, :, :]), dim=0)
        # model_inputs_xyplus  = torch.concat((model_inputs_xy[:, 1:, :, :], model_inputs_xy[:, :1, :, :]), dim=1)
        # model_inputs_xyminus = torch.concat((model_inputs_xy[:, :-1, :, :], model_inputs_xy[:, -1:, :, :]), dim=1)
        
        # model_inputs = torch.concat( (model_inputs_xy, model_inputs_xminusy, model_inputs_xplusy, model_inputs_xyminus, model_inputs_xyplus), dim = -1)



        # Closure model output
        if (self.model_type == 'source' or self.model_type == 'visc'):
            model_outputs = model(model_inputs_1)
            
        elif (self.model_type == 'viscNew'):
            model_outputs = model(model_inputs_1, model_inputs_2)
        else: 
            model_outputs = model()
        # print(model_outputs.shape)
        if self.model_type == 'source':
            return model_outputs[:,:,:,0]
        #model_outputs = model(torch.ones_like(rho[0,0,:]))
        # mo1 = model_outputs[:,:,:,0]
        # mo2 = model_outputs[:,:,:,1]       
        # # Applying model
        # mu_NN = mo1  + tol_mu
        # kappa_NN = mo2 + tol_kappa
        # #kappa_NN = 0.0 * kappa_NN         
        # #mu_NN = model_outputs + tol_mu
        # #kappa_NN = 0.0 * mu_NN            
        # # print(torch.amax(torch.abs(mu_NN)), torch.amax(torch.abs(kappa_NN)))
        # #return torch.stack( (mu_NN[:,:,None],kappa_NN[:,:,None]),dim = 0)
        return model_outputs#torch.stack( (mu_NN, kappa_NN), dim = -1)

    def loss(self, comms, grid, param, metrics, EOS, Q, Q_T, model_outputs = None):

        rho = Q['rho'].interior()#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ]
        rhoU =  Q['rhoU'].interior()
        rhoV = Q['rhoV'].interior()
        u   = Q['rhoU'].interior() / rho#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ] / rho
        v   = Q['rhoV'].interior() / rho#[cutoff_xi:-cutoff_xi, cutoff+_eta:-cutoff_eta, : ] / rho
        T, p, e = EOS.get_TPE(Q, interior = True)#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ]
        rhoE =  Q['rhoE'].interior()
        E    = rhoE / rho

        # rho_T = Q_T['rho'].interior()#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ]
        # rhoU_T = Q_T['rhoU'].interior()
        # rhoV_T = Q_T['rhoV'].interior()
        # rhoE_T = Q_T['rhoE'].interior()
        # u_T   = Q_T['rhoU'].interior() / rho_T#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ] / rho
        # v_T   = Q_T['rhoV'].interior() / rho_T#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ] / rho

        # T_T = EOS.get_T(Q_T)[5:-5,5:-5,:]#[cutoff_xi:-cutoff_xi, cutoff_eta:-cutoff_eta, : ]
        #print(T_T.shape)

        # Directly loading primitives
        rho_T = Q_T['rho'].interior()
        u_T = Q_T['U'].interior()
        v_T = Q_T['V'].interior()
        T_T = Q_T['T'].interior()     

        e_T = EOS.get_internal_energy_TY(T_T) 
        
        rhoU_T = rho_T * u_T
        rhoV_T = rho_T * v_T
        rhoE_T =  (rho_T * e_T + 0.5*(rhoU_T**2 + rhoV_T**2)/rho_T) 
        E_T    = rhoE_T / rho_T
        a_T    = torch.sqrt(self.gamma * self.Rgas * T_T)
        rhoa_T = rho_T * a_T


        if self.lossVar == 'all':
            #print('Correct Loss')
            cutoff_xi = 5
            cutoff_eta = 5

            # J = 0.5 * torch.mean( ((rho - rho_T )**2 
            #     + (rhoU - rhoU_T)**2
            #     #+ (u - u_T)**2
            #     #+ (v - v_T)**2
            #     #+ (T - T_T )**2
            #     )
            #     )
        elif (self.lossVar == 'U'):
            #J = torch.mean(torch.abs(Q['rhoU'].interior() - Q_T['rhoU'].interior())**2) + torch.mean(torch.abs(Q['rho'].interior() - Q_T['rho'].interior())**2)
            #print('Correct Loss')
            #J = 0.5 * (torch.sum((u - u_T)**2)) # + torch.sum((Q['rho'].interior() - Q_T['rho'].interior())**2))
            # J = (0.5/(grid.Nx1 * grid.Nx2)) * (torch.sum((Q['rhoU'].interior() - Q_T['rhoU'].interior())**2) 
            #            + torch.sum((Q['rho'].interior() - Q_T['rho'].interior())**2)
            #            + torch.sum((Q['rhoV'].interior() - Q_T['rhoV'].interior())**2)
            #            + torch.sum((Q['rhoE'].interior() - Q_T['rhoE'].interior())**2))  
            # J =  0.5 * ((1/self.rho0**2) * (1/1.3332) * torch.sum((rho[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop] - rho_T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop])**2)
            #             + (1/self.rhoU0**2) * (1/4.8606) * torch.sum((rhoU[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop] - rhoU_T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop])**2)
            #            + (1/self.rhoE0**2) * (1/3.2736) * torch.sum((rhoE[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop] - rhoE_T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop])**2))
                       #+ torch.sum((v - v_T)**2))
                       #+ torch.sum((Q['rhoE'].interior() - Q_T['rhoE'].interior())**2))  

            # J = (0.5/(grid.Nx1 * grid.Nx2)) * (1/self.rho0**2) *  torch.sum((rho[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop] - rho_T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop])**2)
            #            #+ (1/self.T0**2) * torch.sum((T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop] - T_T[grid.xIndOptLeft:grid.xIndOptRight, grid.yIndOptBot:grid.yIndOptTop])**2))
            #            #+ torch.sum((v - v_T)**2))
            #            #+ torch.sum((Q['rhoE'].interior() - Q_T['rhoE'].interior())**2))  


            # J_1     = (0.5) * torch.sum((rho[1:-1,1:-1,:] - rho_T[1:-1,1:-1,:])**2 * (1/rho_T[1:-1,1:-1,:]**2))  #* (1/2.5205) 
            # J_2     = (0.5) * torch.sum((rhoU[1:-1,1:-1,:] - rhoU_T[1:-1,1:-1,:])**2 * (1/rhoU_T[1:-1,1:-1,:]**2)) #* (1/12.1408)
            # J_3     = (0.5) * torch.sum((rhoV[1:-1,1:-1,:] - rhoV_T[1:-1,1:-1,:])**2 * (1/rhoa_T[1:-1,1:-1,:]**2)) #* (1/1.3683)
            # J_4     = (0.5) * torch.sum((rhoE[1:-1,1:-1,:] - rhoE_T[1:-1,1:-1,:])**2 * (1/rhoE_T[1:-1,1:-1,:]**2)) * (1/10)

            # J_1     = (0.5) * torch.sum((rho[1:-1,1:-1,:] - rho_T[1:-1,1:-1,:])**2 * (1/rho_T[1:-1,1:-1,:]**2))  * (1/2.2709) 
            # J_2     = (0.5) * torch.sum((u[1:-1,1:-1,:] - u_T[1:-1,1:-1,:])**2 * (1/a_T[1:-1,1:-1,:]**2)) * (1/9.2175)
            # J_3     = (0.5) * torch.sum((v[1:-1,1:-1,:] - v_T[1:-1,1:-1,:])**2 * (1/a_T[1:-1,1:-1,:]**2)) * (1/0.8848)
            # J_4     = (0.5) * torch.sum((E[1:-1,1:-1,:] - E_T[1:-1,1:-1,:])**2 * (1/E_T[1:-1,1:-1,:]**2)) * (1/5.3786)

            J_1     = (0.5) * torch.sum((rho[90:200,0:150,:] - rho_T[90:200,0:150,:])**2 * (1/rho_T[90:200,0:150,:]**2))  #* (1/1000) 
            J_2     = (0.5) * torch.sum((u[90:200,0:150,:] - u_T[90:200,0:150,:])**2 * (1/u_T[90:200,0:150,:]**2)) #* (1/1000)
            J_3     = (0.5) * torch.sum((v[90:200,0:150,:] - v_T[90:200,0:150,:])**2 * (1/a_T[90:200,0:150,:]**2)) #* (1/1000)
            J_4     = (0.5) * torch.sum((E[90:200,0:150,:] - E_T[90:200,0:150,:])**2 * (1/E_T[90:200,0:150,:]**2)) #* (1/1000)

            # J_1     = (0.5) * torch.sum((rho[grid.wallPoint:,0,:] - rho_T[grid.wallPoint:,0,:])**2 * (1/self.rho0**2) * (1/100))
            # J_2     = (0.5) * torch.sum((u[grid.wallPoint:,0,:] - u_T[grid.wallPoint:,0,:])**2 * (1/self.U0**2))
            # J_3     = (0.5) * torch.sum((v[grid.wallPoint:,0,:] - v_T[grid.wallPoint:,0,:])**2 * (1/self.a0**2))
            # J_4     = (0.5) * torch.sum((E[grid.wallPoint:,0,:] - E_T[grid.wallPoint:,0,:])**2 * (1/self.E0**2))


            J = J_1 + J_2 + J_3 + J_4
       
            # J = (0.5)  * (torch.sum((rho[1:-1,1:-1,:] - rho_T[1:-1,1:-1,:])**2 * (1/rho_T[1:-1,1:-1,:]**2))  * (1/2.5205)
            #            +  torch.sum((rhoU[1:-1,1:-1,:] - rhoU_T[1:-1,1:-1,:])**2 * (1/rhoa_T[1:-1,1:-1,:]**2) * (1/12.1408))
            #            +  torch.sum((rhoV[1:-1,1:-1,:] - rhoV_T[1:-1,1:-1,:])**2 * (1/rhoa_T[1:-1,1:-1,:]**2)) * (1/1.3683)
            #            +  torch.sum((rhoE[1:-1,1:-1,:] - rhoE_T[1:-1,1:-1,:])**2 * (1/rhoE_T[1:-1,1:-1,:]**2)) * (1/2.7901))
            
            # J_wall = (0.5/(grid.Nx1 * grid.Nx2)) *((1/self.U0**2) * (torch.sum((u[grid.wallPoint:,0] - u_T[grid.wallPoint:,0]))**2) 
            #            + (1/self.rho0**2) *  +torch.sum((rho[grid.wallPoint:,0] - rho_T[grid.wallPoint:,0])**2)
            #            + (1/self.T0**2) * torch.sum((T[grid.wallPoint:,0] - T_T[grid.wallPoint:,0])**2))

            print(J)
            print(J_1)
            print(J_2)
            print(J_3)
            print(J_4)

        else:
            raise Exception('Loss var type' + self.lossVar + 'not implemented')
        return J, J_1, J_2, J_3, J_4

        
def driver(argv):

    # Initial grid size, dt, nsteps
    #Nx1 = 256; Nx2 = 256;  dt = 1e-3; Nsteps = 100000
    #Nx1 = 256; Nx2 = 256;  dt = 1.0e-8; Nsteps = 2000000#2000000
    Nx1 = 256; Nx2 = 256;  dt = 2.0e-08; Nsteps = 20000000 # Re=1

    resume = int(argv[0]) if argv else False
    
    # Generate the input configuration
    inputConfig = inputConfigClass(Nx1,Nx2,dt,Nsteps, resume)

    print('LR : {}'.format(inputConfig.LR))


    # Run it
    PyFlowCL_aniso_tf.run( inputConfig )
    
# END MAIN


if __name__ == "__main__":
    driver(sys.argv[1:])
