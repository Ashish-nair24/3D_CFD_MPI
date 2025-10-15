"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Monitor.py
@authors Jonathan F. MacArt

"""

__copyright__ = """
Copyright (c) 2022 University of Notre Dame
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

import time, torch
from .. import Data
import pickle
#from . import Monitor_Cylinder

# --------------------------------------------------------------
# Simulation monitor
# --------------------------------------------------------------
class PCL_Monitor:
    def __init__(self,cfg,grid,param,comms,decomp):
        
        # Save data needed by member functions

        # Set up auxiliary monitors

        # Print header for terminal output
        if (param.rank==0):
            fmtStr = '{:6}  {:10}  {:6}  {:9}  {:9}  {:9}'
            fmtLst = ['iter','time','CFL','max ρ','max u','max T']
            if (grid.ndim > 1):
                fmtStr += '  {:9}'
                fmtLst.insert(5,'max v')
            if (grid.ndim > 2):
                fmtStr += '  {:9}'
                fmtLst.insert(6,'max w')

            print(fmtStr.format(*fmtLst))

        # Format string for terminal output
        self.fmtStr = '{:6d}  {:9.4e}  {:5.4f}  {:8.3e}  {:8.3e}  {:8.3e}'
        if (grid.ndim > 1):
            self.fmtStr += '  {:8.3e}'
        if (grid.ndim > 2):
            self.fmtStr += '  {:8.3e}'
        self.fmtStr += ', time={:7.3f}'

        self.data_time_avg = 0.0
        self.data_time_counter = 0
        
        self.F_res_norm = []
        self.timeHist   = []

        return

    
    # --------------------------------------------------------------
    # Step the monitor
    # --------------------------------------------------------------
    def step(self,cfg,grid,param,EOS,comms,decomp,n,t,dt,q_cons,Q_A,Q_T,time1,F_res=None):
        time2 = time.time()

        # Compute primitives
        rho = q_cons['rho'].interior()
        u   = q_cons['rhoU'].interior()/rho
        v   = q_cons['rhoV'].interior()/rho
        w   = q_cons['rhoW'].interior()/rho
        T, p, e = EOS.get_TPE(q_cons, interior=True)
        
        
        # Residuals
        
        if not (F_res is None):
            
            # Implicit 
            Res_rho = abs(F_res[0][5:-5,5:-5,:]/torch.amax(rho))
            Res_rhoU = abs(F_res[1][5:-5,5:-5,:]/torch.amax(q_cons['rhoU'].interior()))
            Res_rhoV = abs(F_res[2][5:-5,5:-5,:]/torch.amax(q_cons['rhoV'].interior()))
            Res_rhoE = abs(F_res[3][5:-5,5:-5,:]/torch.amax(q_cons['rhoE'].interior()))
            self.F_res_norm.append(torch.norm(F_res[:,30:-30,30:-30,:]).cpu().detach().numpy())
            
        
            # RK4
            # Res_rho = abs(F_res[0]/torch.amax(rho))
            # Res_rhoU = abs(F_res[1]/torch.amax(q_cons['rhoU'].interior()))
            # Res_rhoV = abs(F_res[2]/torch.amax(q_cons['rhoV'].interior()))
            # Res_rhoE = abs(F_res[3]/torch.amax(q_cons['rhoE'].interior()))
            # self.F_res_norm.append(torch.norm(F_res[:,20:-20,20:-20,:]).cpu().numpy())
            # self.F_res_norm.append(torch.mean(torch.abs(F_res)))

            self.timeHist.append(time2-time1)
            
            # Writing residual convergence and time hist
            pickle.dump(self.F_res_norm, open(cfg.outDir+"/F_hist.npy", "wb")) 
            pickle.dump(self.timeHist, open(cfg.outDir+"/time_hist.npy", "wb"))
            
            
            


        if (param.RANS == True):
        
            k   = q_cons['rhok'].interior()/rho
            eps   = q_cons['rhoeps'].interior()/rho
            names = ['rho','rhoU','rhoV','rhoW','rhoE','rhok','rhoeps','U','V','W','T','p','k','eps']
            
            q = torch.stack( (q_cons['rho'].interior(),
                        q_cons['rhoU'].interior(),
                        q_cons['rhoV'].interior(),
                        q_cons['rhoW'].interior(),
                        q_cons['rhoE'].interior(),
                        q_cons['rhok'].interior(),
                        q_cons['rhoeps'].interior(),
                        u,
                        v,
                        w,
                        T,
                        p,
                        k,
                        eps), dim=0)
        
        
        else:

            if not (F_res is None):
                # Concatenate data
                names = ['rho','U','V','W','e','T','p','Res_rho','Res_rhoU','Res_rhoV','Res_rhoE']

                q_list = [q_cons['rho'].interior(),
                        u,
                        v,
                        w,
                        e,
                        T,
                        p,
                        Res_rho,
                        Res_rhoU,
                        Res_rhoV,
                        Res_rhoE]
                for sc_name in EOS.sc_names:
                    names.append( sc_name[3:] )
                    q_list.append( q_cons[sc_name].interior()/rho )
        
                q = torch.stack(q_list, dim=0)

                
            else:
                # Concatenate data - save primitives only
                names  = ['rho','U','V','W','e','T','p']
                q_list = [q_cons['rho'].interior(),
                        u,
                        v,
                        w,
                        e,
                        T,
                        p]
                for sc_name in EOS.sc_names:
                    names.append( sc_name[3:] )
                    q_list.append( q_cons[sc_name].interior()/rho )
        
                q = torch.stack(q_list, dim=0)


        if param.Train:
            names += ['rho_A', 'rhoU_A', 'rhoV_A', 'rhoE_A']
            q = torch.cat((q, torch.stack( (Q_A['rho_A'].interior(),
                                           Q_A['rhoU_A'].interior(),
                                           Q_A['rhoV_A'].interior(),
                                           Q_A['rhoE_A'].interior()), dim=0)), dim=0 )

        # Write data (HDF5) and metadata (XMF)
        q_cpu = q.detach().cpu().numpy()
        #time_data_1 = time.time()
        Data.write_data(cfg,grid,decomp,q_cpu,names,n,t,dt)
        

        #data_time = comms.parallel_sum(time.time()-time_data_1)/comms.size
        #self.data_time_avg += data_time
        #self.data_time_counter += 1
        #if decomp.rank==0:
        #    print('Data time  {:6.4f}'.format(data_time))
        #    print('Average    {:6.4f}'.format(self.data_time_avg/float(self.data_time_counter)))

        # Write and plot Cd_hist and Cl_hist 
        if  cfg.IC_opt=='cylinder' and cfg.Cd_flag and (param.rank==0):
            pickle.dump(Cd_hist, open(cfg.outDir+"/Cd_hist.npy", "wb")) 
            pickle.dump(Cdf_hist, open(cfg.outDir+"/Cdf_hist.npy", "wb")) 
            pickle.dump(Cdp_hist, open(cfg.outDir+"/Cdp_hist.npy", "wb")) 
            pickle.dump(Cl_hist, open(cfg.outDir+"/Cl_hist.npy", "wb")) 
            pickle.dump(Clf_hist, open(cfg.outDir+"/Clf_hist.npy", "wb")) 
            pickle.dump(Clp_hist, open(cfg.outDir+"/Clp_hist.npy", "wb")) 
            pickle.dump(t_hist, open(cfg.outDir+"/t_hist.npy", "wb")) 

        # # Write and plot Cd_hist and Cl_hist 
        # if  cfg.IC_opt=='cylinder' and cfg.Cd_flag and (param.rank==0):
        #     pickle.dump(Cd_hist, open(cfg.outDir+"/Cd_hist.npy", "wb")) 
        #     pickle.dump(Cdf_hist, open(cfg.outDir+"/Cdf_hist.npy", "wb")) 
        #     pickle.dump(Cdp_hist, open(cfg.outDir+"/Cdp_hist.npy", "wb")) 
        #     pickle.dump(Cl_hist, open(cfg.outDir+"/Cl_hist.npy", "wb")) 
        #     pickle.dump(Clf_hist, open(cfg.outDir+"/Clf_hist.npy", "wb")) 
        #     pickle.dump(Clp_hist, open(cfg.outDir+"/Clp_hist.npy", "wb")) 
        #     pickle.dump(t_hist, open(cfg.outDir+"/t_hist.npy", "wb")) 

        #     CdCl.plot_CdCl_ls(cfg.outDir+'/',[t_hist, t_hist,t_hist], [Cd_hist, Cdf_hist, Cdp_hist], label_ls=['$C_d$','$C_{df}$', '$C_{dp}$'], dir=[1,0], plot_name='Cd', title='')
        #     CdCl.plot_CdCl_ls(cfg.outDir+'/',[t_hist, t_hist,t_hist], [Cl_hist, Clf_hist, Clp_hist], label_ls=['$C_l$','$C_{lf}$', '$C_{lp}$'], dir=[0,1], plot_name='Cl', title='')

        # Maximums for terminal output
        rho_max = comms.parallel_max( torch.amax(q[names.index('rho'), :,:,:],dim=(0,1,2)).cpu().numpy() )
        u_max   = comms.parallel_max( torch.amax(q[names.index('U'),   :,:,:],dim=(0,1,2)).cpu().numpy() )
        v_max   = comms.parallel_max( torch.amax(q[names.index('V'),   :,:,:],dim=(0,1,2)).cpu().numpy() )
        w_max   = comms.parallel_max( torch.amax(q[names.index('W'),   :,:,:],dim=(0,1,2)).cpu().numpy() )
        T_max   = comms.parallel_max( torch.amax(q[names.index('T'),:,:,:],dim=(0,1,2)).cpu().numpy() )

        # Print to terminal
        if (param.rank==0):
            fmtLst = [n, t, param.CFL, rho_max, u_max]
            if (grid.ndim > 1):
                fmtLst.append(v_max)
            if (grid.ndim > 2):
                fmtLst.append(w_max)
            fmtLst += [T_max, time2-time1]

            print(self.fmtStr.format(*fmtLst))

        return time.time()

    
    # --------------------------------------------------------------
    # Unsteady adjoint monitor
    # --------------------------------------------------------------
    def unsteady_adjoint_step(self, grid, comms, n, t, k1_A):

        # Maxima for terminal output
        rho_resid_max  = comms.parallel_max( torch.amax(k1_A[0,:,:,:], dim=(0,1,2)).cpu().numpy() )
        rhoU_resid_max = comms.parallel_max( torch.amax(k1_A[1,:,:,:], dim=(0,1,2)).cpu().numpy() )
        rhoV_resid_max = comms.parallel_max( torch.amax(k1_A[2,:,:,:], dim=(0,1,2)).cpu().numpy() )
        rhoW_resid_max = comms.parallel_max( torch.amax(k1_A[3,:,:,:], dim=(0,1,2)).cpu().numpy() )
        rhoE_resid_max = comms.parallel_max( torch.amax(k1_A[4,:,:,:], dim=(0,1,2)).cpu().numpy() )

        # Print to terminal
        if (comms.rank==0):
            fmtLst = [n, t, 0.0, rho_resid_max, rhoU_resid_max]
            if (grid.ndim > 1):
                fmtLst.append(rhoV_resid_max)
            if (grid.ndim > 2):
                fmtLst.append(rhoW_resid_max)
            fmtLst.append(rhoE_resid_max)

            print(self.fmtStr[:-14].format(*fmtLst))
