"""
------------------------------------------------------------------------
PyFlowCL: A Python-native, compressible Navier-Stokes solver for
curvilinear grids
------------------------------------------------------------------------

@file Monitor_Cylinder.py
@authors Xuemin Liu and Jonathan F. MacArt

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


class PCL_Monitor_Cylinder:
    def __init__(self):

        # Member data

    # load drag and lift histories (if exist) for cylinder flow -- MOVE ALL CYLINDER STUFF TO Monitor_Cylinder.py
    if cfg.IC_opt=='cylinder':
        if cfg.Cd_flag :
            if  Nstart==0 or (not os.path.exists(cfg.outDir+"/Cdf_hist.npy")):
                Cd_hist = []
                Cl_hist = []
                t_hist  = []
                Cdf_hist = []
                Cdp_hist = []
                Clf_hist = []
                Clp_hist = []

            else:
                Cd_hist = pickle.load( open( cfg.outDir+"/Cd_hist.npy",  "rb" ) )
                Cl_hist = pickle.load( open( cfg.outDir+"/Cl_hist.npy",  "rb" ) )
                t_hist  = pickle.load( open( cfg.outDir+"/t_hist.npy",  "rb" ) )
                Cdf_hist = pickle.load( open( cfg.outDir+"/Cdf_hist.npy",  "rb" ) )
                Cdp_hist = pickle.load( open( cfg.outDir+"/Cdp_hist.npy",  "rb" ) )
                Clf_hist = pickle.load( open( cfg.outDir+"/Clf_hist.npy",  "rb" ) )
                Clp_hist = pickle.load( open( cfg.outDir+"/Clp_hist.npy",  "rb" ) )                    
         
        else:
            Cd_hist = None; Cdf_hist = None; Cdp_hist = None
            Cl_hist = None; Clf_hist = None; Clp_hist = None
            t_hist  = None

# After RK4 update:

            if param.debug:
                param.plot_dir = cfg.outDir
                param.NCd = 1
                Cd, Cdf, Cdp, Cl, Clf, Clp = CdCl.compute_CdCl(comms, grid, param, metrics, Q)
                if (param.rank==0): print('----------- Cd = {:.3f}, Cdf = {:.3f}, Cdp = {:.3f}, Cl = {:.3f}, Clf = {:.3f}, Clp = {:.3f} ----------'.format(Cd, Cdf, Cdp, Cl, Clf, Clp))
                time1 = monitor(n,t,grid,metrics,cfg,param,comms,decomp,Q,time1,Cd_hist,Cdf_hist,Cdp_hist,Cl_hist,Clf_hist,Clp_hist,t_hist,Cd_flag=cfg.Cd_flag)
                        
                lowpass_filter.apply(Q)


                param.NCd = 2
                Cd, Cdf, Cdp, Cl, Clf, Clp = CdCl.compute_CdCl(comms, grid, param, metrics, Q)
                if (param.rank==0): print('----------- Cd = {:.3f}, Cdf = {:.3f}, Cdp = {:.3f}, Cl = {:.3f}, Clf = {:.3f}, Clp = {:.3f} ----------'.format(Cd, Cdf, Cdp, Cl, Clf, Clp))
                time1 = monitor(n,t,grid,metrics,cfg,param,comms,decomp,Q,time1,Cd_hist,Cdf_hist,Cdp_hist,Cl_hist,Clf_hist,Clp_hist,t_hist,Cd_flag=cfg.Cd_flag)

                enforce_walls( Q )
                

                param.NCd = 3
                Cd, Cdf, Cdp, Cl, Clf, Clp = CdCl.compute_CdCl(comms, grid, param, metrics, Q)
                if (param.rank==0): print('----------- Cd = {:.3f}, Cdf = {:.3f}, Cdp = {:.3f}, Cl = {:.3f}, Clf = {:.3f}, Clp = {:.3f} ----------'.format(Cd, Cdf, Cdp, Cl, Clf, Clp))

                time1 = monitor(n,t,grid,metrics,cfg,param,comms,decomp,Q,time1,Cd_hist,Cdf_hist,Cdp_hist,Cl_hist,Clf_hist,Clp_hist,t_hist,Cd_flag=cfg.Cd_flag)


            else:
                if cfg.IC_opt=='cylinder' and cfg.Cd_flag:
                    Cd, Cdf, Cdp, Cl, Clf, Clp = CdCl.compute_CdCl(comms, grid, param, metrics, Q)
                    if (param.rank==0):
                        # print('----------- Cd = {0:.3f}, Cl={1:.3f} ----------'.format(Cd, Cl))
                        Cd_hist.append(Cd)
                        Cdf_hist.append(Cdf)
                        Cdp_hist.append(Cdp)
                        Cl_hist.append(Cl)
                        Clf_hist.append(Clf)
                        Clp_hist.append(Clp)
                        t_hist.append(t)
