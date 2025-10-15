import torch
from .. import Data

class Turbulence:
        def __init__(self, cfg, decomp):
                if not cfg.Save_Statistics:
                        return
                
                self.stat_names = ['u_mean','v_mean','w_mean','uu_mean','vv_mean',
                                   'ww_mean','uv_mean','uw_mean','vw_mean']
    
                self.S = Data.State(self.stat_names, decomp)

                self.Niter_start = cfg.Niter_Stat_Start

        def save(self, n, Q, cfg, grid, decomp ,param, t):

                # JFM: THIS NEEDS TO REDUCE 3D to 2D

                u = Q['rhoU'].interior()/Q['rho'].interior()
                v = Q['rhoV'].interior()/Q['rho'].interior()
                w = Q['rhoW'].interior()/Q['rho'].interior()

                self.S['u_mean'].add_(u)
                self.S['v_mean'].add_(v)
                self.S['w_mean'].add_(w)
                self.S['uu_mean'].add_(u*u)
                self.S['vv_mean'].add_(v*v)
                self.S['ww_mean'].add_(w*w)
                self.S['uv_mean'].add_(u*v)
                self.S['uw_mean'].add_(u*w)
                self.S['vw_mean'].add_(v*w)

                if n%cfg.N_monitor==0:
                        u_mean  = self.S['u_mean'].interior()/( n - self.Niter_start )
                        v_mean  = self.S['v_mean'].interior()/( n - self.Niter_start )
                        w_mean  = self.S['w_mean'].interior()/( n - self.Niter_start )
                        uu_mean = self.S['uu_mean'].interior()/(n - self.Niter_start )
                        vv_mean = self.S['vv_mean'].interior()/(n - self.Niter_start )
                        ww_mean = self.S['ww_mean'].interior()/(n - self.Niter_start )
                        uv_mean = self.S['uv_mean'].interior()/(n - self.Niter_start )
                        uw_mean = self.S['uw_mean'].interior()/(n - self.Niter_start )
                        vw_mean = self.S['vw_mean'].interior()/(n - self.Niter_start )

                        q = torch.stack( (u_mean,
                                          v_mean,
                                          w_mean,
                                          uu_mean,
                                          vv_mean,
                                          ww_mean,
                                          uv_mean,
                                          uw_mean, 
                                          vw_mean), dim=0)

                        # Write data (HDF5) and metadata (XMF)
                        q_cpu = q.cpu().numpy()

                        # JFM: NEED TO ENSURE ONLY 2D DATA IS WRITTEN
                        Data.write_data(cfg,grid,decomp,q_cpu,self.stat_names,n,t,param.dt)

                return


