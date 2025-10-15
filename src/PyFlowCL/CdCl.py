from . import Operator as op
from .Library import Parallel

import torch
import numpy as np
import pickle
from scipy.fft import fft, ifft
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
# plt.rcParams['text.usetex'] = True

import pdb
import os

# --------------------------------------------------------------
# compute drag or lift function - 2D
# --------------------------------------------------------------
def compute_CdCl(comms, grid, param, metrics, q, adjoint=False):
    # grid.X[:,0]**2 + grid.Y[:,0]**2 == [0.25, ..., 0.25] on the cylinder bnd
    # q[:,0] should on the cylinder bnd

    # ----------------------------- Copy from RHS.py Start---------------------------------
    # Extract conserved variables
    rho  = q['rho']
    rhoU = q['rhoU']
    rhoV = q['rhoV']
    rhoE = q['rhoE']

    # Compute primitives including overlaps
    u = rhoU.var/rho.var
    v = rhoV.var/rho.var
    e = rhoE.var/rho.var - 0.5*(u**2 + v**2)
    p = (param.gamma-1.0) * rho.var * e * param.Ma**2
    T = param.gamma * e * param.Ma**2 
    
    # Compute 1st derivatives - extended interior for 2nd derivatives
    drho_dx,drho_dy   = metrics.grad_node( rho.var,  compute_extended=True )[:2]
    drhoU_dx,drhoU_dy = metrics.grad_node( rhoU.var, compute_extended=True )[:2] 
    drhoV_dx,drhoV_dy = metrics.grad_node( rhoV.var, compute_extended=True )[:2]

    # Velocity gradients - extended interior
    # du
    du_dx = (drhoU_dx - metrics.full2ext( u ) * drho_dx) / metrics.full2ext( rho.var )
    du_dy = (drhoU_dy - metrics.full2ext( u ) * drho_dy) / metrics.full2ext( rho.var )
    # dv
    dv_dx = (drhoV_dx - metrics.full2ext( v ) * drho_dx) / metrics.full2ext( rho.var )
    dv_dy = (drhoV_dy - metrics.full2ext( v ) * drho_dy) / metrics.full2ext( rho.var )
    
    # Velocity divergence
    div_vel = du_dx + dv_dy

    # Variable transport properties
    # mu = (param.gamma-1.0) * metrics.full2ext(T)**0.7
    # mu = ( (param.gamma-1.0) * metrics.full2ext(T) )**0.7  # mu=1
    mu = 1.0 +  0*metrics.full2ext(T)   ## mu=1
    

    #kappa = mu
    kappa = mu#param.mu

    # Artificial diffusivity (Kawai & Lele JCP 2008)
    if param.artDiss:
        # Exclude from adjoint calculation
        with torch.inference_mode():
            # Strain-rate magnitude
            S = op.strainrate_mag_2D(du_dx,du_dy,dv_dx,dv_dy)

            # Evaulate the artificial transport coefficients
            curl_u = dv_dx - du_dy
            mu_art,beta_art,kappa_art = op.art_diff4_D_2D(rho,drho_dx,drho_dy,S,tmp_grad,
                                                          div_vel,curl_u,T,e,
                                                          param.cs,grid,metrics,param.device)

            #mu_eff    = param.mu + mu_art *param.Re
            mu_eff    = mu + mu_art *param.Re
            beta_art  = beta_art *param.Re

            #DN  = torch.amax(mu_eff)/param.Re * param.dt / param.dx_min**2
            #print(torch.amax(mu_art),
            #      torch.amax(kappa_art),
            #      torch.amax(beta_art),
            #      DN)
    else:
        mu_eff    = mu #param.mu
        beta_art  = 0.0

        
    # Viscous stress
    #   Divergence terms are computed on true interior
    div_term = (beta_art - 2.0*mu_eff/3.0)*div_vel
    # import pdb; pdb.set_trace()
    sigma_11 = 2.0*mu_eff*du_dx + div_term
    sigma_22 = 2.0*mu_eff*dv_dy + div_term    
    sigma_12 = mu_eff*( du_dy + dv_dx )

    sigma_11 = metrics.ext2int( sigma_11 )
    sigma_22 = metrics.ext2int( sigma_22 )
    sigma_12 = metrics.ext2int( sigma_12 )

    p = metrics.full2int( p )


    # ----------------------------- Copy from RHS.py End ---------------------------------


    # Cauchy stress tensor  sigma_ij = tau_ij - p*I_ij
    
    ind = 0  # index for on cylinder boundary Qs
    if not param.debug: 
        assert ind==0  # grids index for [on cylinder boundary] Qs
    elif param.debug and ind<=3 and (not param.Train) and (not param.adjoint_verification):  # grids index for [near cylinder boundary] Qs
        pdb.set_trace()
        # plot_dir = '/scratch365/xliu24/test_verification/cylinder_2D_adjoint/implicit_filter/Output_cylinder_Nx1_512_Ma0.1_Re100/R150/Cd_wiggles/Nf100/Cd_bf_ind{}'
        plot_CdCl_ls(param.plot_dir+'/',\
            [grid.X[:256,ind],grid.X[:256,ind],grid.X[:256,ind],grid.X[:256,ind]],\
            [ p[:256,ind], sigma_11[:256,ind],sigma_12[:256,ind],sigma_22[:256,ind]],\
            label_ls=['p','sigma_11','sigma_12','sigma_22'], xylabel=['x','Values'], plot_name='cyl_top_100_{}'.format(param.NCd), title='Top half cylinder',ylim=[-22,22])
        
        plot_CdCl_ls(param.plot_dir+'/',\
            [grid.X[256:,ind],grid.X[256:,ind],grid.X[256:,ind],grid.X[256:,ind]],\
            [ p[256:,ind], sigma_11[256:,ind],sigma_12[256:,ind],sigma_22[256:,ind]],\
            label_ls=['p','sigma_11','sigma_12','sigma_22'], xylabel=['x','Values'], plot_name='cyl_bop_100_{}'.format(param.NCd), title='Bottom half cylinder',ylim=[-22,22])
        
        plot_CdCl_ls(param.plot_dir+'/',\
            [grid.X[:256,ind],grid.X[:256,ind],grid.X[:256,ind],grid.X[:256,ind],grid.X[:256,ind]],\
            [metrics.ext2int(du_dx)[:256,ind],metrics.ext2int(du_dy)[:256,ind],metrics.ext2int(dv_dx)[:256,ind],metrics.ext2int(dv_dy)[:256,ind],metrics.ext2int(div_term)[:256,ind]],\
            label_ls=['du_dx','du_dy','dv_dx','dv_dy','div_term'], xylabel=['x','Values'], plot_name='cyl_dri_top_100_{}'.format(param.NCd), title='Top half cylinder',ylim=[-16,16])
        plot_CdCl_ls(param.plot_dir+'/',\
            [grid.X[256:,ind],grid.X[256:,ind],grid.X[256:,ind],grid.X[256:,ind],grid.X[256:,ind]],\
            [metrics.ext2int(du_dx)[256:,ind],metrics.ext2int(du_dy)[256:,ind],metrics.ext2int(dv_dx)[256:,ind],metrics.ext2int(dv_dy)[256:,ind],metrics.ext2int(div_term)[:256,ind]],\
            label_ls=['du_dx','du_dy','dv_dx','dv_dy','div_term'], xylabel=['x','Values'], plot_name='cyl_dri_bot_100_{}'.format(param.NCd), title='Bottom half cylinder',ylim=[-16,16])


    # L = 2*param.R_min # = 1.0
    if True:
        p_inf = 101352.0
        rho_inf = 1.0 
        u_inf = param.Ma*np.sqrt(param.gamma*p_inf/rho_inf) # = 37.6686607141799

        # p_inf = 101352.0
        # u_inf = 1.0
        # rho_inf = param.Ma**2*param.gamma*p_inf/u_inf**2  # = 1418.928

        # rho_inf = 1.0 # L=2*param.R_min=1.0
        # u_inf = 1.0
        # p_inf = (u_inf/param.Ma)**2 * rho_inf/param.gamma  # = 71.42857142857143

        # Cauchy stress
        sigma11 = (u_inf**2*rho_inf/param.Re) * (sigma_11[:,ind].squeeze()) - (p[:,ind].squeeze())*p_inf*param.gamma
        sigma12 = (u_inf**2*rho_inf/param.Re) * (sigma_12[:,ind].squeeze())
        sigma22 = (u_inf**2*rho_inf/param.Re) * (sigma_22[:,ind].squeeze()) - (p[:,ind].squeeze())*p_inf*param.gamma

        if ind>3:
            # sigma11 = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_11[:,1:2],1).squeeze()) - (torch.mean(p[:,1:2],1).squeeze())*p_inf*param.gamma
            # sigma12 = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_12[:,1:2],1).squeeze())
            # sigma22 = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_22[:,1:2],1).squeeze()) - (torch.mean(p[:,1:2],1).squeeze())*p_inf*param.gamma

            sigma11 = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_11[:,0]+sigma_11[:,2]).squeeze()) - (0.5*(p[:,0]+p[:,2]).squeeze())*p_inf*param.gamma
            sigma12 = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_12[:,0]+sigma_11[:,2]).squeeze()) 
            sigma11 = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_22[:,0]+sigma_11[:,2]).squeeze()) - (0.5*(p[:,0]+p[:,2]).squeeze())*p_inf*param.gamma


        # normalizer
        cef = 0.5*rho_inf*(u_inf**2)*(2*grid.R_min) #.cpu().numpy()

    # unit vector normal  [nx, ny]
    nx = grid.X[:,0]/grid.R_min 
    ny = grid.Y[:,0]/grid.R_min
    pi = torch.acos(torch.zeros(1)).item() * 2

    # debug y_plus
    if False:
        tau_w_1 =  (u_inf**2*rho_inf/param.Re) * sigma_11[:,ind].squeeze()*nx + (u_inf**2*rho_inf/param.Re) * sigma_12[:,ind].squeeze()*ny
        tau_w_2 =  (u_inf**2*rho_inf/param.Re) * sigma_12[:,ind].squeeze()*nx + (u_inf**2*rho_inf/param.Re) * sigma_22[:,ind].squeeze()*ny
        dy = np.sqrt(grid.Y[:,ind+1]**2+grid.X[:,ind+1]**2)-grid.R_min
        y_plus = computed_yplus(y=dy, tau_w=[tau_w_1, tau_w_2], rho=1.0, mu=1/param.Re)
        if param.jproc >0: y_plus *= 0.0
        y_plus_max = comms.parallel_max( torch.amax(y_plus).cpu().numpy() )
        if (param.rank==0):
            print('------- y_plus_max = {:0.5f} -------'.format(y_plus_max))

    
    # compute total drag and lift cef Cd & Cl
    Fd = torch.sum(sigma11*nx + sigma12*ny) * ( 2*pi*grid.R_min/grid.Nx1 ) 
    Fl = torch.sum(sigma12*nx + sigma22*ny) * ( 2*pi*grid.R_min/grid.Nx1 )       
    if param.jproc >0: Fd *= 0.0; Fl *= 0.0
    
    if (param.Train or param.adjoint_verification) and adjoint: # train NN using NS_2D_A func
        Cd = Fd/cef
        Cl = Fl/cef
        return Cd, Cl

    else: 
        Fd_g = comms.parallel_sum(Fd.detach().numpy())
        Fl_g = comms.parallel_sum(Fl.detach().numpy())
        Cd_g = Fd_g/cef      # anyalitically, when Re<1,  Cd=3*np.pi*mu*param.U0**2*2*grid.R_min (unconfined cylinder)
        Cl_g = Fl_g/cef

        # compute friction drag cef Cdf & Clf
        sigma11w = (u_inf**2*rho_inf/param.Re) * (sigma_11[:,ind].squeeze()) #- (p[:,ind].squeeze())*p_inf*param.gamma
        sigma12w = (u_inf**2*rho_inf/param.Re) * (sigma_12[:,ind].squeeze())
        sigma22w = (u_inf**2*rho_inf/param.Re) * (sigma_22[:,ind].squeeze())
        if ind>3:
            # sigma11w = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_11[:,1:2],1).squeeze()) #- (p[:,ind].squeeze())*p_inf*param.gamma
            # sigma12w = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_12[:,1:2],1).squeeze())
            # sigma22w = (u_inf**2*rho_inf/param.Re) * (torch.mean(sigma_22[:,1:2],1).squeeze())

            sigma11w = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_11[:,0]+sigma_11[:,2]).squeeze()) 
            sigma12w = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_11[:,0]+sigma_11[:,2]).squeeze()) 
            sigma22w = (u_inf**2*rho_inf/param.Re) * (0.5*(sigma_11[:,0]+sigma_11[:,2]).squeeze()) 


        Fdf = torch.sum(sigma11w*nx + sigma12w*ny) * ( 2*pi*grid.R_min/grid.Nx1 ) # 2*np.pi/grid.d_xi=grid.Nx1
        if param.jproc >0: Fdf *= 0.0 
        Fdf_g = comms.parallel_sum(Fdf.detach().numpy())
        Cdf_g = Fdf_g/cef

        Flf = torch.sum(sigma12w*nx + sigma22w*ny) * ( 2*pi*grid.R_min/grid.Nx1 ) 
        if param.jproc >0: Flf *= 0.0 
        Flf_g = comms.parallel_sum(Flf.detach().numpy())
        Clf_g = Flf_g/cef
        
        # compute pressure drag cef Cdp & Clp
        # sigma11wp = - (p[:,ind].squeeze())*p_inf*param.gamma
        # sigma12wp = 0*(p[:,ind].squeeze())
        # sigma22wp = - (p[:,ind].squeeze())*p_inf*param.gamma
        # Fdp = torch.sum(sigma11wp*nx + sigma12wp*ny) * ( 2*pi*grid.R_min/grid.Nx1 ) 
        # if param.jproc >0: Fdp *= 0.0 
        # Fdp_g = comms.parallel_sum(Fdp.cpu().numpy())
        # Cdp_g = Fdp_g/cef
        # Flp = torch.sum(sigma12wp*nx + sigma22wp*ny) * ( 2*pi*grid.R_min/grid.Nx1 ) 
        # if param.jproc >0: Fdp *= 0.0 
        # Flp_g = comms.parallel_sum(Flp.cpu().numpy())
        # Clp_g = Flp_g/cef

        Cdp_g = Cd_g - Cdf_g
        Clp_g = Cl_g - Clf_g

        return Cd_g, Cdf_g, Cdp_g, Cl_g, Clf_g, Clp_g


def plot_CdCl(plot_dir,time_hist,Cd_hist, label='', dir=[1,0], plot_name=''):
    # print(Cd_hist)
    plt.plot(time_hist,Cd_hist,'k--', label=label)
    # plt.show()
    # plt.plot(x,u_target,'k--',label='u_target')
    # plt.ylim(-0.05, 1.35)
    plt.legend(loc="upper right")
    plt.xlabel('$t$ (non-dimensional)')
    if dir==[1,0]:
        plt.ylabel('$C_d$')
    elif dir==[0,1]:
        plt.ylabel('$C_l$')

    plt.tight_layout()
    # plt.show()
    # plt.title('$C_d$ history with control')
    plt.savefig(plot_dir+'/{}_hist.png'.format(plot_name)) 
    plt.close()


style_list = ['ro','b-','k--','g+','ms','c*']
def plot_CdCl_ls(plot_dir,t_hist_ls,Cd_hist_ls,label_ls,sty_ls=None,dir=[1,0],plot_name='baseline',title='Re=100',xylabel=None,ylim=None,shift=0):
    for i, (t_hist,Cd_hist,label) in enumerate(zip(t_hist_ls,Cd_hist_ls,label_ls)):
        if sty_ls is None:
            plt.plot(t_hist[shift:],Cd_hist[shift:], label=label)
        else:
            plt.plot(t_hist[shift:],Cd_hist[shift:], sty_ls[i], label=label)
    plt.legend(loc="upper right")
    plt.xlabel('$t$ (non-dimensional)')
    if xylabel is not None:
        plt.xlabel(xylabel[0])
        plt.ylabel(xylabel[1])
    else:
        if dir==[1,0]:
            plt.ylabel('$C_d$')
        elif dir==[0,1]:
            plt.ylabel('$C_l$')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_dir+plot_name+'.png')
    # print('saved figure '+plot_dir+plot_name+'.png')
    plt.close()



# -----------------------------------------------------------------------------------
# compute vortex shedding frequency and Strouhal number given Cd Cl hist data
# -----------------------------------------------------------------------------------
def compute_f(plot_dir, inputConfig, shift=200):
    t_hist = pickle.load( open(inputConfig.t_hist_file, "rb" ) )
    Cl_hist = pickle.load( open( inputConfig.Cl_hist_file, "rb" ) )
    Clf_hist = pickle.load( open( inputConfig.Clf_hist_file, "rb" ) )
    Clp_hist = pickle.load( open( inputConfig.Clp_hist_file, "rb" ) )
    Cd_hist = pickle.load( open( inputConfig.Cd_hist_file, "rb" ) )
    Cdf_hist = pickle.load( open( inputConfig.Cdf_hist_file, "rb" ) )
    Cdp_hist = pickle.load( open( inputConfig.Cdp_hist_file, "rb" ) )
    dt = inputConfig.dt
    # shift = 200
    if inputConfig.Re<60:
        mean_Cl = np.mean( Cl_hist[shift:])
        mean_Clf = np.mean( Clf_hist[shift:])
        mean_Clp = np.mean( Clp_hist[shift:])
        mean_Cd = np.mean( Cd_hist[shift:])
        mean_Cdf = np.mean( Cdf_hist[shift:])
        mean_Cdp = np.mean( Cdp_hist[shift:])
        St = None
        fmtStr ='>>> Re={}, Rmax={}, mean(Cd)={:0.4f}, mean(Cdf)={:0.4f}, mean(Cdp)={:0.4f}, mean(Cl)={:0.4e}, mean(Clf)={:0.4e}, mean(Clp)={:0.4e} <<<'
        print(fmtStr.format(inputConfig.Re, inputConfig.Rmax, mean_Cd,mean_Cdf,mean_Cdp, mean_Cl,mean_Clf,mean_Clp))
    else:    
        fft_wave = np.fft.fft(np.asarray(Cl_hist[shift:]))
        fft_fre = np.fft.fftfreq(len(Cl_hist[shift:]), dt)  # 1/SAMPLE_RATE = dt
        # pdb.set_trace()
        mag = np.abs(fft_wave)
        ind = np.argmax(mag)
        f = np.abs(fft_fre[ind])
        U0 = 1.0
        St = f*(2*inputConfig.Rmin)/U0
        N_samples_per_pefriod = int(1/(f*dt))
        N_samples = int(np.floor(len(t_hist[shift:])/N_samples_per_pefriod)*N_samples_per_pefriod)
        print('>>> N_samples = {}'.format(N_samples))
        mean_Cl  = np.mean( Cl_hist[shift : shift  + N_samples])
        mean_Clf = np.mean( Clf_hist[shift : shift + N_samples])
        mean_Clp = np.mean( Clp_hist[shift : shift + N_samples])
        mean_Cd  = np.mean( Cd_hist[shift : shift  + N_samples])
        mean_Cdf = np.mean( Cdf_hist[shift : shift + N_samples])
        mean_Cdp = np.mean( Cdp_hist[shift : shift + N_samples])
        fmtStr = '>>> Re={}, Rmax={}, mean(Cd)={:0.4f}, mean(Cdf)={:0.4f}, mean(Cdp)={:0.4f}, mean(Cl)={:0.4e}, mean(Clf)={:0.4e}, mean(Clp)={:0.4e}, Cl_fre={:0.4f}, St={:0.4f} <<<'
        print(fmtStr.format(inputConfig.Re, inputConfig.Rmax, mean_Cd,mean_Cdf,mean_Cdp, mean_Cl,mean_Clf,mean_Clp,fft_fre[ind],St))
        plt.figure(figsize=(10,10))
        plt.subplot(311)
        plt.plot(t_hist[shift:],Cd_hist[shift:], label='$C_d$, Re={}'.format(inputConfig.Re))
        plt.legend(loc="upper right")
        plt.ylabel("$C_d$")
        plt.xlabel('$t$ (non-dimensional), mean($C_d$)={:0.4f}'.format(mean_Cd))

        plt.subplot(312)
        plt.plot(t_hist[shift:],Cl_hist[shift:], label='$C_l$, Re={}'.format(inputConfig.Re))
        plt.legend(loc="upper right")
        plt.ylabel("$C_l$")
        plt.xlabel('$t$ (non-dimensional), mean($C_l$)={:0.4e}'.format(mean_Cl))
        
        plt.subplot(313)
        plt.plot(fft_fre, mag, label='FFT, Re={}'.format(inputConfig.Re))    
        plt.xlabel("frequency (Hz)" + ', $S_t$={:0.4f}'.format(St))
        plt.ylabel("Power")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.xlim(-50,50)
        # plt.title(title)
        plt.savefig(plot_dir+'/Cl_fre_Re{}.png'.format(inputConfig.Re))
        plt.close()

    return mean_Cd, mean_Cdf, mean_Cdp, mean_Cl,mean_Clf,mean_Clp, St


def plot_Cd_fit(Re_ls, mean_Cd_ls, mean_Cdf_ls, mean_Cdp_ls, mean_Cl_ls, mean_Clf_ls,mean_Clp_ls,Sts):
    x = np.linspace(1.0, 400.0, num=1000)
    y = 0.26 + 7.89*x**(-0.5)
    Re_ls_Rajani = [1, 10, 50, 100, 200, 300, 400]
    mCd_ls_Rajani = [None, None, 1.41, 1.3353, 1.3365, 1.3667, 1.3905]
    mCdf_ls_Rajani = [None, None, 0.46, 0.3253, 0.2665, 0.2167, None]
    mCdp_ls_Rajani = [None, None, 0.95, 1.00, 1.07, 1.15, None]
    St_ls_Rajani  = [None, None, 0.112, 0.1569, 0.1957, 0.2150, 0.2348]
    plt.figure(figsize=(12,10))

    plt.subplot(321)
    plt.scatter(Re_ls, mean_Cd_ls, marker='o', c='r', label='PyFlowCL')
    plt.scatter(Re_ls_Rajani, mCd_ls_Rajani, marker='x', c='k', label='Rajani, Kandasamy, Majumdar(2009)')
    plt.plot(x, y, 'k--', label='Sen, Mittal, and Biswas(2009), Eq.(7.8)')
    plt.ylabel('mean ($C_d$)')
    plt.xlim(-10, 410)
    plt.legend(loc='upper right', prop={'size': 10})

    plt.subplot(323)
    plt.scatter(Re_ls, mean_Cdf_ls, marker='o', c='r')
    plt.scatter(Re_ls_Rajani, mCdf_ls_Rajani, marker='x', c='k')
    plt.ylabel('mean ($C_{df}$)')
    plt.xlim(-10, 410)
    # plt.legend(loc='upper right', prop={'size': 10})


    plt.subplot(325)
    plt.scatter(Re_ls, mean_Cdp_ls, marker='o', c='r')
    plt.scatter(Re_ls_Rajani, mCdp_ls_Rajani, marker='x', c='k')
    plt.xlabel('Re')
    plt.ylabel('mean ($C_{dp}$)')
    plt.xlim(-10, 410)
    # plt.legend(loc='upper right', prop={'size': 10})

    plt.subplot(322)
    plt.scatter(Re_ls, mean_Cl_ls, marker='o', c='r')
    # plt.xlabel('Re')
    plt.ylabel('mean ($C_l$)')
    plt.xlim(-10, 410)
    # plt.legend(loc='upper right', prop={'size': 10})

    plt.subplot(324)
    plt.scatter(Re_ls, mean_Clf_ls, marker='o', c='r')
    # plt.xlabel('Re')
    plt.ylabel('mean ($C_{lf}$)')
    plt.xlim(-10, 410)
    # plt.legend(loc='upper right', prop={'size': 10})

    plt.subplot(326)
    plt.scatter(Re_ls, mean_Clp_ls, marker='o', c='r')
    plt.xlabel('Re')
    plt.ylabel('mean ($C_{lp}$)')
    plt.xlim(-10, 410)
    # plt.legend(loc='upper right', prop={'size': 10})

    plt.savefig('./implicit_filter/compare/Cd_fitted_mu**0.7.png')
    plt.close()

    plt.plot(figsize=(7,5))
    plt.scatter(Re_ls, Sts, marker='o', c='r', label='PyFlowCL')
    plt.scatter(Re_ls_Rajani, St_ls_Rajani, marker='x', c='k', label='Rajani, Kandasamy, Majumdar(2009)')
    plt.xlabel('Re')
    plt.ylabel('$S_t$')
    plt.xlim(-10, 410)
    plt.ylim(0.1, 0.3)
    plt.legend(loc='upper right', prop={'size': 10})
    plt.savefig('./implicit_filter/compare/St_mu**0.7.png')
    plt.close()



def computed_yplus(y, tau_w, rho, mu):
    tau_w_mag = np.sqrt( tau_w[0]**2 + tau_w[1]**2 )
    u_tau = np.sqrt(tau_w_mag/rho)
    y_plus = y*u_tau*rho/mu
    return y_plus



# --------------------------------------------------------------------------
# compute J=int_{0}^{T}Cd^2 when verifying adjoint gradient computation
# --------------------------------------------------------------------------
def sum_CdCl(cfg):
    Cd_hist = pickle.load( open( cfg.outDir+"/Cd_hist.npy",  "rb" ) )
    return  (np.array(Cd_hist)**2).sum()


# -------------------------------------------------------
# compute and plot adjoint gradient convergence rate
# -------------------------------------------------------
def dJdu_loglog(cfg, du_ls, dJ_ls, plot_name, label_ls, style_ls=['-ro', ':b*']):
    plt.figure(figsize=(8, 6))    
    for (dus, dJs, label,style) in zip(du_ls, dJ_ls, label_ls, style_ls):
        dJs = np.abs(dJs)
        dus = np.array(dus)
        cvg_rate = (np.log(dJs[0:-2]/dJs[1:-1]))/(np.log(dus[0:-2]/dus[1:-1]))
        print('{} convergence rate = {}'.format(label, cvg_rate.tolist()))
        plt.loglog(dus, dJs, style, label=label)
    # plot 1:1 error
    dJ11 = [dus[i]*np.mean(dJs[2:-2])/np.mean(dus[2:-2]) for i in range(len(dus))]
    plt.loglog(dus, dJ11, 'k--', label='1:1')
    # if cfg.Nx3>1:
    #     plt.title('Gradient accuracy by perturbing u at point $[{:.2f}, {:.2f}, {:.2f}]$'.format(cfg.per_X, cfg.per_Y, cfg.per_Z))
    # else:
    #     plt.title('Gradient accuracy by perturbing u at point $[{:.2f}, {:.2f}]$'.format(cfg.per_X, cfg.per_Y))
    plt.title('Gradient accuracy by perturbing u at point $[{:.2f}, {:.2f}]$'.format(cfg.per_X, cfg.per_Y))
    plt.xlabel('log($\\theta$)')
    plt.ylabel('log($\\epsilon$)')
    plt.legend(loc="upper left")
    plt.savefig(cfg.outDir+'/../{}.png'.format(plot_name)) 
    plt.close()

