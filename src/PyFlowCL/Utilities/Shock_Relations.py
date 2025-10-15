import numpy as np

def isen_T_ratio(M,g):
    # T0 / T
    return 1.0 + 0.5*(g-1.0) * M**2

def isen_p_ratio(M,g):
    # p0 / p
    return isen_T_ratio(M,g)**(g/(g-1.0))

def jump_P(M1,g):
    if M1 <= 1.0:
        return 1.0
    else:
        return 1.0 + 2.0*g/(g+1.0)*(M1**2 - 1.0)

def jump_rho(M1,g):
    if M1 <= 1.0:
        return 1.0
    else:
        return (g+1.0)*M1**2/(2.0 + (g-1.0)*M1**2)

def jump_T(M1,g):
    if M1 <= 1.0:
        return 1.0
    else:
        return jump_P(M1,g)/jump_rho(M1,g)

def jump_M2(M1,g):
    if M1 <= 1.0:
        return M1
    else:
        return np.sqrt((1.0 + 0.5*(g-1.0)*M1**2)/(g*M1**2 - 0.5*(g-1.0)))

# Theta-delta-M relation
def TDM(th,M,g):
    return np.arctan(2./np.tan(th) * (M**2*np.sin(th)**2 - 1.) / \
                     (M**2*(g+np.cos(2.*th)) + 2.))

# Compute weak oblique-shock angle (theta) from M1 and turn angle (delta)
def oblique_theta(M1,delta_target,g):
    if (M1<=1.0): return None,None,None,None
    
    theta = 35
    stop = False
    inc  = 0.1
    it = 0; it_max = 10000
    while (not stop and it < it_max):
        it += 1;

        # Predict deflection angle
        delta = np.degrees(TDM(np.radians(theta),M1,g))
        #print(it, theta, delta)

        err = delta - delta_target
        if (abs(err) < inc):
            inc /= 10
        elif (abs(err) < 1e-5):
            stop = True
        elif (theta > 90):
            theta = None
            stop  = True
            print('No attached shock solution')
        elif (err>0):
            theta -= inc
        else:
            theta += inc

    # Checks
    if (not stop): raise Exception('oblique_theta failed to converge')

    if (theta is not None):
        # Get jump properties
        M1n = M1*np.sin(np.radians(theta))
        T_ratio = jump_T(M1n,g)
        p_ratio = jump_P(M1n,g)
        rho_ratio = jump_rho(M1n,g)
        
        M2n = jump_M2(M1n,g)
        M2t = M1*np.cos(np.radians(theta))*np.sqrt(1./T_ratio)
        M2  = np.sqrt(M2n**2 + M2t**2)
        #print(it, M2, theta, delta)
    else:
        M2 = None; p_ratio = None; rho_ratio = None

    return theta,M2,p_ratio,rho_ratio
