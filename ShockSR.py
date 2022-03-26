# reference:
#   K. W. Thompson, "The special relativistic shock tube"
#     Jounal of Fluid Mechanics 171 (1986) 365

import numpy as np
from scipy.optimize import newton

def ShockSR(rho1,p1,rho5,p5,gamma=4/3,
            N_rarefac=33, margin=0.2):
    """ special relativistic shock tube
    fluids are initially at rest (v1=v5=0)
    rho1,p1 = density and pressure in x<0
    rho5,p5 = density and pressure in x>0
    gamma = ratio cp/cv of specific heats
    N_rarefac = number of plotting points in rarefac wave
    margin = width between left edge and rarefaction wave
           = width between right edge and shock wave
    return xi,rho,p,v (each shape(N_rarefac+6)) where
      xi = x/t (similarity variable)
      rho,p,v = density,pressure,velocity
    """
    gg = gamma/(gamma-1)

    def shock(u4,p4):
        u42 = u4**2
        w42 = 1 + u42
        w4 = w42**0.5
        pp = p4 - p5
        rho4 = rho5*(pp + gg*p4*u42)/(w4*pp - rho5*u42)
        vs = (pp + gg*p4*u42)/(rho5 + gg*w4*p4)/u4
        return rho4,vs

    def shock_p(u4):
        u42 = u4**2
        w42 = 1 + u42
        w4 = w42**0.5
        def eq(p4):
            pp = p4 - p5
            rho4 = rho5*(pp + gg*p4*u42)/(w4*pp - rho5*u42)
            vs = (pp + gg*p4*u42)/(rho5 + gg*w4*p4)/u4
            h4 = rho4 + gg*p4
            f = vs*(h4*w42 - p4 - rho5 - p5/(gamma-1))
            f -= h4*w4*u4
            return f

        p_min = p5 + rho5*u42/w4
        return newton(eq, 2*p_min)

    def f(x): return x + (1+x**2)**0.5 # exp(arcsinh(x))
    y1 = (gg*p1/rho1)**0.5
    fy1 = f(y1)
    sg2 = (gamma-1)**0.5/2

    def rarefac_p(u):
        fu = f(u)**sg2
        fyu = fy1/fu
        y = np.abs(fyu - 1/fyu)/2
        p = p1*(y/y1)**(2*gg)
        return p

    def rarefac(u):
        fu = f(u)**sg2
        fyu = fy1/fu
        y = np.abs(fyu - 1/fyu)/2
        rho = rho1*(y/y1)**(2/(gamma-1))
        p = p1*(rho/rho1)**gamma
        c = ((gamma-1)*(1 - 1/(1+y**2)))**0.5
        w2 = 1 + u**2
        v = u/w2**0.5
        xi = v - c/w2/(1 - c*v)
        return rho,p,v,xi

    h1 = rho1 + gg*p1
    c1 = (gamma*p1/h1)**0.5
    xi1 = -c1

    fy1g = fy1**(1/sg2)
    u_max = (fy1g - 1/fy1g)/2
    u4 = newton(lambda x: shock_p(x) - rarefac_p(x), u_max)
    rho3,p3,v3,xi2 = rarefac(u4)
    rho4,xi4 = shock(u4,p3)
    xi3 = v3
    u = np.linspace(0,u4,N_rarefac)
    rho,p,v,xi = rarefac(u)

    rho = np.r_[rho1,rho,rho3,rho4,rho4,rho5,rho5]
    p = np.r_[p1,p,p3,p3,p3,p5,p5]
    v = np.r_[0,v,v3,v3,v3,0,0]
    xi = np.r_[xi1-margin, xi, xi3, xi3, xi4, xi4, xi4+margin]

    return xi,rho,p,v


def ShockNR(rho1, p1, rho5, p5, gamma=1.4,
            N=33, L=1):
    """ non-relativistic shock tube
    shock wave starts from x=0 at t=0
    solution depends only on xi = x/t
    rho1, p1 = density and pressure at x<0, t=0
    rho5, p5 = density and pressure at x>0, t=0
    gamma = c_p/c_v: specific heat ratio
    N,L = same as N_rarefac and margin in ShockSR
    assume rho1 > rho5 and p1 > p5
    return xi,r,u,p = np.array of
      coordinate (x/t), density, velocity, pressure
    """
    a1 = (gamma*p1/rho1)**.5 # sound speed at x<0
    a5 = (gamma*p5/rho5)**.5 # sound speed at x>0

    # Mach number of shock wave
    Ms = newton(lambda x: x - 1/x
                - a1/a5*(gamma+1)/(gamma-1)
                *(1 - (p5/p1*(2*gamma*x**2 - (gamma-1))
                       /(gamma+1))**((gamma-1)/(2*gamma)))
                ,1)

    # from contact discontinuity to shock wave
    r4 = rho5/(2/Ms**2 + gamma-1)*(gamma+1)
    u4 = 2*a5/(gamma+1)*(Ms - 1/Ms)
    p4 = p5*(2*gamma*Ms**2 - (gamma-1))/(gamma+1)

    # from contact discontinuity to expansion fan
    r3 = rho1*(p4/p1)**(1/gamma)
    a3 = (gamma*p4/r3)**.5

    x1 = -a1
    x2 = u4 - a3
    x3 = u4 # contact discontinuity
    x4 = Ms*a5 # shock wave

    # expansion fan
    xE = np.linspace(x1,x2,N)
    uE = 2/(gamma+1)*(a1 + xE)
    rE = rho1*(1 - (gamma-1)/2*uE/a1)**(2/(gamma-1))
    pE = p1*(rE/rho1)**gamma

    x = np.hstack((x1-L, xE, x3,x3,x4,x4,x4+L))
    r = np.hstack((rho1, rE, r3,r4,r4,rho5,rho5))
    u = np.hstack((0, uE, u4,u4,u4,0,0))
    p = np.hstack((p1, pE, p4,p4,p4,p5,p5))

    return x,r,p,u
