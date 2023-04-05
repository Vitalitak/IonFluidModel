import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

"""
1D Fluid model of collisionless Ar plasma sheath for electrons and ions
Electrode potential in equivalent circuit model
"""

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
for dn/dt = 0 and du/dt = 0

dPsi/dx=F(x, Psi)

"""


def RKPoisN(dx, Psi, Nsh, Nx, n0, Ti, Te, Psil, FN):
    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    Psi(0)=0
    dPsi/dx(0) = 0
    dPsi/dx<0



    dPsi/dx=F(x, Psi)
    F = -(A*exp(Psi)+B*int(N(dzetta), dzetta=[0, Psi(x)])+C)^1/2

    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=-2*e*e*n0/eps0/kTe
    C=-2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)

    Psi(0)=0

    Four order Runge-Kutta method
    f1=F(x[n], Psi[n])
    f2=F(x[n]+dx/2, Psi[n]+dx/2*f1)
    f3=F(x[n]+dx/2, Psi[n]+dx/2*f2)
    f4=F(x[n]+dx, Psi[n]+dx*f3)

    Psi[n+1]=Psi[n]+dx/6*(f1+2*f2+2*f3+f4)

    """

    # dx = x[Npl - 1]-x[Npl - 2]
    # Nx = len[Ksi]

    Psi[Nsh + 2] = -0.01 * dx * dx * e * e * n0 / eps0 / kTe
    A = 2 * e * e * n0 / eps0 / kTe
    B = -2 * e * e * n0 / eps0 / kTe
    C = -2 * e * e * n0 / eps0 / kTe

    # print(A)
    # print(B)
    # print(C)
    # print(D)
    i = 2
    # for i in range(Nsh+2, Nx-1):
    while (Psi[i] > Psil) and (i < Nx):
        print(i)
        f1 = -m.pow(-(A * m.exp(Psi[i]) + B * quad(FN, 0, Psi[i])[0] + C), 0.5)
        f2 = -m.pow(-(A * m.exp(Psi[i] + dx / 2 * f1) + B * quad(FN, 0, Psi[i] + dx / 2 * f1)[0] + C), 0.5)
        f3 = -m.pow(-(A * m.exp(Psi[i] + dx / 2 * f2) + B * quad(FN, 0, Psi[i] + dx / 2 * f2)[0] + C), 0.5)
        f4 = -m.pow(-(A * m.exp(Psi[i] + dx * f3) + B * quad(FN, 0, Psi[i] + dx * f3)[0] + C), 0.5)
        Psi[i + 1] = Psi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        i = i + 1

    Nel = i-1

    return Psi, Nel

def Pois(ne, ni, Ve, dx, Nel, Nx):

    """
    sweep method solution of Poisson equation
    electrode boundary condition Ve

    """

    e = 1.6E-19
    eps0 = 8.85E-12

    #Nx = len(ne)
    #dx = boxsize / Nx
    V = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nel)]
    b = [0 for k in range(0, Nel)]

    # forward
    # boundary conditions on plasma surface: (V)pl = 0
    #a[0] = 0.5
    #b[0] = 0.5 * (ne[0] - ni[0]) * dx * dx
    a[0] = 0
    b[0] = 0

    for i in range(1, Nel-1):
        a[i] = 1/ (2-a[i-1])
        b[i] = (b[i-1] - e / eps0 * (ne[i] - ni[i]) * dx * dx)/(2-a[i-1])

    # boundary condition on electrode surface: (V)el = Ve
    a[Nel-1] = 0
    #b[Nx-1] = (b[Nx-2] - (ne[Nx-1] - ni[Nx-1]) * dx * dx)/(2-a[Nx-2])
    b[Nel-1] = Ve  #  (V)p = 0
    #b[Nx - 1] = b[Nx - 2] / (1 - a[Nx - 2])  # (dV/dx)p = 0

    # backward
    V[Nel-1] = b[Nel-1]
    for i in range(Nel-1, 0, -1):
        V[i-1] = a[i-1]*V[i]+b[i-1]

    return V

def momentum(V, n, uprev, mi, kTi, dx, dt, Nel, Nx):

    """
    sweep method solution of momentum balance equation
    """

    e = 1.6E-19
    u = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nel)]
    b = [0 for k in range(0, Nel)]

    # forward
    # boundary conditions on plasma surface: (du/dx)pl = 0
    #a[0] = -uprev[1] * dt / 4.0 / dx
    #b[0] = (V[1] - V[0])/dx/m - uprev[0]/dt
    a[0] = 1
    b[0] = 0

    for i in range(1, Nel - 1):
        a[i] = uprev[i+1] / 4.0 / dx / (-1 / dt + uprev[i - 1] * a[i-1] / 4.0 / dx)
        b[i] = (-uprev[i-1] / 4.0 / dx * b[i - 1] + e*(V[i+1]-V[i]) /dx/mi - uprev[i] / dt - kTi*(n[i+1]-n[i])/n[i]) / (-1 / dt + uprev[i-1] * a[i-1] / 4.0 / dx)

    # boundary condition on electrode surface: (du/dx)el = 0
    a[Nel - 1] = 0
    #b[Nx - 1] = (-uprev[Nx-2] / 4.0 / dx * b[Nx-2] + (V[Nx-1]-V[Nx-2])/dx - uprev[Nx-1] / dt) / (-1 / dt + uprev[Nx-2] * a[Nx-2] / 4.0 / dx)  # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = 0  # (u)p = 0
    b[Nel - 1] = b[Nel - 2]/(1 - a[Nel - 2])  # (du/dx)el = 0

    # backward
    u[Nel - 1] = b[Nel - 1]
    for i in range(Nel - 1, 0, -1):
        u[i - 1] = a[i - 1] * u[i] + b[i - 1]

    return u

def main():
    # initialisation of parameters
    boxsize = 3.5E-4  # m
    #a = 1E-6
    dt = 1E-11 # s
    dx = 1E-7
    Nx = int(boxsize/dx)
    #Nsh = int(a/dx)
    Nsh = 0
    #Nx = 500000
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.3  # eV
    Ti = 0.06  # eV
    n0 = 4E17  # m-3
    Vdc = -12
    C = 1.4E-16
    C /= 1.6E-19
    gamma = 5/3


    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    #V0 = 0
    # V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    Nt = int(tEnd / dt)

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    dPsidx = [0 for k in range(0, Nx)]
    Vrf = 0

    Psil = e*Vdc/kTe

    FPsi = lambda x: (5*gamma-3)*Ti/Te/2/(gamma-1)*(1-3*(gamma-1)/(5*gamma-3)*m.pow(x, -2)-2*gamma/(5*gamma-3)*m.pow(x, gamma-1))

    FN = inversefunc(FPsi, domain=[0.001, 1])

    """
    res, err = quad(FN, 0, -1)
    print(print("The numerical result is {:f} (+-{:g})"
    .format(res, err)))
    print(quad(FN, 0, -1)[0])
    """

    Psi, Nel = RKPoisN(dx, Psi, Nsh, Nx, n0, Ti, Te, Psil, FN)

    #for i in range(0, NPsi):
        #xNi[i] = FN(xPsi[i])
        #intNx[i] = quad(FN, 0, xPsi[i])[0]


    for i in range(Nsh, Nx):
        Ni[i] = FN(Psi[i])

    for i in range(Nsh, Nx-1):
        dPsidx[i] = (Psi[i+1]-Psi[i])/dx

    for i in range(0, Nx):
        V[i] = Psi[i]*kTe/e
        ni[i] = Ni[i]*n0
        ne[i] = n0*m.exp(e*V[i]/kTe)
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]

    Psi_1 = [0 for k in range(0, Nx)]
    Ni_1 = [0 for k in range(0, Nx)]
    ui_1 = [0 for k in range(0, Nx)]
    #ui_m = [0 for k in range(0, Nx)]
    V_1 = [0 for k in range(0, Nx)]
    ni_1 = [0 for k in range(0, Nx)]
    #Psil_1 = e * (Vdc+100*m.sin(13560000*2*m.pi*dt)) / kTe
    Vel = Vdc+100*m.sin(13560000*2*m.pi*dt)

    V_1 = Pois(ne, ni, Vel, dx, Nel, Nx)
    ui_1 = momentum(V_1, ni, ui, mi, kTi, dx, dt, Nel, Nx)
    #Psi_1 = RKPoisN(dx, Psi_1, Nsh, Nx, n0, Ti, Te, Psil_1, FN)
    #for i in range(Nsh, Nx):
        #Ni_1[i] = FN(Psi_1[i])
    """
    for i in range(0, Nx):
        V_1[i] = Psi_1[i]*kTe/e
        ni_1[i] = Ni_1[i]*n0
        ui_1[i] = n0 * m.sqrt(kTi / mi) / ni_1[i]
    #ui_m = momentum(V_1, ni, ui, mi, kTi, boxsize, dt)
    """

    plt.plot(x, Psi)
    plt.ylabel('Psi')
    plt.show()

    plt.plot(x, Ni)
    plt.ylabel('Ni')
    plt.show()

    plt.plot(x, dPsidx)
    plt.ylabel('dPsi/dx')
    plt.show()

    plt.plot(x, V, 'r')
    plt.plot(x, V_1, 'b')
    plt.ylabel('V')
    plt.show()

    plt.plot(x, ne, 'b')
    plt.plot(x, ni, 'r')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, ui, 'r')
    plt.plot(x, ui_1, 'b')
    #plt.plot(x, ui_m, 'g')
    plt.ylabel('u')
    plt.show()

    return 0


if __name__ == "__main__":
    main()