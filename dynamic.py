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

    Nel = i+1

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
    # boundary conditions on plasma surface: (dV/dx)pl = 0
    #a[0] = 0.5
    #b[0] = 0.5 * (ne[0] - ni[0]) * dx * dx
    #a[0] = 0
    #b[0] = 0
    a[0] = 1
    b[0] = 0

    for i in range(1, Nel-1):
        a[i] = -1 / (-2+a[i-1])
        b[i] = (-b[i-1] - e / eps0 * (ni[i] - ne[i]) * dx * dx)/(-2+a[i-1])

    # boundary condition on electrode surface: (V)el = Ve
    a[Nel-1] = 0
    #b[Nx-1] = (b[Nx-2] - (ne[Nx-1] - ni[Nx-1]) * dx * dx)/(2-a[Nx-2])
    b[Nel-1] = Ve  #  (V)p = 0
    #b[Nx - 1] = b[Nx - 2] / (1 - a[Nx - 2])  # (dV/dx)p = 0

    # backward
    V[Nel-1] = b[Nel-1]
    for i in range(Nel-1, 0, -1):
        V[i-1] = a[i-1]*V[i]-b[i-1]

    return V

def momentum(V, n, uprev, kTi, kTe, n0, Nel, Nx, dt):

    """
    sweep method solution of momentum balance equation
    """
    #dt = 1E-11  # s
    dx = 1E-7
    e = 1.6E-19
    mi = 6.68E-26  # kg
    gamma = 5 / 3
    u = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nel)]
    N = [0 for k in range(0, Nel)]

    for i in range(0, Nel):
        Psi[i] = e*V[i]/kTe
        N[i] = n[i]/n0

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
        a[i] = -uprev[i+1]*dt / 4.0 / dx / (1 - uprev[i - 1]*dt * a[i-1] / 4.0 / dx)
        b[i] = (uprev[i-1]*dt / 4.0 / dx * b[i - 1] - kTe/mi*dt*(Psi[i+1]-Psi[i]) /dx - kTi/mi*dt*m.pow(N[i], gamma-2)*(N[i+1]-N[i])/dx + uprev[i]) / (1 - uprev[i - 1]*dt * a[i-1] / 4.0 / dx)

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

def momentum_e(V, n, uprev, kTe, de, n0, Nel, Nx, dt):

    """
    sweep method solution of momentum balance equation
    """
    #dt = 1E-11  # s
    dx = 1E-7
    e = 1.6E-19
    me = 9.11E-31  # kg
    gamma = 1+de
    u = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nel)]
    N = [0 for k in range(0, Nel)]

    for i in range(0, Nel):
        Psi[i] = e*V[i]/kTe
        N[i] = n[i]/n0
    """
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
        a[i] = -uprev[i+1]*dt / 4.0 / dx / (1 - uprev[i - 1]*dt * a[i-1] / 4.0 / dx)
        b[i] = (uprev[i-1]*dt / 4.0 / dx * b[i - 1] + kTe/me*dt*(Psi[i+1]-Psi[i]) /dx - kTe/me*dt*m.pow(N[i], gamma-2)*(N[i+1]-N[i])/dx + uprev[i]) / (1 - uprev[i - 1]*dt * a[i-1] / 4.0 / dx)
        #print(b[i])

    # boundary condition on electrode surface: (du/dx)el = 0
    a[Nel - 1] = 0
    #b[Nx - 1] = (-uprev[Nx-2] / 4.0 / dx * b[Nx-2] + (V[Nx-1]-V[Nx-2])/dx - uprev[Nx-1] / dt) / (-1 / dt + uprev[Nx-2] * a[Nx-2] / 4.0 / dx)  # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = 0  # (u)p = 0
    b[Nel - 1] = b[Nel - 2]/(1 - a[Nel - 2])  # (du/dx)el = 0

    # backward
    u[Nel - 1] = b[Nel - 1]
    for i in range(Nel - 1, 0, -1):
        u[i - 1] = a[i - 1] * u[i] + b[i - 1]
    """
    # Explicit conservative upwind scheme

    u[0] = m.sqrt(3*kTe/me)

    for i in range(1, Nx-1):
        u[i] = uprev[i]+dt*(kTe/me*dt*(Psi[i]-Psi[i-1]) /dx - kTe/me*dt*m.pow(N[i], gamma-2)*(N[i]-N[i-1])/dx-(uprev[i]*uprev[i]-uprev[i-1]*uprev[i-1])/dx)
        print(u[i])


    return u

def continuity(u, nprev, Nel, Nx, dt):

    """
    sweep method solution of continuity equation
    """

    #dt = 1E-11  # s
    dx = 1E-7
    n = [0 for k in range(0, Nx)]

    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nel)]
    b = [0 for k in range(0, Nel)]

    # forward
    # boundary conditions on plasma surface: (dn/dt)pl = 0
    #a[0] = u[0] / (-1/dt-(u[1]-u[0])/dx)
    #b[0] = -nprev[0] / (-1-(u[1]-u[0])*dt/dx)
    #a[0] = 0
    #b[0] = nprev[0]
    a[0] = 1
    b[0] = 0
    #b[0] = nprev[0] - dn

    for i in range(1, Nel - 1):
        a[i] = -u[i] / (2*(dx/dt+u[i+1]-u[i]) - u[i]*a[i-1])
        b[i] = (u[i]/2.0/dx*b[i-1]+nprev[i]/dt) / ((1/dt+(u[i+1]-u[i])/dx) - u[i]/2.0/dx*a[i-1])

    # boundary condition on electrode surface: (dn/dt)el = 0
    a[Nel - 1] = 0
    #b[Nx - 1] = (-u[Nx - 1]/2.0/dx*b[Nx-2]-nprev[Nx-1]/dt) / ((-1/dt-(u[Nx-1]-u[Nx-2])/dx) + u[Nx-1]/2.0/dx*a[Nx-2]) # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = nprev[Nx - 1] + dn  # (dn/dt)p = dn0/dt
    #b[Nx - 1] = nprev[Nx - 1]  # (n)p = np (dn/dt)p = 0
    b[Nel - 1] = b[Nel - 2] / (1 - a[Nel - 2])

    # backward
    n[Nel - 1] = b[Nel - 1]
    for i in range(Nel - 1, 0, -1):
        n[i - 1] = a[i - 1] * n[i] + b[i - 1]

    return n

def main():
    # initialisation of parameters
    boxsize = 3.5E-4  # m
    #a = 1E-6
    dt = 1E-14 # s
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
    C = 1E-10
    #C /= 1.6E-19
    gamma = 5/3
    de = 0.23277685


    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    #V0 = 0
    # V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    Nt = int(tEnd / dt)

    # stationary system for initial conditions

    # read initial conditions from file

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]

    i=0

    with open("V.txt", "r") as f1:
        for line in f1.readlines():
            for ind in line.split():
                V[i] = float(ind)
                i +=1
    f1.close()
    i = 0
    with open("ni.txt", "r") as f2:
        for line in f2.readlines():
            for ind in line.split():
                ni[i] = float(ind)
                i += 1
    f2.close()
    i = 0

    with open("ne.txt", "r") as f3:
        for line in f3.readlines():
            for ind in line.split():
                ne[i] = float(ind)
                i += 1
    f3.close()
    i = 0

    with open("ui.txt", "r") as f4:
        for line in f4.readlines():
            for ind in line.split():
                ui[i] = float(ind)
                i += 1
    f4.close()
    i = 0

    with open("Nel.txt", "r") as f5:
        for line in f5.readlines():
            Nel = int(line)
    f5.close()

    # initial conditions for ue
    for i in range(0, Nx):
        ue[i] = m.sqrt(kTe / me) * m.sqrt(3+2*e*V[i]/kTe+2*(de+1)/de*(1-m.exp(de*e*V[i]/kTe)))


    """
    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]
    ue = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    dPsidx = [0 for k in range(0, Nx)]
    Vrf = 0

    Psil = e*Vdc/kTe

    FPsi = lambda x: (5*gamma-3)*Ti/Te/2/(gamma-1)*(1-3*(gamma-1)/(5*gamma-3)*m.pow(x, -2)-2*gamma/(5*gamma-3)*m.pow(x, gamma-1))

    FN = inversefunc(FPsi, domain=[0.001, 1])

    Psi, Nel = RKPoisN(dx, Psi, Nsh, Nx, n0, Ti, Te, Psil, FN)

    for i in range(Nsh, Nx):
        Ni[i] = FN(Psi[i])

    for i in range(Nsh, Nx-1):
        dPsidx[i] = (Psi[i+1]-Psi[i])/dx

    for i in range(0, Nx):
        V[i] = Psi[i]*kTe/e
        ni[i] = Ni[i]*n0
        ne[i] = n0*m.exp(e*V[i]/kTe)
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
        ue[i] = n0 * m.sqrt(kTe / me) / ne[i]
    """

    # dynamic calculations

    ui_1 = [0 for k in range(0, Nx)]
    V_1 = [0 for k in range(0, Nx)]
    ni_1 = [0 for k in range(0, Nx)]
    ne_1 = [0 for k in range(0, Nx)]
    ue_1 = [0 for k in range(0, Nx)]
    q = 0
    #Vel = V[Nel-1] - 10 * m.sin(13560000*2*m.pi*dt)+q
    Vel = V[Nel-1]+q

    V_1 = Pois(ne, ni, Vel, dx, Nel, Nx)
    ui_1 = momentum(V_1, ni, ui, kTi, kTe, n0, Nel, Nx, dt)
    ue_1 = momentum_e(V_1, ne, ue, kTe, de, n0, Nel, Nx, dt)
    ni_1 = continuity(ui_1, ni, Nel, Nx, dt)
    ne_1 = continuity(ue_1, ne, Nel, Nx, dt)
    q += e*(ni_1[Nel-1]*ui_1[Nel-1]-ne_1[Nel-1]*ue_1[Nel-1])*dt/C

    #for i in range(0, Nel):
        #ne_1[i] = n0*m.exp(e*V_1[i]/kTe)

    for i in range(2, 50):
        print(q)
        #Vel2 = V[Nel-1] - 10 * m.sin(13560000 * 2 * m.pi * i / 2 * dt)+q
        Vel2 = V[Nel-1] + q

        V_2 = Pois(ne_1, ni_1, Vel2, dx, Nel, Nx)
        ui_2 = momentum(V_2, ni_1, ui_1, kTi, kTe, n0, Nel, Nx, dt)
        ue_2 = momentum_e(V_2, ne_1, ue_1, kTe, de, n0, Nel, Nx, dt)
        ni_2 = continuity(ui_2, ni_1, Nel, Nx, dt)
        ne_2 = continuity(ue_2, ne_1, Nel, Nx, dt)
        q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[Nel - 1] * ue_2[Nel - 1])*dt / C



        #Vel3 = V[Nel - 1] - 10 * m.sin(13560000 * 2 * m.pi * (i + 1) / 2 * dt)+q
        Vel3 = V[Nel-1] + q

        V_1 = Pois(ne_2, ni_2, Vel3, dx, Nel, Nx)
        ui_1 = momentum(V_1, ni_2, ui_2, kTi, kTe, n0, Nel, Nx, dt)
        ue_1 = momentum_e(V_1, ne_2, ue_2, kTe, de, n0, Nel, Nx, dt)
        ni_1 = continuity(ui_1, ni_2, Nel, Nx, dt)
        ne_1 = continuity(ue_1, ne_2, Nel, Nx, dt)
        q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[Nel - 1] * ue_1[Nel - 1])*dt / C


    """
    #Psi_1 = RKPoisN(dx, Psi_1, Nsh, Nx, n0, Ti, Te, Psil_1, FN)
    #for i in range(Nsh, Nx):
        #Ni_1[i] = FN(Psi_1[i])
    
    
    for i in range(0, Nx):
        V_1[i] = Psi_1[i]*kTe/e
        ni_1[i] = Ni_1[i]*n0
        ui_1[i] = n0 * m.sqrt(kTi / mi) / ni_1[i]
    """
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
    """
    Ii = [0 for k in range(0, Nx)]
    Ie = [0 for k in range(0, Nx)]

    for i in range(0, Nx):
        Ii[i] = ni[i]*ui[i]
        Ie[i] = ne[i]*ue[i]

    plt.plot(x, Ii, 'r')
    plt.plot(x, Ie, 'b')
    plt.ylabel('I')
    plt.show()

    plt.plot(x, V, 'r')
    plt.plot(x, V_1, 'b')
    #plt.plot(x, V_2, 'g')
    #plt.plot(x, V_3, 'm')
    plt.ylabel('V')
    plt.show()

    plt.plot(x, ni, 'r')
    plt.plot(x, ni_1, 'b')
    #plt.plot(x, ni_2, 'g')
    #plt.plot(x, ni_3, 'm')
    plt.ylabel('Ni')
    plt.show()

    plt.plot(x, ne, 'r')
    plt.plot(x, ne_1, 'b')
    #plt.plot(x, ne_2, 'g')
    #plt.plot(x, ne_3, 'm')
    plt.ylabel('Ne')
    plt.show()

    plt.plot(x, ui, 'r')
    plt.plot(x, ui_1, 'b')
    #plt.plot(x, ui_2, 'g')
    #plt.plot(x, ui_3, 'm')
    plt.ylabel('u')
    plt.show()

    plt.plot(x, ue, 'r')
    plt.plot(x, ue_1, 'b')
    #plt.plot(x, ue_2, 'g')
    #plt.plot(x, ue_3, 'm')
    plt.ylabel('u')
    plt.show()

    return 0


if __name__ == "__main__":
    main()