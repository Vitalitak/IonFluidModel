import math as m
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

"""
1D Fluid model of collisionless Ar plasma sheath
Ions are described in fluid model, electrons have Boltzmann distribution
Electrode potential in equivalent circuit model
"""

"""
Initial conditions are given by stationary.py 
Dynamic calculation cyclic substitution of functions values into the system
Electrons have Boltzmann distribution
Electron current is calculated as diode current

"""


def Pois(ne, ni, Vprev, Ve, n0, dx, Nel, Nsh, Nx):

    """
    sweep method solution of Poisson equation
    electrode boundary condition Ve

    """

    e = 1.6E-19
    eps0 = 8.85E-12

    V = [0 for k in range(0, Nx)]


    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nel)]
    b = [0 for k in range(0, Nel)]

    # forward
    # boundary conditions on plasma surface: (dV/dx)pl = 0 or (V)pl = 0

    """
    V[0] = 0
    V[1] = 0

    a[2] = 0
    b[2] = 0.001*dx * dx * e * n0 / eps0

    #a[0] = 0
    #b[0] = 0
    #a[0] = 1
    #b[0] = 0

    for i in range(3, Nel-1):
        a[i] = -1 / (-2+a[i-1])
        b[i] = (-b[i-1] - e / eps0 * (ni[i] - ne[i]) * dx * dx)/(-2+a[i-1])
    """

    for i in range(0, Nsh):
        V[i] = Vprev[i]
        a[i] = 0
        b[i] = 0
    a[Nsh] = 0
    b[Nsh] = Vprev[Nsh]

    for i in range(Nsh+1, Nel - 1):
        a[i] = -1 / (-2 + a[i - 1])
        b[i] = (-b[i - 1] - e / eps0 * (ni[i] - ne[i]) * dx * dx) / (-2 + a[i - 1])

    # boundary condition on electrode surface: (V)el = Ve
    a[Nel-1] = 0
    b[Nel-1] = Ve  #  (V)p = 0

    # backward
    V[Nel-1] = b[Nel-1]
    for i in range(Nel-1, Nsh, -1):
        V[i-1] = a[i-1]*V[i]+b[i-1]

    return V

def momentum(V, n, uprev, kTi, kTe, n0, Nel, Nsh, Nx, dt):

    """
    Explicit conservative upwind scheme
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
    """


    # Explicit conservative upwind scheme
    """
    u[0] = uprev[0]
    u[1] = uprev[1]
    u[2] = uprev[2]
    """

    for i in range(0, Nsh):
        u[i] = uprev[i]

    for i in range(Nsh, Nel):
        u[i] = uprev[i] + dt * (-kTe / mi * (Psi[i] - Psi[i - 1]) / dx - kTi / mi * m.pow(N[i], gamma - 2) * (
                    N[i] - N[i - 1]) / dx - (uprev[i] * uprev[i] - uprev[i - 1] * uprev[i - 1]) / dx)
        #print(dt * (-kTe / mi * (Psi[Nel-1] - Psi[Nel-2]) / dx - kTi / mi * m.pow(N[Nel-1], gamma - 2) * (
                    #N[Nel-1] - N[Nel-2]) / dx - (uprev[Nel-1] * uprev[Nel-1] - uprev[Nel-2] * uprev[Nel-2]) / dx))
    #print(dt * (-kTe / mi * (Psi[Nel - 1] - Psi[Nel - 2]) / dx - kTi / mi * m.pow(N[Nel - 1], gamma - 2) * (
            #N[Nel - 1] - N[Nel - 2]) / dx - (uprev[Nel - 1] * uprev[Nel - 1] - uprev[Nel - 2] * uprev[Nel - 2]) / dx))

    return u


def continuity(u, nprev, Nel, Nsh, Nx, dt):

    """
    Explicit conservative upwind scheme
    """

    #dt = 1E-11  # s
    dx = 1E-7
    n = [0 for k in range(0, Nx)]

    """
    N = [0 for k in range(0, Nel)]
    Nprev = [0 for k in range(0, Nel)]

    for i in range(0, Nel):
        N[i] = n[i]/nprev[0]
        Nprev[i] = nprev[i]/nprev[0]
    """
    """
    # initialisation of sweeping coefficients
    a = [0 for k in range(0, Nel)]
    b = [0 for k in range(0, Nel)]

    # forward
    # boundary conditions on plasma surface: (dn/dt)pl = 0
    #a[0] = u[0] / (-1/dt-(u[1]-u[0])/dx)
    #b[0] = -nprev[0] / (-1-(u[1]-u[0])*dt/dx)
    a[0] = 0
    b[0] = nprev[0]
    #a[0] = 1
    #b[0] = 0
    #b[0] = nprev[0] - dn

    for i in range(1, Nel - 1):
        a[i] = -u[i] / (2*(dx/dt+u[i+1]-u[i]) - u[i]*a[i-1])
        b[i] = (u[i]/2.0/dx*b[i-1]+nprev[i]/dt) / ((1/dt+(u[i+1]-u[i])/dx) - u[i]/2.0/dx*a[i-1])

    # boundary condition on electrode surface: (dn/dx)el = (dn/dx)0
    a[Nel - 1] = 0
    #b[Nx - 1] = (-u[Nx - 1]/2.0/dx*b[Nx-2]-nprev[Nx-1]/dt) / ((-1/dt-(u[Nx-1]-u[Nx-2])/dx) + u[Nx-1]/2.0/dx*a[Nx-2]) # boundary conditions for u (u[Nx-1]-u[Nx-2])
    #b[Nx - 1] = nprev[Nx - 1] + dn  # (dn/dt)p = dn0/dt
    #b[Nx - 1] = nprev[Nx - 1]  # (n)p = np (dn/dt)p = 0
    b[Nel - 1] = (b[Nel - 2] + nprev[Nel - 1] - nprev[Nel - 2]) / (1 - a[Nel - 2])

    # backward
    n[Nel - 1] = b[Nel - 1]
    for i in range(Nel - 1, 0, -1):
        n[i - 1] = a[i - 1] * n[i] + b[i - 1]
    """

    # Explicit conservative upwind scheme
    """
    n[0] = nprev[0]
    n[1] = nprev[1]
    n[2] = nprev[2]
    """
    for i in range(0, Nsh):
        n[i] = nprev[i]

    for i in range(Nsh, Nel):
        n[i] = nprev[i] - dt * ((nprev[i]*u[i]-nprev[i-1]*u[i-1])/dx)
        #print(((nprev[i]-nprev[i-1])*u[i]+(u[i]-u[i-1])*nprev[i]))

    #print(((nprev[3] - nprev[2]) * u[3] + (u[3] - u[2]) * nprev[3]))
    return n

def concentration_e(V, kTe, n0, Nel, Nx):

    """
    Boltzmann distribution for electrons
    """

    dx = 1E-7
    e = 1.6E-19
    n = [0 for k in range(0, Nx)]

    for i in range(0, Nel):
        n[i] = n0 * m.exp(e*V[i]/kTe)

    return n

def main():
    # initialisation of parameters
    boxsize = 3.5E-4  # m
    #a = 1E-6
    dt = 1E-12 # s
    dx = 1E-7
    Nx = int(boxsize/dx)
    Nsh = 1000
    #Nt = 200000
    Nper = 2
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.70  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    C0 = 1e-6 # F
    S = 1e-2 # m^2 electrode area
    C = C0/S
    gamma = 5/3
    Arf = 19
    w = 13560000 # Hz

    Nt = int(Nper / w / dt / 2)

    print(Nt)
    print(int((Nper-2)/w/dt))
    print(int((Nper - 1) / w / dt))

    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J


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

    """
    # initial conditions for ue
    for i in range(0, Nx):
        ue[i] = m.sqrt(kTe / me) * m.sqrt(3+2*e*V[i]/kTe+2*(de+1)/de*(1-m.exp(de*e*V[i]/kTe)))
    """


    # dynamic calculations

    ui_p = [0 for k in range(0, Nx)]
    V_p = [0 for k in range(0, Nx)]
    ni_p = [0 for k in range(0, Nx)]
    ne_p = [0 for k in range(0, Nx)]
    #ue_1 = [0 for k in range(0, Nx)]
    VdcRF = [0 for k in range(0, int(2*Nt+1))]
    Iel = [0 for k in range(0, int(2*Nt+1))]
    Ii = [0 for k in range(0, int(2*Nt+1))]
    VRF = [0 for k in range(0, int(2*Nt+1))]
    P = [0 for k in range(0, int(2*Nt+1))]
    #Pav = [0 for k in range(0, Nper)]
    time = [dt * k for k in range(0, int(2*Nt+1))]
    q = 0
    #Vel = V[Nel-1] - 10 * m.sin(13560000*2*m.pi*dt)+q
    Vel = V[Nel-1]+q

    V_1 = Pois(ne, ni, V, Vel, n0, dx, Nel, Nsh, Nx)
    ui_1 = momentum(V_1, ni, ui, kTi, kTe, n0, Nel, Nsh, Nx, dt)
    #ue_1 = momentum_e(V_1, ne, ue, kTe, de, n0, Nel, Nx, dt)
    ni_1 = continuity(ui_1, ni, Nel, Nsh, Nx, dt)
    ne_1 = concentration_e(V_1, kTe, n0, Nel, Nx)
    #ne_1 = continuity(ue_1, ne, Nel, Nx, dt)
    #q += e*(ni_1[Nel-1]*ui_1[Nel-1]-ne_1[Nel-1]*ue_1[Nel-1])*dt/C

    # electron current in diode model

    if V_1[Nel - 1] < 0:
        q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0]*m.sqrt(kTe/me)/4*m.exp(e*(V_1[Nel - 1]-V_1[0])/kTe)) * dt / C
    else:
        q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(kTe / me) / 4) * dt / C
    VdcRF[0] = q
    Iel[0] = e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0]*m.sqrt(kTe/me)/4*m.exp(e*(V_1[Nel - 1]-V_1[0])/kTe))
    Ii[0] = e * ni_1[Nel - 1] * ui_1[Nel - 1]
    VRF[0] = 0
    #print(e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0]*m.sqrt(kTe/me)/4*m.exp(e*(V_1[Nel - 1]-V_1[0])/kTe)) * dt / C)

    t = 0

    for i in range(1, Nt):
        #print(i)
        t += dt

        Vel2 = V[Nel - 1] + q - Arf * m.sin(w * 2 * m.pi * t)
        #Vel2 = V[Nel-1] + q - Arf * m.sin(1e-3 * 2 * m.pi * (2 * i - 1))
        #Vel2 = V[Nel-1] + q

        V_2 = Pois(ne_1, ni_1, V_1, Vel2, n0, dx, Nel, Nsh, Nx)
        ui_2 = momentum(V_2, ni_1, ui_1, kTi, kTe, n0, Nel, Nsh, Nx, dt)
        #ue_2 = momentum_e(V_2, ne_1, ue_1, kTe, de, n0, Nel, Nx, dt)
        ni_2 = continuity(ui_2, ni_1, Nel, Nsh, Nx, dt)
        ne_2 = concentration_e(V_2, kTe, n0, Nel, Nx)
        #ne_2 = continuity(ue_2, ne_1, Nel, Nx, dt)
        #q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[Nel - 1] * ue_2[Nel - 1])*dt / C
        if V_2[Nel - 1]<0:
            q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(kTe / me) / 4 * m.exp(
            e * (V_2[Nel - 1] - V_2[0]) / kTe)) * dt / C
        else:
            q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(kTe / me) / 4) * dt / C
        VdcRF[int(2 * i - 1)] = q
        Iel[int(2 * i - 1)] = e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(kTe / me) / 4 * m.exp(
            e * (V_2[Nel - 1] - V_2[0]) / kTe))
        Ii[int(2 * i - 1)] = e * ni_2[Nel - 1] * ui_2[Nel - 1]
        VRF[int(2 * i - 1)] = - Arf * m.sin(w * 2 * m.pi * t)
        #print(e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(3*kTe / me) / 4 * m.exp(
            #e * (V_2[Nel - 1] - V_2[0]) / kTe)) * dt / C)


        t += dt
        Vel3 = V[Nel - 1] + q - Arf * m.sin(w * 2 * m.pi * t)
        #Vel3 = V[Nel-1] + q - Arf * m.sin(1e-3 * 2 * m.pi * (2 * i))
        #Vel3 = V[Nel - 1] + q

        V_1 = Pois(ne_2, ni_2, V_2, Vel3, n0, dx, Nel, Nsh, Nx)
        ui_1 = momentum(V_1, ni_2, ui_2, kTi, kTe, n0, Nel, Nsh, Nx, dt)
        #ue_1 = momentum_e(V_1, ne_2, ue_2, kTe, de, n0, Nel, Nx, dt)
        ni_1 = continuity(ui_1, ni_2, Nel, Nsh, Nx, dt)
        ne_1 = concentration_e(V_1, kTe, n0, Nel, Nx)
        #ne_1 = continuity(ue_1, ne_2, Nel, Nx, dt)

        """
        ne_1 = continuity(ue_2, ne_2, Nel, Nx, dt)
        ni_1 = continuity(ui_2, ni_2, Nel, Nx, dt)
        ue_1 = momentum_e(V_2, ne_1, ue_2, kTe, de, n0, Nel, Nx, dt)
        ui_1 = momentum(V_2, ni_1, ui_2, kTi, kTe, n0, Nel, Nx, dt)
        V_1 = Pois(ne_1, ni_1, Vel3, dx, Nel, Nx)
        """
        #q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[Nel - 1] * ue_1[Nel - 1])*dt / C
        if V_1[Nel - 1] < 0:
            q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(kTe / me) / 4 * m.exp(
            e * (V_1[Nel - 1]-V_1[0]) / kTe)) * dt / C
        else:
            q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(kTe / me) / 4) * dt / C

        VdcRF[int(2 * i)] = q
        VRF[int(2 * i)] = - Arf * m.sin(w * 2 * m.pi * t)
        Iel[int(2 * i)] = e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(kTe / me) / 4 * m.exp(
            e * (V_1[Nel - 1]-V_1[0]) / kTe))
        Ii[int(2 * i)] = e * ni_1[Nel - 1] * ui_1[Nel - 1]
        #print(e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[Nel - 1] * ue_1[Nel - 1])*dt / C)

    for i in range(0, int(2*Nt+1)):
        P[i] = Iel[i] * S * VdcRF[i]
    """
    for j in range(0, Nper-1):
        for i in range(int(j/w/dt), int((j+1)/w/dt)):
            Pav[j] += 0.5*(P[i]+P[i+1]) * dt
        Pav[j] = w * Pav[j]

    #Pav = Pav * w
    print(Pav[Nper-1])
    """
    # graph plot
    """
    Ii = [0 for k in range(0, Nx)]
    Ii_1 = [0 for k in range(0, Nx)]
    Ie = [0 for k in range(0, Nx)]

    for i in range(0, Nx):
        Ii[i] = ni[i]*ui[i]
        Ii_1[i] = ni_1[i]*ui_1[i]
        #print(ni_1[i]*ui_1[i])

    #print(Ii[Nel-1]-Ie[Nel-1])
    """

    plt.plot(time, Ii, 'r')
    plt.plot(time, Iel, 'b')
    plt.ylabel('j, A/m2')
    plt.show()

    plt.plot(time, P, 'b')
    plt.ylabel('P, W')
    plt.show()

    plt.plot(x, V, 'r')
    plt.plot(x, V_1, 'b')
    #plt.plot(x, V_2, 'g')
    #plt.plot(x, V_3, 'm')
    plt.ylabel('V')
    plt.show()

    plt.plot(time, VdcRF, 'r')
    plt.plot(time, VRF, 'b')
    plt.ylabel('V')
    #plt.axis([-1e-9, 5e-7, -13, 12])
    plt.grid(visible='True', which='both', axis='y')
    plt.show()

    f = open("VDC.txt", "w")
    for d in VdcRF:
        f.write(f"{d}\n")
    f.close()
    """
    f = open("P.txt", "w")
    for d in Pav:
        f.write(f"{d}\n")
    f.close()
    """
    plt.plot(x, ni, 'r--')
    plt.plot(x, ni_1, 'r-')
    plt.plot(x, ne, 'b--')
    plt.plot(x, ne_1, 'b-')
    #plt.plot(x, ni_2, 'g')
    #plt.plot(x, ni_3, 'm')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, ui, 'r')
    plt.plot(x, ui_1, 'b')
    #plt.plot(x, ui_2, 'g')
    #plt.plot(x, ui_3, 'm')
    plt.ylabel('u')
    plt.show()


    """
    cur = [0 for i in range(0, Nx)]
    for i in range(0, Nel):
        cur[i] = ni_1[i] * ui_1[i] - ne_1[i] * ue_1[i]

    plt.plot(x, cur, 'r')
    plt.show()
    """
    return 0


if __name__ == "__main__":
    main()