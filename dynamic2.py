import math as m
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from scipy.integrate import quad
# from sympy import Function, dsolve, Eq, Derivative, exp, symbols
# from sympy.solvers.ode.systems import dsolve_system
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


def Pois(ne, V, n0, dx, Nel, kTe, Nx):
    """
    sweep method solution of Poisson equation
    electrode boundary condition Ve

    """

    e = 1.6E-19
    eps0 = 8.85E-12

    # V = [0 for k in range(0, Nx)]
    ni = np.zeros(Nx)

    # initialisation of sweeping coefficients
    # a = [0 for k in range(0, Nel)]
    # b = [0 for k in range(0, Nel)]
    #a = np.zeros(Nel)
    #b = np.zeros(Nel)

    # forward
    # boundary conditions on plasma surface: (dV/dx)pl = 0 or (V)pl = 0

    ni[0] = n0

    for i in range(1, Nel):
        ni[i] = n0*m.exp(e*V[i]/kTe)-eps0*kTe/e/e*(-V[i-1]+2*V[i]-V[i+1])/dx/dx


    return ni


def momentum(n, u, uprev, kTi, kTe, n0, Nel, Nsh, Nx, dt):
    """
    Explicit conservative upwind scheme
    """
    # dt = 1E-11  # s
    dx = 1E-7
    e = 1.6E-19
    mi = 6.68E-26  # kg
    gamma = 5 / 3
    # u = [0 for k in range(0, Nx)]
    V = np.zeros(Nx)

    # Psi = [0 for k in range(0, Nel)]
    # N = [0 for k in range(0, Nel)]
    Psi = np.zeros(Nx)
    N = np.zeros(Nx)

    """
    for i in range(0, Nel):
        Psi[i] = e*V[i]/kTe
        N[i] = n[i]/n0
    """
    #Psi = e * V / kTe
    N = n / n0


    Psi[0] = 0
    # print(Psi[Nsh-1:Nel-1])
    # print(Psi[Nsh:Nel])
    # print(N[Nsh:Nel] ** (gamma - 2))
    Psi[1:Nel] = Psi[0:Nel-1] - dx * mi / kTe * ((u[1:Nel]-uprev[1:Nel])/dt+u[1:Nel]*(u[1:Nel]-u[0:Nel-1])/dx+kTi/mi*(N[1:Nel] ** (gamma - 2)) * (
                    N[1:Nel] - N[0:Nel-1]) / dx)

    V = Psi*kTe/e

    return V


def continuity(n, nprev, uprev, Nel, Nsh, Nx, dt):


    # dt = 1E-11  # s
    dx = 1E-7
    # n = [0 for k in range(0, Nx)]
    u = np.zeros(Nx)


    u[0] = uprev[0]

    u[1:Nel] = -((n[1:Nel]-nprev[1:Nel])/dt+nprev[1:Nel]*(uprev[1:Nel]-uprev[0:Nel-1])/dx)*dx/(n[1:Nel]-n[0:Nel-1])


    return u


def concentration_e(V, kTe, n0, Nel, Nx):
    """
    Boltzmann distribution for electrons
    """

    dx = 1E-7
    e = 1.6E-19
    # n = [0 for k in range(0, Nx)]
    n = np.zeros(Nx)
    """
    for i in range(0, Nel):
        n[i] = n0 * m.exp(e*V[i]/kTe)
    """
    n[0:Nel] = n0 * m.e ** (e * V[0:Nel] / kTe)

    return n


def main():
    # initialisation of parameters
    boxsize = 4E-4  # m
    # a = 1E-6
    dt = 1E-12  # s
    dx = 1E-7
    Nx = int(boxsize / dx)
    Nsh = 1
    # Nt = 200000
    Nper = 0.2
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.78  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    C0 = 3e-6  # F
    S = 1e-2  # m^2 electrode area
    C = C0 / S
    gamma = 5 / 3
    Arf = 20
    w = 13560000  # Hz

    #Nt = int(Nper / w / dt / 2)
    Nt = 0

    print(Nt)
    print(int((Nper - 2) / w / dt))
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

    i = 0

    with open("V.txt", "r") as f1:
        for line in f1.readlines():
            for ind in line.split():
                V[i] = float(ind)
                i += 1
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

    x = np.array(x)
    V = np.array(V)
    ni = np.array(ni)
    ne = np.array(ne)
    ui = np.array(ui)

    # dynamic calculations

    ui_p = [0 for k in range(0, Nx)]
    V_p = [0 for k in range(0, Nx)]
    ni_p = [0 for k in range(0, Nx)]
    ne_p = [0 for k in range(0, Nx)]
    # ue_1 = [0 for k in range(0, Nx)]
    """
    VdcRF = [0 for k in range(0, int(2*Nt+1))]
    Iel = [0 for k in range(0, int(2*Nt+1))]
    Ii = [0 for k in range(0, int(2*Nt+1))]
    VRF = [0 for k in range(0, int(2*Nt+1))]
    P = [0 for k in range(0, int(2*Nt+1))]
    Pav = [0 for k in range(0, Nper)]
    time = [dt * k for k in range(0, int(2*Nt+1))]
    """

    VdcRF = np.zeros(int(2 * Nt + 1))
    Iel = np.zeros(int(2 * Nt + 1))
    Ii = np.zeros(int(2 * Nt + 1))
    VRF = np.zeros(int(2 * Nt + 1))
    P = np.zeros(int(2 * Nt + 1))
    Pav = np.zeros(int(Nper))
    time = np.arange(2 * Nt + 1) * dt

    q = 0
    # Vel = V[Nel-1] - 10 * m.sin(13560000*2*m.pi*dt)+q
    Vel = V[Nel - 1] + q

    ni_1 = Pois(ne, V, n0, dx, Nel, kTe, Nx)
    ui_1 = continuity(ni_1, ni, ui, Nel, Nsh, Nx, dt)
    V_1 = momentum(ni_1, ui_1, ui, kTi, kTe, n0, Nel, Nsh, Nx, dt)
    #ne_1 = concentration_e(V_1, kTe, n0, Nel, Nx)


    # electron current in diode model

    if V_1[Nel - 1] < 0:
        q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - n0 * m.sqrt(3 * kTe / me) / 4 * m.exp(
            e * (V_1[Nel - 1] - V_1[0]) / kTe)) * dt / C
    else:
        q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - n0 * m.sqrt(3 * kTe / me) / 4) * dt / C
    VdcRF[0] = q
    Iel[0] = e * (ni_1[Nel - 1] * ui_1[Nel - 1] - n0 * m.sqrt(3 * kTe / me) / 4 * m.exp(
        e * (V_1[Nel - 1] - V_1[0]) / kTe))
    Ii[0] = e * ni_1[Nel - 1] * ui_1[Nel - 1]
    VRF[0] = 0
    # print(e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0]*m.sqrt(kTe/me)/4*m.exp(e*(V_1[Nel - 1]-V_1[0])/kTe)) * dt / C)

    t = 0

    for i in range(1, Nt):
        # print(i)
        t += dt

        Vel2 = V[Nel - 1] + q - Arf * m.sin(w * 2 * m.pi * t)
        # Vel2 = V[Nel-1] + q - Arf * m.sin(1e-3 * 2 * m.pi * (2 * i - 1))
        # Vel2 = V[Nel-1] + q

        V_2 = Pois(ne_1, ni_1, V_1, Vel2, n0, dx, Nel, Nsh, Nx)
        ui_2 = momentum(V_2, ni_1, ui_1, kTi, kTe, n0, Nel, Nsh, Nx, dt)
        # ue_2 = momentum_e(V_2, ne_1, ue_1, kTe, de, n0, Nel, Nx, dt)
        ni_2 = continuity(ui_2, ni_1, Nel, Nsh, Nx, dt)
        ne_2 = concentration_e(V_2, kTe, n0, Nel, Nx)
        # ne_2 = continuity(ue_2, ne_1, Nel, Nx, dt)
        # q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[Nel - 1] * ue_2[Nel - 1])*dt / C
        if V_2[Nel - 1] < 0:
            q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(3 * kTe / me) / 4 * m.exp(
                e * (V_2[Nel - 1] - V_2[0]) / kTe)) * dt / C
        else:
            q += e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(3 * kTe / me) / 4) * dt / C
        VdcRF[int(2 * i - 1)] = q
        Iel[int(2 * i - 1)] = e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(3 * kTe / me) / 4 * m.exp(
            e * (V_2[Nel - 1] - V_2[0]) / kTe))
        Ii[int(2 * i - 1)] = e * ni_2[Nel - 1] * ui_2[Nel - 1]
        VRF[int(2 * i - 1)] = - Arf * m.sin(w * 2 * m.pi * t)
        # print(e * (ni_2[Nel - 1] * ui_2[Nel - 1] - ne_2[0] * m.sqrt(3*kTe / me) / 4 * m.exp(
        # e * (V_2[Nel - 1] - V_2[0]) / kTe)) * dt / C)

        t += dt
        Vel3 = V[Nel - 1] + q - Arf * m.sin(w * 2 * m.pi * t)
        # Vel3 = V[Nel-1] + q - Arf * m.sin(1e-3 * 2 * m.pi * (2 * i))
        # Vel3 = V[Nel - 1] + q

        V_1 = Pois(ne_2, ni_2, V_2, Vel3, n0, dx, Nel, Nsh, Nx)
        ui_1 = momentum(V_1, ni_2, ui_2, kTi, kTe, n0, Nel, Nsh, Nx, dt)
        # ue_1 = momentum_e(V_1, ne_2, ue_2, kTe, de, n0, Nel, Nx, dt)
        ni_1 = continuity(ui_1, ni_2, Nel, Nsh, Nx, dt)
        ne_1 = concentration_e(V_1, kTe, n0, Nel, Nx)
        # ne_1 = continuity(ue_1, ne_2, Nel, Nx, dt)

        """
        ne_1 = continuity(ue_2, ne_2, Nel, Nx, dt)
        ni_1 = continuity(ui_2, ni_2, Nel, Nx, dt)
        ue_1 = momentum_e(V_2, ne_1, ue_2, kTe, de, n0, Nel, Nx, dt)
        ui_1 = momentum(V_2, ni_1, ui_2, kTi, kTe, n0, Nel, Nx, dt)
        V_1 = Pois(ne_1, ni_1, Vel3, dx, Nel, Nx)
        """
        # q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[Nel - 1] * ue_1[Nel - 1])*dt / C
        if V_1[Nel - 1] < 0:
            q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(3 * kTe / me) / 4 * m.exp(
                e * (V_1[Nel - 1] - V_1[0]) / kTe)) * dt / C
        else:
            q += e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(3 * kTe / me) / 4) * dt / C

        VdcRF[int(2 * i)] = q
        VRF[int(2 * i)] = - Arf * m.sin(w * 2 * m.pi * t)
        Iel[int(2 * i)] = e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[0] * m.sqrt(3 * kTe / me) / 4 * m.exp(
            e * (V_1[Nel - 1] - V_1[0]) / kTe))
        Ii[int(2 * i)] = e * ni_1[Nel - 1] * ui_1[Nel - 1]
        # print(e * (ni_1[Nel - 1] * ui_1[Nel - 1] - ne_1[Nel - 1] * ue_1[Nel - 1])*dt / C)

    for i in range(0, int(2 * Nt + 1)):
        P[i] = Iel[i] * S * VdcRF[i]
    """
    for j in range(0, Nper-1):
        for i in range(int(j/w/dt), int((j+1)/w/dt)):
            Pav[j] += 0.5*(P[i]+P[i+1]) * dt
        Pav[j] = w * Pav[j]
    """
    # Pav = Pav * w
    # print(Pav[Nper-1])
    NdV2 = [0 for k in range(0, Nx)]
    dV2 = [0 for k in range(0, Nx)]
    """
    for i in range(1, Nx - 2):
        NdV2[i] = - e / eps0 * (ni_1[i] - ne_1[i])
        dV2[i] = (-V_1[i - 1] + 2 * V_1[i] - V_1[i + 1]) / dx / dx

    plt.plot(x, NdV2, 'r')
    plt.plot(x, dV2, 'b')

    plt.ylabel('d2V/dx2')
    plt.show()
    """
    # graph plot

    plt.plot(time, Ii, 'r')
    plt.plot(time, Iel, 'b')
    plt.ylabel('j, A/m2')
    plt.show()

    plt.plot(time, P, 'b')
    plt.ylabel('P, W')
    plt.show()

    plt.plot(x, V, 'r')
    plt.plot(x, V_1, 'b')
    # plt.plot(x, V_2, 'g')
    # plt.plot(x, V_3, 'm')
    plt.ylabel('V')
    plt.show()

    plt.plot(time, VdcRF, 'r')
    plt.plot(time, VRF, 'b')
    plt.ylabel('V')
    # plt.axis([-1e-9, 5e-7, -13, 12])
    plt.grid(visible='True', which='both', axis='y')
    plt.show()
    """
    f = open("VDC.txt", "w")
    for d in VdcRF:
        f.write(f"{d}\n")
    f.close()

    f = open("P.txt", "w")
    for d in Pav:
        f.write(f"{d}\n")
    f.close()
    """
    plt.plot(x, ni, 'r--')
    plt.plot(x, ni_1, 'r-')
    plt.plot(x, ne, 'b--')
    #plt.plot(x, ne_1, 'b-')
    # plt.plot(x, ni_2, 'g')
    # plt.plot(x, ni_3, 'm')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, ui, 'r')
    plt.plot(x, ui_1, 'b')
    # plt.plot(x, ui_2, 'g')
    # plt.plot(x, ui_3, 'm')
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