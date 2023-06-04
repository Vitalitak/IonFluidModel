import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

def main():
    e = 1.6E-19
    eps0 = 8.85E-12
    dx = 1E-6
    boxsize = 1E-4  # m
    Nx = int(boxsize / dx)

    # plasma parameters
    Te = 2.70  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    gamma = 5 / 3
    kTe = Te * 1.6E-19

    #Psi0 = -1e-16
    Psi0 = -1e-16
    dPsi = 5e-3
    Psil = e * Vdc / kTe

    FPsi = lambda x: (5 * gamma - 3) * Ti / Te / 2 / (gamma - 1) * (
                1 - 3 * (gamma - 1) / (5 * gamma - 3) * m.pow(x, -2) - 2 * gamma / (5 * gamma - 3) * m.pow(x,
                                                                                                           gamma - 1)) + Psi0

    FN = inversefunc(FPsi, domain=[0.000000001, 1])

    Psi = [Psi0-k*dPsi for k in range(0, Nx)]
    Integ = [0 for k in range(0, Nx)]
    Integ_old = [0 for k in range(0, Nx)]
    rightpart = [0 for k in range(0, Nx)]
    rightpart_old = [0 for k in range(0, Nx)]
    N = [0 for k in range(0, Nx)]
    dInteg = [0 for k in range(0, Nx)]

    Psi[0] = Psi0

    #i = 0
    # for i in range(Nsh+2, Nx-1):
    #while (Psi[i] > Psil) and (i < Nx - 1):

    for i in range(0, Nx):
        Integ[i] = quad(FN, Psi[i], Psi0)[0]
        rightpart[i] = m.exp(Psi0) - m.exp(Psi[i]) - Integ[i]
        Integ_old[i] = quad(FN, Psi0, Psi[i])[0]
        rightpart_old[i] = -(m.exp(Psi[i]) - m.exp(Psi0) - Integ_old[i])
        N[i] = FN(Psi[i])

    dInteg[0] = N[0]
    for i in range(1, Nx):
        dInteg[i] = (Integ[i] - Integ[i-1])/(-dPsi)

    plt.plot(Psi, Integ, 'r')
    #plt.plot(Psi, N, 'b')
    plt.plot(Psi, rightpart, 'm')
    #plt.plot(Psi, dInteg, 'g')
    plt.plot(Psi, Integ_old, 'b')
    plt.plot(Psi, rightpart_old, 'g')

    #plt.ylabel('d2V/dx2')
    plt.show()

    return 0

if __name__ == "__main__":
        main()