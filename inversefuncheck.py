import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

def main():
    gamma = 5/3
    Te = 2.68  # eV
    Ti = 0.05  # eV
    dPsi = -0.01
    dN = 0.001
    #Psi0 = -0.5
    V0 = -2.68
    Psi0 = 1e-7

    FPsi = lambda x: (5 * gamma - 3) * Ti / Te / 2 / (gamma - 1) * (
                1 - 3 * (gamma - 1) / (5 * gamma - 3) * m.pow(x, -2) - 2 * gamma / (5 * gamma - 3) * m.pow(x,
                                                                                                           gamma - 1)) + Psi0
    FN = inversefunc(FPsi, domain=[0.001, 1])

    Ni = [0 for k in range(0, 1000)]
    Ne = [0 for k in range(0, 1000)]
    Psi = [k * dPsi+Psi0 for k in range(0, 1000)]
    V = [0 for k in range(0, 1000)]
    N = [1-k*dN for k in range(0, 100)]
    Psii = [0 for k in range(0, 100)]
    Psie = [0 for k in range(0, 100)]

    for i in range(0, 1000):
        Ni[i] = FN(Psi[i])
        #print(Ni[i])
        Ne[i] = m.exp(Psi[i])
        V[i] = Psi[i]*Te

    for i in range(0, 100):
        Psii[i] = FPsi(N[i])
        Psie[i] = np.log(N[i])

    plt.plot(V, Ne, 'b')
    plt.plot(V, Ni, 'r')
    plt.ylabel('N')
    plt.xlabel('V')
    plt.show()

    plt.plot(N, Psii, 'r')
    plt.plot(N, Psie, 'b')
    plt.ylabel('Psi')
    plt.xlabel('N')
    plt.show()

    return 0


if __name__ == "__main__":
    main()