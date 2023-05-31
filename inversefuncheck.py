import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

def main():
    gamma = 5/3
    Te = 2.70  # eV
    Ti = 0.05  # eV
    dPsi = -0.01
    Psi0 = -1e-7

    FPsi = lambda x: (5 * gamma - 3) * Ti / Te / 2 / (gamma - 1) * (
                1 - 3 * (gamma - 1) / (5 * gamma - 3) * m.pow(x, -2) - 2 * gamma / (5 * gamma - 3) * m.pow(x,
                                                                                                           gamma - 1))
    FN = inversefunc(FPsi, domain=[0.001, 1])

    Ni = [0 for k in range(0, 1000)]
    Ne = [0 for k in range(0, 1000)]
    Psi = [k * dPsi+Psi0 for k in range(0, 1000)]
    V = [0 for k in range(0, 1000)]

    for i in range(0, 1000):
        Ni[i] = FN(Psi[i])
        #print(Ni[i])
        Ne[i] = m.exp(Psi[i])
        V[i] = Psi[i]*Te

    plt.plot(V, Ne, 'b')
    plt.plot(V, Ni, 'r')
    plt.ylabel('N')
    plt.xlabel('V')
    plt.show()

    return 0


if __name__ == "__main__":
    main()