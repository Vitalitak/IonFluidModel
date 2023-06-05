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
    dx = 1E-7
    boxsize = 4E-4  # m
    Nx = int(boxsize / dx)

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    NdV2 = [0 for k in range(0, Nx)]
    dV2 = [0 for k in range(0, Nx)]

    i = 0

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

    for i in range(1, Nx-2):
        NdV2[i] = - e / eps0 * (ni[i] - ne[i])
        dV2[i] = (V[i-1] - 2 * V[i] + V[i+1]) / dx / dx

    plt.plot(x, NdV2, 'r')
    plt.plot(x, dV2, 'b')

    plt.ylabel('d2V/dx2')
    plt.show()

    return 0

if __name__ == "__main__":
        main()