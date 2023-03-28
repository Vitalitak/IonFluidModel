import math as m
import matplotlib.pyplot as plt
import numpy as np
from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc


def main():
    # initialisation of parameters
    boxsize = 5E-5  # m
    dt = 0.1  # ns
    Nx = 500000
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.3  # eV
    Ti = 0.06  # eV
    n0 = 1E16  # m-3
    Vdc = -18
    C = 1.4E-16
    C /= 1.6E-19

    # stitching parameters
    a = 3.5E-5  # m
    P = 0.995  # P = ni(a)/n0 boundary N(x)
    b = 8

    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    V0 = -Ti
    # V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    Nt = int(tEnd / dt)
    dx = boxsize / Nx

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    Vrf = 0

    Psil = -7
    NPsi = 1000
    dPsi = Psil/NPsi

    FPsi = lambda x: 4*Ti/Te*(1-3/8*m.pow(x, -2)-5/8*m.pow(x, 2/3))

    Number = FPsi(0.01)
    #print(Number)

    FN = inversefunc(FPsi, domain=[0.001, 1])

    xPsi = [k * dPsi for k in range(0, NPsi)]

    xNi = [0 for k in range(0, NPsi)]

    for i in range(0, NPsi):
        xNi[i] = FN(xPsi[i])

    plt.plot(xPsi, xNi)
    plt.ylabel('Ni')
    plt.show()

    #print(FN(Number))
    """
    f = symbols('f', cls=Function)
    #f, g = symbols("f g", cls=Function)
    x = symbols("x")
    #init_printing(use_unicode=True)
    diffeq = Eq(f(x).diff(x, x)-e*e*n0/eps0/kTe*m.exp(e*V0/kTe)*exp(f(x)), 0)
    dsolve(diffeq, f(x), hint='linear_coefficients', ics={f(0): 0, f(x).diff(x).subs(x, 0): 0})
    """
    return 0


if __name__ == "__main__":
    main()