import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

"""
four order Runge-Kutta method for solution equation
dy/dx=f(x, y)

Poisson equation with Maxwell-Boltzmann electrons and ion concentration from fluid model
for dn/dt = 0 and du/dt = 0

N=3
gammai = 5/3

dPsi/dx=F(x, Ksi)

"""


def RKPois1(dx, Psi, Nsh, n0, Ti, Te, V0):
    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    Psi(0)=0
    dPsi/dx(0) = 0
    dPsi/dx<0

    boundary N(x) = 0.995

    dPsi/dx=F(x, Psi)
    F = -(A*exp(Psi)+B*Psi+C*(1-19*Te/2/Ti*Psi)^3/2+D)^1/2

    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=-32/19*e*e*n0/eps0/kTe
    C=8/361*Ti/Te*e*e*n0/eps0/kTe
    D=-2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)-8/361*Ti/Te*e*e*n0/eps0/kTe
    Ksi(0)=0

    Four order Runge-Kutta method
    f1=F(x[n], Psi[n])
    f2=F(x[n]+dx/2, Psi[n]+dx/2*f1)
    f3=F(x[n]+dx/2, Psi[n]+dx/2*f2)
    f4=F(x[n]+dx, Psi[n]+dx*f3)

    Psi[n+1]=Psi[n]+dx/6*(f1+2*f2+2*f3+f4)

    """

    # dx = x[Npl - 1]-x[Npl - 2]
    # Nx = len[Ksi]

    Psi[2] = dx * dx * e * e * n0 / eps0 / kTe * (m.exp(e * V0 / kTe) - 1)
    A = 2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe)
    B = -32 / 19 * e * e * n0 / eps0 / kTe
    C = 8 / 361 * Ti / Te * e * e * n0 / eps0 / kTe
    D = -2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe) - 8 / 361 * Ti / Te * e * e * n0 / eps0 / kTe

    # print(A)
    # print(B)
    # print(C)
    # print(D)

    for i in range(2, Nsh):
        f1 = -m.pow((A * m.exp(Psi[i]) + B * Psi[i] + C * m.pow((1 - 19 * Te / 2 / Ti * Psi[i]), 1.5) + D), 0.5)
        f2 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f1) + B * (Psi[i] + dx / 2 * f1) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Psi[i] + dx / 2 * f1)), 1.5) + D), 0.5)
        f3 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f2) + B * (Psi[i] + dx / 2 * f2) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Psi[i] + dx / 2 * f2)), 1.5) + D), 0.5)
        f4 = -m.pow((A * m.exp(Psi[i] + dx * f3) + B * (Psi[i] + dx * f3) + C * m.pow(
            (1 - 19 * Te / 2 / Ti * (Psi[i] + dx * f3)), 1.5) + D), 0.5)
        Psi[i + 1] = Psi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

    return Psi

def RKPoisN(dx, Psi, Nsh, Nx, n0, Ti, Te, V0, FN):
    e = 1.6E-19
    eps0 = 8.85E-12
    kTe = Te * 1.6E-19  # J

    """
    Psi(0)=0
    dPsi/dx(0) = 0
    dPsi/dx<0



    dPsi/dx=F(x, Psi)
    F = -(A*exp(Psi)+B*int(N(dzetta), dzetta=[0, Psi(x)]))^1/2

    A=2*e*e*n0/eps0/kTe*m.exp(e*V0/kTe)
    B=-2*e*e*n0/eps0/kTe
    
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

    #Psi[2] = dx * dx * e * e * n0 / eps0 / kTe * (m.exp(e * V0 / kTe) - 1)
    A = 2 * e * e * n0 / eps0 / kTe * m.exp(e * V0 / kTe)
    B = -2 * e * e * n0 / eps0 / kTe

    # print(A)
    # print(B)
    # print(C)
    # print(D)

    for i in range(Nsh, Nx-1):
        print(i)
        f1 = -m.pow((A * m.exp(Psi[i]) + B * quad(FN, 0, Psi[i])[0]), 0.5)
        f2 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f1) + B * quad(FN, 0, Psi[i]+ dx / 2 * f1)[0]), 0.5)
        f3 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f2) + B * quad(FN, 0, Psi[i]+ dx / 2 * f2)[0]), 0.5)
        f4 = -m.pow((A * m.exp(Psi[i] + dx * f3) + B * quad(FN, 0, Psi[i]+ dx * f3)[0]), 0.5)
        Psi[i + 1] = Psi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

    return Psi


def main():
    # initialisation of parameters
    boxsize = 15E-5  # m
    a = 0.1E-5
    dt = 0.1  # ns
    dx = 1E-7
    Nx = int(boxsize/dx)
    Nsh = int(a/dx)
    #Nx = 500000
    tEnd = 50  # ns

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.3  # eV
    Ti = 0.06  # eV
    n0 = 1E16  # m-3
    Vdc = -15
    C = 1.4E-16
    C /= 1.6E-19

    # stitching parameters
    P = 0.995  # P = ni(a)/n0 boundary N(x)
    b = 8

    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J
    V0 = -Ti
    # V0 = kTe / e * (1 - P) / (m.cosh(m.sqrt(e * e * n0 / 2 / eps0 / kTi) * a) - 1)

    Nt = int(tEnd / dt)

    x = [k * dx for k in range(0, Nx)]
    V = [0 for k in range(0, Nx)]
    ni = [0 for k in range(0, Nx)]
    ne = [0 for k in range(0, Nx)]
    ui = [0 for k in range(0, Nx)]

    Psi = [0 for k in range(0, Nx)]
    Ni = [0 for k in range(0, Nx)]
    Vrf = 0

    Psil = e*(Vdc-V0)/kTe
    NPsi = 1000
    dPsi = Psil/NPsi

    FPsi = lambda x: 4*Ti/Te*(1-3/8*m.pow(x, -2)-5/8*m.pow(x, 2/3))

    Number = FPsi(0.01)
    #print(Number)

    FN = inversefunc(FPsi, domain=[0.001, 1])

    xPsi = [k * dPsi for k in range(0, NPsi)]

    xNi = [0 for k in range(0, NPsi)]

    res, err = quad(FN, 0, -1)
    print(print("The numerical result is {:f} (+-{:g})"
    .format(res, err)))
    print(quad(FN, 0, -1)[0])

    Psi = RKPois1(dx, Psi, Nsh, n0, Ti, Te, V0)

    Psi = RKPoisN(dx, Psi, Nsh, Nx, n0, Ti, Te, V0, FN)

    for i in range(0, NPsi):
        xNi[i] = FN(xPsi[i])

    for i in range(0, Nsh):
        Ni[i] = 1 + 3 / 19 * (1 - m.sqrt(1 - 19 * Te / 2 / Ti * Psi[i]))

    for i in range(Nsh, Nx):
        Ni[i] = FN(Psi[i])

    plt.plot(xPsi, xNi)
    plt.ylabel('Ni')
    plt.show()

    plt.plot(x, Psi)
    plt.ylabel('Psi')
    plt.show()

    plt.plot(x, Ni)
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