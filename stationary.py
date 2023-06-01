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

dPsi/dx=F(x, Psi)

"""


def RKPoisN(dx, Psi, Nsh, Nx, n0, Te, Psi0, Psil, FN):
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

    """
    Psi[0] = Psi0
    A = 2 * e * e * n0 / eps0 / kTe * m.exp(Psi0)
    B = -2 * e * e * n0 / eps0 / kTe
    C = -2 * e * e * n0 / eps0 / kTe * m.exp(Psi0)



    # print(A)
    # print(B)
    # print(C)
    # print(D)
    i=0
    #for i in range(Nsh+2, Nx-1):
    while (Psi[i] > Psil) and (i<Nx-1):
        print(i)
        f1 = -m.pow(-(A * m.exp(Psi[i]) + B * quad(FN, Psi0, Psi[i])[0]+C), 0.5)
        #print(f1)
        f2 = -m.pow(-(A * m.exp(Psi[i] + dx / 2 * f1) + B * quad(FN, Psi0, Psi[i]+ dx / 2 * f1)[0]+C), 0.5)
        f3 = -m.pow(-(A * m.exp(Psi[i] + dx / 2 * f2) + B * quad(FN, Psi0, Psi[i]+ dx / 2 * f2)[0]+C), 0.5)
        f4 = -m.pow(-(A * m.exp(Psi[i] + dx * f3) + B * quad(FN, Psi0, Psi[i]+ dx * f3)[0]+C), 0.5)
        #print(B * quad(FN, Psi0, Psi[i]+ dx * f3)[0])
        Psi[i + 1] = Psi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        i=i+1
    """
    dPsi = 0.1
    Psi[0] = Psi0-dPsi
    A = 2 * e * e * n0 / eps0 / kTe
    B = 2 * e * e * n0 / eps0 / kTe
    C = -2 * e * e * n0 / eps0 / kTe * m.exp(Psi0)

    # print(A)
    # print(B)
    # print(C)
    # print(D)
    i = 0
    # for i in range(Nsh+2, Nx-1):
    while (Psi[i] > Psil) and (i < Nx - 1):
        print(i)
        f1 = -m.pow((A * m.exp(Psi[i]) + B * quad(FN, Psi[i], Psi0)[0] + C), 0.5)
        f2 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f1) + B * quad(FN, Psi[i] + dx / 2 * f1, Psi0)[0] + C), 0.5)
        f3 = -m.pow((A * m.exp(Psi[i] + dx / 2 * f2) + B * quad(FN, Psi[i] + dx / 2 * f2, Psi0)[0] + C), 0.5)
        f4 = -m.pow((A * m.exp(Psi[i] + dx * f3) + B * quad(FN, Psi[i] + dx * f3, Psi0)[0] + C), 0.5)
        Psi[i + 1] = Psi[i] + dx / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        i = i + 1

    Nel = i + 1

    return Psi, Nel


def main():
    # initialisation of parameters
    boxsize = 4E-4  # m
    #a = 1E-6
    dt = 0.1  # ns
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
    Te = 2.68  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    #C = 1.4E-16
    #C /= 1.6E-19
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

    Psil = e*Vdc/kTe
    Psi0 = -1

    FPsi = lambda x: (5*gamma-3)*Ti/Te/2/(gamma-1)*(1-3*(gamma-1)/(5*gamma-3)*m.pow(x, -2)-2*gamma/(5*gamma-3)*m.pow(x, gamma-1)) + Psi0

    FN = inversefunc(FPsi, domain=[0.001, 1])

    """
    res, err = quad(FN, 0, -1)
    print(print("The numerical result is {:f} (+-{:g})"
    .format(res, err)))
    print(quad(FN, 0, -1)[0])
    """

    Psi, Nel = RKPoisN(dx, Psi, Nsh, Nx, n0, Te, Psi0, Psil, FN)

    #for i in range(0, NPsi):
        #xNi[i] = FN(xPsi[i])
        #intNx[i] = quad(FN, 0, xPsi[i])[0]


    for i in range(Nsh, Nel):
        Ni[i] = FN(Psi[i])
        #print(Ni[i])

    """
    for i in range(Nsh, Nx-1):
        dPsidx[i] = (Psi[i+1]-Psi[i])/dx
    """
    for i in range(0, Nel):
        V[i] = Psi[i]*kTe/e
        ni[i] = Ni[i]*n0
        ne[i] = n0*m.exp(e*V[i]/kTe)
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]


    plt.plot(x, Psi)
    plt.ylabel('Psi')
    plt.show()

    plt.plot(x, Ni)
    plt.ylabel('Ni')
    plt.show()

    plt.plot(x, dPsidx)
    plt.ylabel('dPsi/dx')
    plt.show()

    plt.plot(x, V)
    plt.ylabel('V')
    plt.show()

    plt.plot(x, ne, 'b')
    plt.plot(x, ni, 'r')
    plt.ylabel('N')
    plt.show()

    plt.plot(x, ui)
    plt.ylabel('u')
    plt.show()

    f = open("V.txt", "w")
    for d in V:
        f.write(f"{d}\n")
    f.close()

    f = open("ni.txt", "w")
    for d in ni:
        f.write(f"{d}\n")
    f.close()

    f = open("ne.txt", "w")
    for d in ne:
        f.write(f"{d}\n")
    f.close()

    f = open("ui.txt", "w")
    for d in ui:
        f.write(f"{d}\n")
    f.close()

    f = open("Nel.txt", "w")
    f.write(f"{Nel}\n")
    f.close()

    return 0


if __name__ == "__main__":
    main()