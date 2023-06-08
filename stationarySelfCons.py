import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#from sympy import Function, dsolve, Eq, Derivative, exp, symbols
#from sympy.solvers.ode.systems import dsolve_system
from pynverse import inversefunc

"""
Psi = e*V/kTe
Delta = -e/kTe*dV/dx
N = ni/n0

Self-consistent system of Poisson equation and ion momentum conservation
Electrons have Maxwell-Boltzmann distribution
Ions have collisions

four order Runge-Kutta method for solution system
dPsi/dx=f(x, Psi, Delta, N)
dDelta/dx=g(x, Psi, Delta, N)
dN/dx=h(x, Psi, Delta, N)

for dn/dt = 0 and du/dt = 0


"""

def RungeKuttasystem(Nx, dx, n0, Te, Ti, Psil, gamma, nu):
    e = 1.6E-19
    eps0 = 8.85E-12
    mi = 6.68E-26  # kg
    kTe = Te * 1.6E-19  # J
    kTi = Ti * 1.6E-19  # J

    """
    Psi(0) = 0
    Delta(0) = 0
    N(0) = 1
    
    dPsi/dx=f(x, Psi, Delta, N)
    dDelta/dx=g(x, Psi, Delta, N)
    dN/dx=h(x, Psi, Delta, N)
    
    f(x, Psi, Delta, N) = -Delta
    g(x, Psi, Delta, N) = e*e*n0/eps0/kTe*(N-exp(Psi))
    h(x, Psi, Delta, N) = (kTe*Delta*N-sqrt(mi*kTi)*nu)/(kTi*(gamma*N^(gamma-1)-1))*N^2
    
    """
    Psi = np.zeros(Nx)
    Delta = np.zeros(Nx)
    N = np.zeros(Nx)
    pcheck1 = np.zeros(Nx)
    pcheck2 = np.zeros(Nx)

    #Psi[0] = -1e-3
    #Delta[0] = 1e-2
    #N[0] = m.exp(Psi[0])
    Psi[0] = 0
    Delta[0] = 1000
    N[0] = 1

    i = 0

    while (Psi[i] > Psil) and (i<Nx-1):
        #print(i)
        k1 = dx * (-Delta[i])
        l1 = dx * e * e * n0 / eps0 / kTe * (N[i]-m.exp(Psi[i]))
        p1 = dx * (kTe*Delta[i]*N[i]-m.sqrt(mi*kTi)*nu) / kTi / (gamma * m.pow(N[i], gamma+1) - 1) * N[i] * N[i]
        k2 = dx * (-Delta[i]-l1/2)
        l2 = dx * e * e * n0 / eps0 / kTe * (N[i] + p1/2 -m.exp(Psi[i]+k1/2))
        p2 = dx * (kTe*(Delta[i]+l1/2)*(N[i]+p1/2)-m.sqrt(mi*kTi)*nu) / kTi / (gamma * m.pow(N[i]+p1/2, gamma+1) - 1) * (N[i]+p1/2) * (N[i]+p1/2)
        k3 = dx * (-Delta[i]-l2/2)
        l3 = dx * e * e * n0 / eps0 / kTe * (N[i] + p2 / 2 - m.exp(Psi[i] + k2 / 2))
        p3 = dx * (kTe * (Delta[i] + l2 / 2) * (N[i] + p2 / 2) - m.sqrt(mi * kTi) * nu) / kTi / (
                    gamma * m.pow(N[i] + p2 / 2, gamma + 1) - 1) * (N[i] + p2 / 2) * (N[i] + p2 / 2)
        k4 = dx * (-Delta[i]-l3)
        l4 = dx * e * e * n0 / eps0 / kTe * (N[i] + p3 - m.exp(Psi[i] + k3))
        p4 = dx * (kTe * (Delta[i] + l3) * (N[i] + p3) - m.sqrt(mi * kTi) * nu) / kTi / (
                gamma * m.pow(N[i] + p3, gamma + 1) - 1) * (N[i] + p3) * (N[i] + p3)
        pcheck1[i] = kTe*Delta[i]*N[i]-m.sqrt(mi*kTi)*nu
        pcheck2[i] = gamma * m.pow(N[i], gamma+1) - 1
        print(k1)
        #print(B * quad(FN, Psi0, Psi[i]+ dx * f3)[0])
        Psi[i + 1] = Psi[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Delta[i + 1] = Delta[i] + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        N[i + 1] = N[i] + 1 / 6 * (p1 + 2 * p2 + 2 * p3 + p4)

        i=i+1

    plt.plot(pcheck1, 'b')
    plt.ylabel('p chisl')
    plt.show()

    plt.plot(pcheck2, 'r')
    plt.ylabel('p znam')
    plt.show()

    Nel = i + 1

    return Psi, Delta, N, Nel

def main():
    # initialisation of parameters
    boxsize = 1E-4  # m
    dx = 1E-7
    Nx = int(boxsize/dx)
    Nsh = 0

    me = 9.11E-31  # kg
    mi = 6.68E-26  # kg
    e = 1.6E-19
    eps0 = 8.85E-12

    # plasma parameters
    Te = 2.68  # eV
    Ti = 0.05  # eV
    n0 = 3E17  # m-3
    Vdc = -17
    gamma = 3
    nu = 100000000



    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J

    x = np.arange(Nx)*dx
    V = np.zeros(Nx)
    ni = np.zeros(Nx)
    ne = np.zeros(Nx)
    ui = np.zeros(Nx)

    Psi = np.zeros(Nx)
    N = np.zeros(Nx)
    Delta = np.zeros(Nx)

    Psil = e*Vdc/kTe
    Psi0 = -1e-7

    """
    FPsi = lambda x: (5*gamma-3)*Ti/Te/2/(gamma-1)*(1-3*(gamma-1)/(5*gamma-3)*m.pow(x, -2)-2*gamma/(5*gamma-3)*m.pow(x, gamma-1))

    FN = inversefunc(FPsi, domain=[0.000000001, 1])

    
    res, err = quad(FN, 0, -1)
    print(print("The numerical result is {:f} (+-{:g})"
    .format(res, err)))
    print(quad(FN, 0, -1)[0])
    """

    Psi, Delta, N, Nel = RungeKuttasystem(Nx, dx, n0, Te, Ti, Psil, gamma, nu)


    for i in range(0, Nel):
        V[i] = Psi[i]*kTe/e
        ni[i] = N[i]*n0
        ne[i] = n0*m.exp(e*V[i]/kTe)
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
    """
    ne[0] = n0
    for i in range(1, Nel-1):
        ne[i] = ni[i] + eps0 / e * (V[i-1] + 2 * V[i] - V[i+1]) / dx / dx
    """
    """
    ni[0] = n0
    ui[0] = n0 * m.sqrt(kTi / mi) / ni[0]
    for i in range(1, Nel - 1):
        ni[i] = ne[i] - eps0 / e * (V[i - 1] + 2 * V[i] - V[i + 1]) / dx / dx
        ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
    """
    #print(Psi)

    plt.plot(x, Psi)
    plt.ylabel('Psi')
    plt.show()

    plt.plot(x, N)
    plt.ylabel('Ni')
    plt.show()

    plt.plot(x, Delta)
    plt.ylabel('-dPsi/dx')
    plt.show()
    """
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
    """
    return 0


if __name__ == "__main__":
    main()