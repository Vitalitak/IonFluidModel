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

def RungeKuttasystem(Nx, dx, n0, Te, Ti, Psil, gammai, gammae, nu, nue, nuiz):
    e = 1.6E-19
    eps0 = 8.85E-12
    me = 9.11E-31  # kg
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
    Ni = np.zeros(Nx)
    Ui = np.zeros(Nx)
    #Ne = np.zeros(Nx)
    pcheck1 = np.zeros(Nx)
    pcheck2 = np.zeros(Nx)
    lcheck1 = np.zeros(Nx)
    lcheck2 = np.zeros(Nx)

    #Psi[0] = -0.5
    #Delta[0] = 50000
    #Ni[0] = m.exp(Psi[0])
    #Ne[0] = m.exp(Psi[0])
    Psi[0] = -0.1  # adjusted value
    Delta[0] = 20000 # adjusted value
    Ni[0] = m.exp(Psi[0])
    #Ui[0] = 1.001
    Ui[0] = 3 # # adjusted value
    #Ne[0] = m.exp(Psi[0])
    print(Ni[0])
    uth=m.sqrt(gammai*kTi/mi)
    # nui=nue=0
    i = 0

    while (Psi[i] > Psil) and (i<Nx-1):
        #print(i)
        k1 = dx * (-Delta[i])
        l1 = dx * e * e * n0 / eps0 / kTe * (Ni[i]-m.exp(Psi[i]))
        #l1 = dx * e * e * n0 / eps0 / kTe * (Ni[i] - Ne[i])
        #p1 = dx * (kTe*Delta[i]*Ni[i]-m.sqrt(mi*kTi)*nu) / kTi / (gammai * m.pow(Ni[i], gammai+1) - 1) * Ni[i] * Ni[i]
        #m1 = dx * (kTe*Delta[i]*Ne[i]-m.sqrt(me*kTe)*nue) / kTe / (gammae * m.pow(Ne[i], gammae+1) - 1) * Ne[i] * Ne[i]
        p1 = dx * (m.exp(Psi[i])*nuiz/Ui[i]/uth*(1-1/(Ui[i]*Ui[i]-1))-kTe/kTi/gammai*Ni[i]*Delta[i]/(Ui[i]*Ui[i]-1))
        m1 = dx * (kTe/kTi/gammai*Ui[i]*Delta[i]/(Ui[i]*Ui[i]-1)-m.exp(Psi[i])*nuiz/Ni[i]/uth/(Ui[i]*Ui[i]-1))

        k2 = dx * (-Delta[i]-l1/2)
        l2 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p1/2 -m.exp(Psi[i]+k1/2))
        #l2 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p1/2 - Ne[i] - m1/2)
        #p2 = dx * (kTe*(Delta[i]+l1/2)*(Ni[i]+p1/2)-m.sqrt(mi*kTi)*nu) / kTi / (gammai * m.pow(Ni[i]+p1/2, gammai+1) - 1) * (Ni[i]+p1/2) * (Ni[i]+p1/2)
        #m2 = dx * (kTe*(Delta[i]+l1/2)*(Ne[i]+m1/2)-m.sqrt(me*kTe)*nue) / kTe / (gammae * m.pow(Ne[i]+m1/2, gammae+1) - 1) * (Ne[i]+m1/2) * (Ne[i]+m1/2)
        p2 = dx * (m.exp(Psi[i]+k1/2) * nuiz / (Ui[i]+m1/2) / uth * (1 - 1 / ((Ui[i]+m1/2) * (Ui[i]+m1/2) - 1)) - kTe / kTi / gammai * (Ni[i]+p1/2) *
                   (Delta[i]+l1/2) / ((Ui[i]+m1/2) * (Ui[i]+m1/2) - 1))
        m2 = dx * (kTe / kTi / gammai * (Ui[i]+m1/2) * (Delta[i]+l1/2) / ((Ui[i]+m1/2) * (Ui[i]+m1/2) - 1) - m.exp(Psi[i]+k1/2) * nuiz / (Ni[i]+p1/2) / uth / (
                    (Ui[i]+m1/2) * (Ui[i]+m1/2) - 1))

        k3 = dx * (-Delta[i]-l2/2)
        l3 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p2 / 2 - m.exp(Psi[i] + k2 / 2))
        #l3 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p2 / 2 - Ne[i] - m2 / 2)
        #p3 = dx * (kTe * (Delta[i] + l2 / 2) * (Ni[i] + p2 / 2) - m.sqrt(mi * kTi) * nu) / kTi / (
                    #gammai * m.pow(Ni[i] + p2 / 2, gammai + 1) - 1) * (Ni[i] + p2 / 2) * (Ni[i] + p2 / 2)
        #m3 = dx * (kTe * (Delta[i] + l2 / 2) * (Ne[i] + m2 / 2)-m.sqrt(me*kTe)*nue) / kTe / (
                    #gammae * m.pow(Ne[i] + m2 / 2, gammae + 1) - 1) * (Ne[i] + m2 / 2) * (Ne[i] + m2 / 2)
        p3 = dx * (m.exp(Psi[i] + k2 / 2) * nuiz / (Ui[i] + m2 / 2) / uth * (
                    1 - 1 / ((Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1)) - kTe / kTi / gammai * (Ni[i] + p2 / 2) *
                   (Delta[i] + l2 / 2) / ((Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1))
        m3 = dx * (kTe / kTi / gammai * (Ui[i] + m2 / 2) * (Delta[i] + l2 / 2) / (
                    (Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1) - m.exp(Psi[i] + k2 / 2) * nuiz / (
                               Ni[i] + p2 / 2) / uth / (
                           (Ui[i] + m2 / 2) * (Ui[i] + m2 / 2) - 1))

        k4 = dx * (-Delta[i]-l3)
        l4 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p3 - m.exp(Psi[i] + k3))
        #l4 = dx * e * e * n0 / eps0 / kTe * (Ni[i] + p3 - Ne[i] - m3)
        #p4 = dx * (kTe * (Delta[i] + l3) * (Ni[i] + p3) - m.sqrt(mi * kTi) * nu) / kTi / (
                #gammai * m.pow(Ni[i] + p3, gammai + 1) - 1) * (Ni[i] + p3) * (Ni[i] + p3)
        #m4 = dx * (kTe * (Delta[i] + l3) * (Ne[i] + m3)-m.sqrt(me*kTe)*nue) / kTe / (
                #gammae * m.pow(Ne[i] + m3, gammae + 1) - 1) * (Ne[i] + m3) * (Ne[i] + m3)
        p4 = dx * (m.exp(Psi[i] + k3) * nuiz / (Ui[i] + m3) / uth * (
                1 - 1 / ((Ui[i] + m3) * (Ui[i] + m3) - 1)) - kTe / kTi / gammai * (Ni[i] + p3) *
                   (Delta[i] + l3) / ((Ui[i] + m3) * (Ui[i] + m3) - 1))
        m4 = dx * (kTe / kTi / gammai * (Ui[i] + m3) * (Delta[i] + l3) / (
                (Ui[i] + m3) * (Ui[i] + m3) - 1) - m.exp(Psi[i] + k3) * nuiz / (
                           Ni[i] + p3) / uth / (
                           (Ui[i] + m3) * (Ui[i] + m3) - 1))

        #pcheck1[i] = kTe*Delta[i]*Ni[i]-m.sqrt(mi*kTi)*nu
        #pcheck2[i] = gammai * m.pow(Ni[i], gammai+1) - 1
        #lcheck1[i] = Ni[i]
        #lcheck2[i] = m.exp(Psi[i])
        #print(p1)
        #print(B * quad(FN, Psi0, Psi[i]+ dx * f3)[0])
        Psi[i + 1] = Psi[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Delta[i + 1] = Delta[i] + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        Ni[i + 1] = Ni[i] + 1 / 6 * (p1 + 2 * p2 + 2 * p3 + p4)
        Ui[i + 1] = Ui[i] + 1 / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        #Ne[i+1] = Ne[i] + 1 / 6 * (m1 + 2 * m2 + 2 * m3 + m4)

        i=i+1

    plt.plot(pcheck1, 'b')
    plt.ylabel('p chisl')
    #plt.ylabel('Ni')
    plt.show()

    plt.plot(pcheck2, 'r')
    plt.ylabel('p znam')
    #plt.ylabel('expPsi')
    plt.show()

    Nel = i + 1

    return Psi, Delta, Ni, Ui, Nel

def main():
    # initialisation of parameters
    boxsize = 2E-4  # m
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
    gammai = 3
    gammae = 3
    #nu = 4e8
    nu = 0
    #nue = 4e12
    nue = 0
    nuiz = 5e7  # adjusted value
    #nuiz = 0



    kTi = Ti * 1.6E-19  # J
    kTe = Te * 1.6E-19  # J

    x = np.arange(Nx)*dx
    V = np.zeros(Nx)
    ni = np.zeros(Nx)
    ne = np.zeros(Nx)
    ui = np.zeros(Nx)
    ue = np.zeros(Nx)

    Psi = np.zeros(Nx)
    Ni = np.zeros(Nx)
    Ui = np.zeros(Nx)
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

    Psi, Delta, Ni, Ui, Nel = RungeKuttasystem(Nx, dx, n0, Te, Ti, Psil, gammai, gammae, nu, nue, nuiz)


    for i in range(0, Nel):
        V[i] = Psi[i]*kTe/e
        ni[i] = Ni[i]*n0
        #ne[i] = Ne[i]*n0
        ne[i] = n0*m.exp(e*V[i]/kTe)
        ui[i] = Ui[i]*m.sqrt(gammai*kTi/mi)
        #ui[i] = n0 * m.sqrt(kTi / mi) / ni[i]
        #ue[i] = n0 * m.sqrt(kTe / me) / ne[i]
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
    """
    plt.plot(x, Ne, 'b')
    plt.plot(x, Ni, 'r')
    plt.ylabel('Ni')
    plt.show()
    """
    plt.plot(x, Delta)
    plt.ylabel('-dPsi/dx')
    plt.show()

    plt.plot(x, V)
    plt.ylabel('V')
    plt.show()

    plt.plot(x, ne, 'b')
    plt.plot(x, ni, 'r')
    plt.ylabel('N')
    plt.show()

    #plt.plot(x, ue, 'b')
    plt.plot(x, ui, 'r')
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