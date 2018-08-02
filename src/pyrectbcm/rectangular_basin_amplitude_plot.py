import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def amplitude_plot(Input):
    Basin = Input.Basin
    Inlets = Input.Inlets
    Ocean = Input.Ocean
    Pars = Input.Pars
    # Inlets.uj = Inlets.uj/Inlets.uj
    # Basin.ub = 1
    # rb = 8/(3*np.pi)*Basin.cd*Basin.ub
    # Basin.mub = 1 - 1j*rb/(Ocean.tidefreq*Basin.depth)

    eta = 0
    g = 9.81

    X = np.linspace(0, Basin.length, num = 100)
    Y = np.linspace(0, Basin.width, num = 100)
    X, Y = np.meshgrid(X, Y)
    inlets_zip = np.squeeze(Inlets.widths > 0)

    l2 = np.full((Pars.mtrunc+1, Pars.ntrunc+1, Basin.numinlets), Basin.length*Basin.width)
    l2[1:, :, :] = l2[1:, :, :]/2
    l2[:, 1:, :] = l2[:, 1:, :]/2
    signs = np.array([1, -1])
    signs = np.tile(signs[:, np.newaxis, np.newaxis], (ceil((Pars.mtrunc+1)/2), Pars.ntrunc+1, Basin.numinlets))
    phij = np.copy(l2)*0
    phij[:, 0, :] = 1
    phij[:, 1:, inlets_zip] = (
        Basin.width/(Pars.nrange[:, 1:, np.newaxis]*np.pi*Inlets.widths[np.newaxis, :, inlets_zip])
        * ( np.sin(Pars.nrange[:, 1:, np.newaxis]*np.pi/Basin.width*(
            Inlets.locations[np.newaxis, :, inlets_zip]
            + Inlets.widths[np.newaxis, :, inlets_zip]/2))
        -np.sin(Pars.nrange[:, 1:, np.newaxis]*np.pi/Basin.width*(
            Inlets.locations[np.newaxis, :, inlets_zip]
            - Inlets.widths[np.newaxis, :, inlets_zip]/2)) ))
    phij = phij * signs[0:Pars.mtrunc+1, 0:Pars.ntrunc+1, 0:(Basin.numinlets)]

    for j in range(0, Basin.numinlets):
        gsum = 0
        for m in Pars.mrange:
            for n in np.squeeze(Pars.nrange):
                gsum = gsum + ((phij[m, n, j]*np.cos(m*np.pi*X/Basin.length)*np.cos(n*np.pi*Y/Basin.width))/
                                ((Pars.kmn2[m, n] - Basin.mub*Basin.kb**2)*l2[m, n, j]))
        eta = eta + Inlets.widths[:, j]*Inlets.depths[:, j]*Inlets.uj[:, j]*gsum

    eta = eta*Ocean.tidefreq/(Basin.depth*g*1j)*Basin.mub

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, np.abs(eta), cmap=cm.viridis)
    plt.show()
