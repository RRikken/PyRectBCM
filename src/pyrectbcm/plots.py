import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
g = 9.81

def amplitude_plot(Input, silent = None):
    """ Produces a plot of the tidal amplitude in the basin

    Args:
        Input (Input class): class containing the model output

    Returns:
        True on succes
    """

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
    if silent == None:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, np.abs(eta), cmap=cm.viridis)
        ax.view_init(elev = 90, azim = -90)
        ax.set_aspect('equal')
        plt.show()
    return True


def evolution_plot(Input, silent = None):
    """ Produces a plot of the evolution of the tidal inlets

    Args:
        Input (Input class): class containing the model output

    Returns:
        True on succes
    """
    Inlets = Input.Inlets
    Basin = Input.Basin

    iw = Inlets.wit
    iw2 = np.zeros((iw.shape[0], Basin.numinlets*4 + 2))
    iw2[:, 2:-2:4] = iw
    iw2[:, 3:-2:4] = iw

    x = np.zeros(iw2.shape)
    x[:, -1] = Basin.width
    x[:, 1:-1:4] = Inlets.locations - 0.5*iw
    x[:, 2:-1:4] = Inlets.locations - 0.5*iw
    x[:, 3:-1:4] = Inlets.locations + 0.5*iw
    x[:, 4:-1:4] = Inlets.locations + 0.5*iw
    y = np.arange(0, iw.shape[0])
    y = np.repeat(y[:, np.newaxis], iw2.shape[1], axis = 1)
    if silent == None:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x/1e3, y/1e3, iw2, cmap=cm.gray_r)
        ax.view_init(elev = 90, azim = 0)
        plt.show()
    return True
