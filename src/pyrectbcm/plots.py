import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec, patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D, art3d

g = 9.81


def amplitude_plot(Input, ax=None, silent=None):
    """ Produces a plot of the tidal amplitude in the basin

    Args:
        Input (ModelData class): class containing the model output
        ax (axis): figure axis
        silent (int): flag for plotting or not plotting

    Returns:
        returnObject (dict):
            True: on succes
            ax: axes object
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

    X = np.linspace(0, Basin.length, num=100)
    Y = np.linspace(0, Basin.width, num=100)
    X, Y = np.meshgrid(X, Y)
    inlets_zip = np.squeeze(Inlets.widths > 0)

    l2 = np.full(
        (Pars.mtrunc + 1, Pars.ntrunc + 1, Basin.numinlets), Basin.length * Basin.width
    )
    l2[1:, :, :] = l2[1:, :, :] / 2
    l2[:, 1:, :] = l2[:, 1:, :] / 2
    signs = np.array([1, -1])
    signs = np.tile(
        signs[:, np.newaxis, np.newaxis],
        (ceil((Pars.mtrunc + 1) / 2), Pars.ntrunc + 1, Basin.numinlets),
    )
    phij = np.copy(l2) * 0
    phij[:, 0, :] = 1
    phij[:, 1:, inlets_zip] = (
        Basin.width
        / (
            Pars.nrange[:, 1:, np.newaxis]
            * np.pi
            * Inlets.widths[np.newaxis, :, inlets_zip]
        )
        * (
            np.sin(
                Pars.nrange[:, 1:, np.newaxis]
                * np.pi
                / Basin.width
                * (
                    Inlets.locations[np.newaxis, :, inlets_zip]
                    + Inlets.widths[np.newaxis, :, inlets_zip] / 2
                )
            )
            - np.sin(
                Pars.nrange[:, 1:, np.newaxis]
                * np.pi
                / Basin.width
                * (
                    Inlets.locations[np.newaxis, :, inlets_zip]
                    - Inlets.widths[np.newaxis, :, inlets_zip] / 2
                )
            )
        )
    )
    phij = phij * signs[0 : Pars.mtrunc + 1, 0 : Pars.ntrunc + 1, 0 : (Basin.numinlets)]

    for j in range(0, Basin.numinlets):
        gsum = 0
        for m in Pars.mrange:
            for n in np.squeeze(Pars.nrange):
                gsum = gsum + (
                    (
                        phij[m, n, j]
                        * np.cos(m * np.pi * X / Basin.length)
                        * np.cos(n * np.pi * Y / Basin.width)
                    )
                    / ((Pars.kmn2[m, n] - Basin.mub * Basin.kb ** 2) * l2[m, n, j])
                )
        eta = eta + Inlets.widths[:, j] * Inlets.depths[:, j] * Inlets.uj[:, j] * gsum

    eta = eta * Ocean.tidefreq / (Basin.depth * g * 1j) * Basin.mub
    if silent is None:
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        ax.plot_surface(X, Y, np.abs(eta), cmap=cm.viridis)
        ax.view_init(elev=90, azim=-90)
        ax.set_aspect("equal")
        ax.set_zticks([])
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot(
            (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
        )
        plt.show()
    return True, ax


def evolution_plot(Input, ax=None, silent=None):
    """ Produces a plot of the evolution of the tidal inlets

    Args:
        Input (ModelData class): class containing the model output
        ax (axis): figure axis
        silent (int): flag for plotting or not plotting

    Returns:
        returnObject (dict):
            True: on succes
            ax: axes object
    """
    Inlets = Input.Inlets
    Basin = Input.Basin
    Pars = Input.Pars
    iw = Inlets.wit
    iw2 = np.zeros((iw.shape[0], Basin.numinlets * 4 + 2))
    iw2[:, 2:-2:4] = iw
    iw2[:, 3:-2:4] = iw

    x = np.zeros(iw2.shape)
    x[:, -1] = Basin.width
    x[:, 1:-1:4] = Inlets.locations - 0.5 * iw
    x[:, 2:-1:4] = Inlets.locations - 0.5 * iw
    x[:, 3:-1:4] = Inlets.locations + 0.5 * iw
    x[:, 4:-1:4] = Inlets.locations + 0.5 * iw
    y = np.arange(0, iw.shape[0]) * Pars.dt
    y = np.repeat(y[:, np.newaxis], iw2.shape[1], axis=1)

    if silent is None:
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        ax.plot_surface(
            x / 1e3, y, iw2, cmap=cm.gray_r, rcount=iw2.shape[0], ccount=iw2.shape[1]
        )

        # ax.view_init(elev = 90, azim = 0)
        ax.set_xlim(
            [-0.3 * Basin.length / 1e3, (Basin.width + 0.3 * Basin.length) / 1e3]
        )
        ax.set_ylim([0, iw.shape[0] * Pars.dt])
        ax.set_zlim([0, 1e4])
        ax.xaxis.set_visible(False)
        ax.zaxis.set_visible(False)
        ax.set_xticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot(
            (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
        )
        ax.grid(False)
        ax.set_ylabel("Time (years)")
        plt.show()
    return True, ax


def geometry_plot(Input, t, ax=None):
    """ Produces a plot of the model geometry
    Args:
        Input (ModelData class): class containing the model output
        ax (axis): figure axis
        silent (int): flag for plotting or not plotting

    Returns:
        returnObject (dict):
            True: on succes
            ax: axes object
    """
    Basin = Input.Basin
    Inlets = Input.Inlets

    offset = 0.3 * Basin.length
    dims = (
        np.array([0, (Basin.length + offset) * 2, 0, (Basin.width + offset * 2)]) / 1e3
    )
    qs = 20
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    ax.set_xlim(dims[2], dims[3])
    ax.set_ylim(dims[0], dims[1])
    ax.set_zlim(0, 1e2)  # large enough to not see stacking effects of different layers
    cl = np.array([74, 124, 89]) / 256
    cs = np.array([104, 176, 171]) / 256
    cb = cs
    ci = cs
    cc = np.array([250, 243, 221]) / 256

    # Draw the land
    land = patches.Rectangle((0, 0), dims[3], dims[1], color=cl)
    ax.add_patch(land)
    art3d.pathpatch_2d_to_3d(land, z=0)

    # Draw the sea
    sea = patches.Rectangle(
        (0, dims[1] / 2 + Inlets.lengths[0] / 1e3),
        dims[3],
        dims[1] / 2 - Inlets.lengths[0] / 1e3,
        color=cs,
    )
    ax.add_patch(sea)
    art3d.pathpatch_2d_to_3d(sea, z=0.1)
    ax.plot(
        (0, dims[3]),
        (dims[1] / 2 + Inlets.lengths[0] / 1e3, dims[1] / 2 + Inlets.lengths[0] / 1e3),
        0.1,
        color=cc,
    )

    # Draw the basin
    basin = patches.Rectangle(
        (offset / 1e3, offset / 1e3),
        Basin.width / 1e3,
        Basin.length / 1e3,
        facecolor=cb,
        edgecolor=cc,
    )
    ax.add_patch(basin)
    art3d.pathpatch_2d_to_3d(basin, z=0.1)

    # Draw the inlets
    for inlet in range(Basin.numinlets):
        if Inlets.wit[t, inlet] > 0:
            xi = (Inlets.locations[:, inlet] - Inlets.wit[t, inlet] / 2 + offset) / 1e3
            inlet_ = patches.Rectangle(
                (xi, dims[1] / 2 - 0.1),
                Inlets.wit[t, inlet] / 1e3,
                Inlets.lengths[inlet] * 1e-3 + 0.1,
                facecolor=cs,
            )
            ax.add_patch(inlet_)
            art3d.pathpatch_2d_to_3d(inlet_, z=0.2)
            ax.plot(
                (xi, xi),
                (dims[1] / 2, dims[1] / 2 + Inlets.lengths[inlet] / 1e3),
                0.2,
                color=cc,
            )
            ax.plot(
                (
                    xi + Inlets.widths[:, inlet] / 1e3,
                    xi + Inlets.widths[:, inlet] / 1e3,
                ),
                (dims[1] / 2, dims[1] / 2 + Inlets.lengths[inlet] / 1e3),
                0.2,
                color=cc,
            )

    ax.axis("off")
    ax.plot(
        (dims[2], dims[2], dims[3], dims[3], dims[2]),
        (dims[0], dims[1], dims[1], dims[0], dims[0]),
        1,
        color="k",
    )
    ax.grid(False)
    ax.set_aspect(((dims[3] - dims[2]) / (dims[1] - dims[0])) ** 2)
    ax.apply_aspect()
    # print(ax.get_aspect())
    plt.show
    return True, ax


def evolution_plot_3p(Input, orientation, silent=None):
    """
    Args:
        Input (ModelData class): class containing the model output
        orientation (string): flag for horizontal ('h') or vertical ('v') orientation
        silent (int): flag for plotting or not plotting

    Returns:
        returnObject (dict):
            True: on succes
            fig: figure object
    """
    if silent is None:
        fig = plt.figure()
        Basin = Input.Basin
        if orientation == "h":
            gs = gridspec.GridSpec(1, 3)
            ax1 = fig.add_subplot(gs[0, 0], projection="3d")
            ax1.view_init(elev=90, azim=0)
            ax2 = fig.add_subplot(gs[0, 1:2], projection="3d")
            ax2.view_init(elev=90, azim=0)
            ax3 = fig.add_subplot(gs[0, 2], projection="3d")
            geometry_plot(Input, Input.Inlets.wit.shape[0] - 1, ax=ax3)
            ax3.view_init(elev=90, azim=0)
        elif orientation == "v":
            gs = gridspec.GridSpec(3, 1)
            ax1 = fig.add_subplot(gs[2, 0], projection="3d")
            ax1.view_init(elev=90, azim=-90)
            ax2 = fig.add_subplot(gs[1:2, 0], projection="3d")
            ax2.view_init(elev=90, azim=-90)
            ax3 = fig.add_subplot(gs[0, 0], projection="3d")
            ax3.view_init(elev=90, azim=-90)
        else:
            raise NameError("No/wrong orientation specified. Choose either 'h' or 'v'.")

        geometry_plot(Input, 0, ax=ax1)
        ax2 = evolution_plot(Input, ax=ax2)[1]
        ax2.set_aspect(
            (
                (ax1.get_xlim()[1] - ax1.get_xlim()[0])
                / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
            )
            ** 2
            * (ax1.get_ylim()[1] / ax2.get_ylim()[1])
        )
        # ax2.set_xlim(
        #     (-0.3 * Basin.length) / 1e3,
        #     (Basin.width + 0.3 * Basin.length) / 1e3,
        # )
        geometry_plot(Input, Input.Inlets.wit.shape[0] - 1, ax=ax3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        fig = None
    return True, fig
