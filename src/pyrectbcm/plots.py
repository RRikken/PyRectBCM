import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec, patches
from mpl_toolkits.mplot3d import Axes3D, art3d

g = 9.81


def zeta_amplitude_plot(Input, ax=None):
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

    pf = 0
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
        pf = 1
    ax.plot_surface(X, Y, np.abs(eta), cmap=cm.viridis)
    ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )

    if pf:
        plt.show(block=False)
    return True, ax


def u_amplitude_plot(Input, ax=None):
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

    u = 0
    v = 0
    g = 9.81

    X = np.linspace(0, Basin.length, num=200)
    Y = np.linspace(0, Basin.width, num=200)
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
        gusum = 0
        gvsum = 0
        for m in Pars.mrange:
            for n in np.squeeze(Pars.nrange):
                gusum += -1 * (
                    (
                        phij[m, n, j]
                        * (m * np.pi / Basin.length)
                        * np.sin(m * np.pi * X / Basin.length)
                        * np.cos(n * np.pi * Y / Basin.width)
                    )
                    / ((Pars.kmn2[m, n] - Basin.mub * Basin.kb ** 2) * l2[m, n, j])
                )
                gvsum += -1 * (
                    (
                        phij[m, n, j]
                        * np.cos(m * np.pi * X / Basin.length)
                        * (n * np.pi / Basin.width)
                        * np.sin(n * np.pi * Y / Basin.width)
                    )
                    / ((Pars.kmn2[m, n] - Basin.mub * Basin.kb ** 2) * l2[m, n, j])
                )
        u = u + Inlets.widths[:, j] * Inlets.depths[:, j] * Inlets.uj[:, j] * gusum
        v = v + Inlets.widths[:, j] * Inlets.depths[:, j] * Inlets.uj[:, j] * gvsum
    u = u * -1
    v = v * -1
    pf = 0
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
        pf = 1

    ax.plot_surface(X, Y, np.angle(u), cmap=cm.viridis)
    plt.title("angle u")
    ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )
    if pf:
        plt.show(block=False)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, np.angle(v), cmap=cm.viridis)
    plt.title("angle v")
    ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )
    plt.show(block=False)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, np.abs(u), cmap=cm.viridis)
    # plt.title("abs u")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, np.abs(v), cmap=cm.viridis)
    # plt.title("abs v")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, np.sqrt(np.abs(u) ** 2 + np.abs(v) ** 2), cmap=cm.viridis)
    # plt.title("abs uv")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(
    #     X[:, 1:],
    #     Y[:, 1:],
    #     np.angle(v[:, 1:] - v[:, 0:-1] + u[:, 1:] - u[:, 0:-1]),
    #     cmap=cm.viridis,
    # )
    # plt.title("angle dv/dy")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(
    #     X[1:, :],
    #     Y[1:, :],
    #     np.angle(v[1:, :] - v[0:-1, :] + u[1:, :] - u[0:-1, :]),
    #     cmap=cm.viridis,
    # )
    # plt.title("angle dv/dx")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(
    #     X[:, 1:],
    #     Y[:, 1:],
    #     np.fmod(np.angle(v)[:, 1:] - np.angle(v)[:, 0:-1], 1.9 * np.pi),
    #     cmap=cm.viridis,
    # )
    # plt.title("d angle v/dy")
    # ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    # ax.set_zticks([])
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    #
    # ax.plot(
    #     (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    # )
    # plt.show(block=False)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(
        X[1:, :],
        Y[1:, :],
        np.fmod(np.angle(v)[1:, :] - np.angle(v)[0:-1, :], 1.9 * np.pi),
        cmap=cm.viridis,
    )
    plt.title("d angle v/dx")
    ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )
    plt.show(block=False)

    Z = np.abs(np.fmod(np.angle(v)[1:, :] - np.angle(v)[0:-1, :], 1.9 * np.pi))
    zix = Z < 0.1
    Z[zix] = np.nan
    Z[~zix] = 1
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X[1:, :], Y[1:, :], Z, cmap=cm.viridis)
    plt.title("d angle v/dx")
    ax.view_init(elev=90, azim=-90)
    # ax.set_aspect("equal")
    ax.set_zticks([])
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )
    plt.show(block=False)

    return True, ax


def evolution_plot(Input, ax=None):
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

    pf = 0
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
        pf = 1
    ax.plot_surface(
        x / 1e3, y, iw2, cmap=cm.gray_r, rcount=iw2.shape[0] / 2, ccount=iw2.shape[1]
    )

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

    xmin = -0.3 * Basin.length / 1e3
    xmax = (Basin.width + 0.3 * Basin.length) / 1e3
    ymin = 0
    ymax = iw.shape[0] * Pars.dt
    zmin = 0
    zmax = 1e4
    ax.plot(
        (xmin, xmin, xmax, xmax, xmin), (ymin, ymax, ymax, ymin, ymin), 1, color="k"
    )
    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])
    ax.grid(False)
    ax.set_ylabel("Time (years)")

    if pf:
        plt.show(block=False)
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

    pf = 0
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
        pf = 1
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
                (xi, dims[1] / 2 - dims[1] / 100),
                Inlets.wit[t, inlet] / 1e3,
                Inlets.lengths[inlet] * 1e-3 + dims[1] / 50,
                facecolor=cs,
            )
            ax.add_patch(inlet_)
            art3d.pathpatch_2d_to_3d(inlet_, z=0.15)
            ax.plot(
                (xi, xi),
                (dims[1] / 2, dims[1] / 2 + Inlets.lengths[inlet] / 1e3),
                0.2,
                color=cc,
            )
            ax.plot(
                (xi + Inlets.wit[t, inlet] / 1e3, xi + Inlets.wit[t, inlet] / 1e3),
                (dims[1] / 2, dims[1] / 2 + Inlets.lengths[inlet] / 1e3),
                0.2,
                color=cc,
            )
    # Set plot options
    ax.axis("off")
    ax.plot(
        (dims[2], dims[2], dims[3], dims[3], dims[2]),
        (dims[0], dims[1], dims[1], dims[0], dims[0]),
        1,
        color="k",
    )
    ax.grid(False)
    # ax.set_aspect(((dims[3] - dims[2]) / (dims[1] - dims[0])) ** 2)
    # ax.apply_aspect()
    ax.set_xlim3d(dims[2], dims[3])
    ax.set_ylim3d(dims[0], dims[1])
    ax.set_zlim3d(
        0, 1e2
    )  # large enough to not see stacking effects of different layers

    if pf:
        plt.show(block=False)
    return True, ax


def evolution_plot_3p(Input, orientation):
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
    fig = plt.figure()
    Basin = Input.Basin
    if orientation == "h":
        gs = gridspec.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax1.view_init(elev=90, azim=0)
        ax1.set_xlabel("along-shore")
        ax1.set_title("Initial\nconfiguration", y=0.875)
        ax1.set_anchor((0, 0))

        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        ax2.view_init(elev=90, azim=0)
        ax2.set_title("Evolution", y=0.875)
        ax2.set_anchor((1 / 3, 0))

        ax3 = fig.add_subplot(gs[0, 2], projection="3d")
        ax3.view_init(elev=90, azim=0)
        ax3.set_title("Equilibrium\n configuration", y=0.875)
        ax3.set_anchor((2 / 3, 0))
    elif orientation == "v":
        gs = gridspec.GridSpec(3, 1)

        ax1 = fig.add_subplot(gs[2, 0], projection="3d")
        ax1.view_init(elev=90, azim=-90)
        ax1.set_title("Initial\nconfiguration")

        ax2 = fig.add_subplot(gs[1, 0], projection="3d")
        ax2.view_init(elev=90, azim=-90)
        ax2.set_title("Evolution")

        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        ax3.view_init(elev=90, azim=-90)
        ax3.set_title("Equilibrium\n configuration")
    else:
        raise NameError("No/wrong orientation specified. Choose either 'h' or 'v'.")
    ax1 = geometry_plot(Input, 0, ax=ax1)[1]
    ax2 = evolution_plot(Input, ax=ax2)[1]
    ax3 = geometry_plot(Input, Input.Inlets.wit.shape[0] - 1, ax=ax3)[1]

    # ax2.set_aspect(
    #     (
    #         (ax1.get_xlim()[1] - ax1.get_xlim()[0])
    #         / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    #     )
    #     ** 2
    #     * (ax1.get_ylim()[1] / ax2.get_ylim()[1])
    # )
    # gs.tight_layout(fig, pad = 0.5, h_pad = 0, w_pad = 0)
    # fig.tight_layout()
    fig.show()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return True, fig
