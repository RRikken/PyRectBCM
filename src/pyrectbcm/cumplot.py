from matplotlib import gridspec, pyplot
from matplotlib.lines import Line2D
import numpy as np
import pickle

plt = pyplot


def mcplot(start, end, path):
    with open("{}{}.pkl".format(path, start), "rb") as input:
        Output = pickle.load(input)
    Pars = Output.Pars
    ndt = Output.Inlets.wit.shape[0]
    ninlet = np.zeros((end, ndt))
    ainlet = np.zeros((end, ndt, Output.Basin.numinlets))
    pinlet = np.zeros((end, ndt))
    t = np.linspace(0, Pars.tend, ndt)

    pad = 0.5
    fig = plt.figure(figsize=(8.3, 8.3 / 1.5), dpi=100)
    gs = gridspec.GridSpec(2, 3, wspace=pad, hspace=pad)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    Abasin = Output.Basin.length * Output.Basin.width

    for rn in range(start, end + 1):
        with open("{}{}.pkl".format(path, rn), "rb") as input:
            Output = pickle.load(input)
        Output.Inlets.wit = np.array(Output.Inlets.wit.todense())
        ninlet[rn - 1, :] = np.count_nonzero(Output.Inlets.wit, axis=1).flatten()
        ninlets = Output.Inlets.wit.shape[1]
        if ninlets > ainlet.shape[2]:
            ainlet_ = np.zeros((end, ndt, ninlets))
            ainlet_[:, :, : ainlet.shape[2]] = ainlet
            ainlet = ainlet_
        uj = (
            np.cbrt(
                (Output.Inlets.wit[1:, :] - Output.Inlets.wit[:-1, :])
                / Output.Pars.dt
                * (Output.Inlets.lengths[0] / Output.Inlets.sedimport)
                + 1
            )
            * Output.Inlets.ueq
        )
        k = Output.Ocean.tidefreq / (2 * uj)
        ainlet[rn - 1, :, :ninlets] = (
            np.power(Output.Inlets.wit, 2) * Output.Inlets.shape
        )
        ainlet[rn - 1, :, :] /= ainlet[rn - 1, 0, :]
        pinlet[rn - 1, :-1] = np.sum(
            np.power(Output.Inlets.wit[:-1, :], 2) * Output.Inlets.shape / k, axis=1
        ).flatten()
        pinlet[rn - 1, -1] = pinlet[rn - 1, -2]
        plt.sca(ax0)
        plt.plot(
            t,
            ainlet[rn - 1, :, :],
            color=(0.7, 0.7, 0.7, 256 / 256),
            linewidth=0.05,
            zorder=1,
        )
        plt.sca(ax1)
        plt.plot(
            t,
            ninlet[rn - 1, :],
            color=(0.7, 0.7, 0.7, 256 / 256),
            linewidth=0.05,
            zorder=1,
        )
        plt.sca(ax2)
        plt.plot(
            t,
            pinlet[rn - 1, :] / (Abasin * Output.Ocean.tideamp),
            color=(0.7, 0.7, 0.7, 256 / 256),
            linewidth=0.05,
            zorder=1,
        )

    ainlet[ainlet == 0] = np.nan
    amean = np.nanmean(ainlet[start : end + 1, :, :], axis=(0, 2))
    nmean = np.mean(ninlet[start : end + 1, :], axis=0)
    pmean = np.mean(pinlet[start : end + 1, :], axis=0)

    prctla = np.nanpercentile(ainlet, q=[0, 25, 50, 75, 100], axis=(0, 2))
    prctln = np.percentile(ninlet, q=[0, 25, 50, 75, 100], axis=0)
    prctlp = np.percentile(pinlet, q=[0, 25, 50, 75, 100], axis=0)

    cs = np.array([0, 0, 0])
    xlim = np.array([0, 0.5e3])
    ylim = np.array([[0, 5], [0, 55], [0, 4]])

    plt.sca(ax0)
    plt.fill_between(
        t, prctla[1, :], prctla[3, :], color=cs, alpha=0.4, linewidth=1, zorder=3
    )
    plt.plot(t, prctla[2, :], "-", color=cs, alpha=0.8, linewidth=2, zorder=4)
    plt.grid(b=True, which="major")
    plt.xlabel("time (years)")
    plt.ylabel(r"$\frac{A_{j,\mathrm{all}}}{A_\mathrm{init}} (-)$")
    plt.title("a")
    plt.xlim(xlim)
    plt.ylim(ylim[0])
    ax0.set_aspect(max(t) / (ylim[0, 1] - ylim[0, 0]))

    plt.sca(ax1)
    plt.fill_between(
        t, prctln[1, :], prctln[3, :], color=cs, alpha=0.4, linewidth=1, zorder=3
    )
    plt.plot(t, prctln[2, :], "-", color=cs, alpha=0.8, linewidth=2, zorder=4)
    plt.grid(b=True, which="major")
    plt.xlabel("time (years)")
    plt.ylabel(r"$J_\mathrm{total} \; (-)$")
    plt.title("b")
    plt.xlim(xlim)
    plt.ylim(ylim[1])
    ax1.set_aspect(max(t) / (ylim[1, 1] - ylim[1, 0]))

    plt.sca(ax2)
    prctlp /= Abasin * Output.Ocean.tideamp
    plt.fill_between(
        t, prctlp[1, :], prctlp[3, :], color=cs, alpha=0.4, linewidth=1, zorder=3
    )
    plt.plot(t, prctlp[2, :], "-", color=cs, alpha=0.8, linewidth=2, zorder=4)
    plt.grid(b=True, which="major")
    plt.xlabel("time (years)")
    plt.ylabel(r"$\frac{P_\mathrm{total}}{P_\mathrm{ref, total}} \; (-)$")
    plt.title("c")
    plt.xlim(xlim)
    plt.ylim(ylim[2])
    ax2.set_aspect(max(t) / (ylim[2, 1] - ylim[2, 0]))

    ax = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    legend_elements = [
        Line2D([0], [0], color=(0.7, 0.7, 0.7), linewidth=0.5),
        Line2D([0], [0], color=cs, alpha=0.8, linewidth=2),
        plt.fill_between([0], [0], [0], color=cs, alpha=0.4, linewidth=0.5),
    ]
    ax.legend(
        legend_elements,
        ["Individual inlet/model run", "Median", "50% envelope"],
        loc="upper center",
        ncol=3,
    )
    plt.axis("off")

    plt.show(block=False)
    return fig


if __name__ == "__main__":
    path = "D:/Koen/Documents/UT/PhD/Repository/PyRectBCM/Output/"
    opath = path + "Output_run"
    start = 0
    end = 10
    fig = mcplot(start, end, opath)
