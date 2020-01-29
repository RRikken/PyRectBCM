import numpy as np

g = 9.81


class Basin:
    """
    The Basin object contains the parameters of the backbarrier basin
    """

    def __init__(self, data):
        self.depth = data.basindepth
        self.length = data.basinlength
        self.width = data.basinwidth
        self.numinlets = data.numinlets
        self.kb = data.tidefreq / np.sqrt(g * data.basindepth)
        self.cd = 2.5e-3
        self.ub = 0


class Inlets:
    """
    The Inlet object contains the parameters and dimensions of the tidal inlets.
    These dimensions change during a simulation.
    N.B. the inlet width is randomized slightly.
    """

    def __init__(self, data, basin, ocean, seed=None):
        self.shape = data.inletshape
        self.sedimport = data.sedimport
        self.ueq = data.ueq
        coastalpoints = np.linspace(0, basin.width, basin.numinlets + 1)
        self.locations = np.array([coastalpoints[0:-1] + coastalpoints[1] / 2])
        # self.widths = np.repeat(np.array(400, dtype = np.float64), basin.numinlets)
        # self.widths = self.widths[np.newaxis, :]
        np.random.seed(seed)
        self.widths = (0.2 * np.random.rand(1, basin.numinlets) + 0.9) * data.inletwidth
        self.depths = self.widths * self.shape
        self.lengths = np.full(basin.numinlets, data.inletlength)
        self.cd = basin.cd
        self.uj = ocean.tideamp * np.sqrt(g / self.depths)
        self.wit = np.zeros((1, basin.numinlets))


class Ocean:
    """
    The Ocean object contains the parameters of the outer sea.
    """

    def __init__(self, data):
        self.depth = data.oceandepth
        self.wavenumber = data.wavenumber
        self.tideamp = data.tideamp
        self.tidefreq = data.tidefreq
        self.ko = data.tidefreq / np.sqrt(g * self.depth)


class Pars:
    """
    The Pars object contains miscallaneous parameters.
    """

    def __init__(self, basin, data):
        self.mtrunc = 50
        self.ntrunc = 50
        self.mrange = np.arange(self.mtrunc + 1)
        self.nrange = np.arange(self.ntrunc + 1)
        self.nrange = self.nrange[np.newaxis, :]
        self.kmn2 = (self.mrange[:, np.newaxis] * np.pi / basin.length) ** 2 + (
            self.nrange * np.pi / basin.width
        ) ** 2
        self.tstart = 0
        self.dt = data.timestep
        self.tend = data.end


class ModelData:
    """
    The ModelData object contains the other parameter objects.
    """

    def __init__(self, location, seed=None):
        if location == "testkees":
            from pyrectbcm.input_locations import testkees

            data = testkees

        elif location == "testlocation":
            from pyrectbcm.input_locations import testlocation

            data = testlocation

        else:
            raise NameError("location unknown")

        self.Basin = Basin(data)
        self.Ocean = Ocean(data)
        self.Inlets = Inlets(data, self.Basin, self.Ocean, seed)
        self.Pars = Pars(self.Basin, data)

    def zeta_amplitude_plot(self, ax=None):
        from pyrectbcm.plots import zeta_amplitude_plot

        return zeta_amplitude_plot(self, ax)

    def u_amplitude_plot(self):
        from pyrectbcm.plots import u_amplitude_plot

        return u_amplitude_plot(self)

    def evolution_plot(self, ax=None):
        from pyrectbcm.plots import evolution_plot

        rv, ax = evolution_plot(self, ax)
        ax.view_init(elev=90, azim=0)
        return rv, ax

    def geometry_plot(self, ax=None):
        from pyrectbcm.plots import geometry_plot

        rv, ax = geometry_plot(self, ax)
        return rv, ax

    def evolution_plot_3p(self, orientation):
        from pyrectbcm.plots import evolution_plot_3p

        return evolution_plot_3p(self, orientation)
