""" This file holds all the classes that contain data for the model."""
import numpy as np
from pyrectbcm.constants import GRAVITY


class Basin:
    """
    The Basin object contains the parameters of the backbarrier basin
    """

    def __init__(self):
        self._depth = None
        self._length = None
        self._width = None
        self._number_of_inlets = None
        self.kb = data.tidefreq / np.sqrt(GRAVITY * data.basindepth)
        self.cd = 2.5e-3
        self.ub = 0

    def get_depth(self) -> float:
        """ Depth of the basin in meters, h_0. """
        return self._depth

    def set_depth(self, basin_depth: float) -> None:
        self._depth = basin_depth

    def get_length(self) -> float:
        """ Length of the basin in meters, L. """
        return self._length

    def set_length(self, basin_length: float) -> None:
        self._length = basin_length

    def get_width(self) -> float:
        """ Width of the basin in meters, B. """
        return self._width

    def set_width(self, basin_width: float) -> None:
        self._width = basin_width

    def get_number_of_inlets(self) -> int:
        """ Number of inlets, J_init """
        return self._number_of_inlets

    def set_number_of_inlets(self, number_of_inlets: int) -> None:
        self._number_of_inlets = number_of_inlets


class Inlets:
    """
    The Inlet object contains the parameters and dimensions of the tidal inlets.
    These dimensions change during a simulation.
    N.B. the inlet width is randomized slightly.
    """

    def __init__(self, seed=None):
        self._shape_factor = data.inletshape
        self._sediment_import = data.sedimport
        self._equilibrium_velocity = data.ueq
        coastalpoints = np.linspace(0, basin.width, basin.numinlets + 1)
        self._locations = np.array([coastalpoints[0:-1] + coastalpoints[1] / 2])
        np.random.seed(seed)
        self._widths = (0.2 * np.random.rand(1, basin.numinlets) + 0.9) * data.inletwidth
        self._depths = self.widths * self.shape
        self._lengths = np.full(basin.numinlets, data.inletlength)
        self.cd = basin.cd
        self.uj = ocean.tideamp * np.sqrt(gravity / self.depths)
        self.wit = np.zeros((1, basin.numinlets))

    def get_shape_factor(self) -> float:
        """ The shape factor, no unit, phi_j^2. """
        return self._shape_factor

    def set_shape_factor(self, factor: float) -> None:
        self._shape_factor = factor

    def get_sediment_import(self) -> float:
        """ The sediment import through the inlet in m^3/year, M. """
        return self._sediment_import

    def set_sediment_import(self, sediment_import: float) -> None:
        self._sediment_import = sediment_import

    def get_equilibrium_velocity(self) -> float:
        """ Equilibrium velocity through the inlet in m/s, U_eq. """
        return self._equilibrium_velocity

    def set_equilibrium_veloctity(self, equilibrium_velocity) -> None:
        self._equilibrium_velocity = equilibrium_velocity

    def get_locations(self) -> np.ndarray:
        """ Numpy array with the locations of the inlets. """
        return self._locations

    def set_locations(self, locations: np.ndarray) -> None:
        self._locations = locations

    def get_widths(self) -> np.ndarray:
        """ The widths of the inlets on the locations in this class in meters, b_j. """
        return self._widths

    def set_widths(self, inlet_widths: np.ndarray) -> None:
        self._widths = inlet_widths

    def get_depths(self) -> np.ndarray:
        """ The depths of the inlets in this class in meters. h_j"""
        return self._depths

    def set_depths(self, inlet_depths: np.ndarray) -> None:
        self._depths = inlet_depths

    def get_lengths(self) -> np.ndarray:
        """ The lengths of the inlets in this class in meters, l. """
        return self._lengths

    def set_lengths(self, inlet_lengths: np.ndarray) -> None:
        self._lengths = inlet_lengths


class Ocean:
    """
    The Ocean object contains the parameters of the outer sea.
    """

    def __init__(self):
        self.ko = data.tidefreq / np.sqrt(GRAVITY * self.depth)

    def get_depth(self) -> float:
        """ The depth of the ocean near the inlets, in meters, h_0"""
        return self._depth

    def set_depth(self, ocean_depth: float) -> None:
        self._depth = ocean_depth

    def get_wave_number(self) -> float:
        """ The wave number in rad/m, k_0. """
        return self._wave_number

    def set_wave_number(self, wave_number: float) -> None:
        self._wave_number = wave_number

    def get_tidal_range(self) -> float:
        """ The range of the tide in meters, H_tide. """
        return self._tidal_range

    def set_tidal_range(self, tidal_range: float):
        self._tidal_range = tidal_range

    def get_tidal_frequency(self) -> float:
        """ The tidal frequency in rad/s, omega. """
        return self._tidal_frequency

    def set_tidal_frequency(self, tidal_frequency: float) -> None:
        self._tidal_frequency = tidal_frequency


class Parameters:
    """
    The Pars object contains miscellaneous parameters.
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
        Holds all the data needed for a model run.
    """
    def __init__(self):
        self._basin = None
        self._ocean = None
        self._inlets = None
        self._parameters = None

    def get_basin(self) -> Basin:
        return self._basin

    def set_basin(self, model_basin: Basin) -> None:
        self._basin = model_basin

    def get_ocean(self) -> Ocean:
        return self._ocean

    def set_ocean(self, model_ocean: Ocean) -> None:
        self._ocean = model_ocean

    def get_inlets(self) -> Inlets:
        return self._inlets

    def set_inlets(self, model_inlets: Inlets) -> None:
        self._inlets = model_inlets

    def get_parameters(self) -> Parameters:
        return self.__parameters

    def set_parameters(self, model_parameters: Parameters) -> None:
        self.__parameters = model_parameters

    def zeta_amplitude_plot(self, ax=None):
        from pyrectbcm.plots import zeta_amplitude_plot

        return zeta_amplitude_plot(self, ax)

    def u_amplitude_plot(self, ax=None):
        from pyrectbcm.plots import u_amplitude_plot

        return u_amplitude_plot(self, ax)

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
