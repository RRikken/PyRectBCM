import numpy as np
from pyrectbcm.data_classes import Basin, Inlets, Ocean, Parameters, ModelData


class ModelDataFactory(object):
    """ The model data factory constructs a ModelData object and adds al the values and objects to it. """

    @staticmethod
    def build(data: dict) -> ModelData:
        """ Functions that builds the ModelData. Data should be a dict with all the needed variables.
            Example of a data dict with all the needed keys (with example values):
            {
                "name": "testkees",
                "tidal_range": 1,
                "tidal_frequency": 2 * pi / 44712,
                "basin_width": 20e3,
                "basin_length": 10e3,
                "basin_depth": 2,
                "ocean_depth": 30,
                "wave_number": 8.2e-6 * 0,
                "inlet_length": 1e3,
                "inlet_depth": 2,
                "inlet_width": 0.4e3,
                "inlet_shape_factor": 0.005,
                "number_of_inlets": 3,
                "equilibrium_velocity": 1,
                "sediment_import": 1e5,
                "time_step": 0.5,
                "end": 2,
            }
        """
        model_data = ModelData()
        model_data.set_basin(ModelDataFactory.build_basin(data))
        model_data.set_ocean(ModelDataFactory.build_ocean(data))
        model_data.set_inlets(ModelDataFactory.build_inlets(data, None))

        return model_data

    @staticmethod
    def build_ocean(data: dict) -> Ocean:
        """ This functions builds an ocean object.
            The following keys need to be in the data dict (with example values):
            {
                "ocean_depth": 30,
                "wave_number": 8.2e-6 * 0,
                "tidal_range": 1,
                "tidal_frequency": 2 * pi / 44712,
            }
        """
        ocean = Ocean()
        ocean.set_depth(data["ocean_depth"])
        ocean.set_wave_number(data["wave_number"])
        ocean.set_tidal_range(data["tidal_range"])
        ocean.set_tidal_frequency(data["tidal_frequency"])
        # TODO: Find out what ko means and add it to Ocean
        # self.ko = data.tidefreq / np.sqrt(gravity * self.depth)

        return ocean

    @staticmethod
    def build_basin(data: dict) -> Basin:
        """ This functions builds a basin object.
            The following keys need to be in the data dict (with example values):
            {
                "basin_width": 20e3,
                "basin_length": 10e3,
                "basin_depth": 2,
                "number_of_inlets": 3,
            }
        """
        basin = Basin()
        basin.set_depth(data["basin_depth"])
        basin.set_length(data["basin_length"])
        basin.set_width(data["basin_width"])
        basin.set_number_of_inlets(data["number_of_inlets"])

        # TODO: find out what kb, cd, and ub mean and add them to basin and the factory.
        # self.kb = data.tidefreq / np.sqrt(gravity * data.basindepth)
        # self.cd = 2.5e-3
        # self.ub = 0

        return basin

    @staticmethod
    def build_inlets(data: dict, seed: str = None) -> Inlets:
        """
        Parameters
        ----------
        data: dict
        seed: int or 1-d array_like, optional

        This functions builds an inlets object.
            The following keys need to be in the data dict (with example values):
            {
                "basin_width": 20e3,
                "inlet_length": 1e3,
                "inlet_depth": 2,
                "inlet_width": 0.4e3,
                "inlet_shape_factor": 0.005,
                "number_of_inlets": 3,
                "equilibrium_velocity": 1,
                "sediment_import": 1e5,
            }
        """
        inlets = Inlets()
        inlets.set_shape_factor(data["inlet_shape_factor"])
        inlets.set_sediment_import(data["sediment_import"])
        inlets.set_equilibrium_veloctity(data["equilibrium_velocity"])

        # The locations are divided over the length of the coast of the basin. The locations of the inlets are evenly
        # spaced.
        coastal_points = np.linspace(0, data["basin_width"], data["number_of_inlets"] + 1)
        locations = np.array([coastal_points[0:-1] + coastal_points[1] / 2])
        inlets.set_locations(locations)

        # The widths of the inlets are randomly generated around the inlet width from the dataset. If a seed is added,
        # the randomness can pseudo!.
        # TODO: random seed is deprecated, see method description for improvement options
        np.random.seed(seed)
        widths = (0.2 * np.random.rand(1, data["number_of_inlets"]) + 0.9) * data["inlet_width"]
        inlets.set_widths(widths)

        inlets.set_lengths(np.full(data["number_of_inlets"], data["inlet_length"]))
        inlets.set_depths(inlets.get_widths() * inlets.get_shape_factor())

        # TODO: find out what cd, uj, wit mean and finish the Inlets.
        # self.cd = basin.cd
        # self.uj = ocean.tideamp * np.sqrt(gravity / self.depths)
        # self.wit = np.zeros((1, basin.numinlets))

        return inlets

    @staticmethod
    def build_parameters(data: dict) -> Parameters:
        """

        Parameters
        ----------
        data

        Returns
        -------
        Parameters
        """
        # TODO: none of the parameter properties are clear, find out what it all means
        parameters = Parameters(Basin(), data)

        return parameters
