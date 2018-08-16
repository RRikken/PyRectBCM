from pyrectbcm.rectangular_input_generator import ModelData
from pyrectbcm.rectangular_model import rec_model
import matplotlib as mpl
import time

mpl.use("qt5agg")


def main(location=None):
    if location is None:
        location = "testkees"
    start_time = time.time()
    Input = ModelData(location)
    Output = rec_model(Input, silent=0)
    print("%s seconds" % (time.time() - start_time))
    # Output.amplitude_plot()
    # Output.evolution_plot()
    Output.evolution_plot_3p(orientation="h")
    return Output


if __name__ == "__main__":
    Output = main("testlocation")
