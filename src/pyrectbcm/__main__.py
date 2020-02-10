from pyrectbcm.rectangular_input_generator import ModelData
from pyrectbcm.rectangular_model import rec_model
import time


def main(location="testkees"):

    # Start timer
    start_time = time.time()

    # Prepare input data
    Input = ModelData(location)

    # Run the model
    Output = rec_model(Input, silent=True)

    # Plot results
    Output.evolution_plot_3p(orientation="h")

    # Print elapsed time
    print("%s seconds" % (time.time() - start_time))
    return Output


if __name__ == "__main__":
    Output = main("testlocation")
