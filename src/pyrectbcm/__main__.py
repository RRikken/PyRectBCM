from pyrectbcm.rectangular_model import rec_model
import time
from pyrectbcm.rectangular_input_generator import Basin, Ocean, Inlets, Pars
from pyrectbcm.model_data_factory import ModelDataFactory


def main(location="testkees"):

    # Start timer
    start_time = time.time()

    # Prepare input data
    data_for_model = ModelDataFactory.build(data)

    # Run the model
    model_output = rec_model(data_for_model, silent=True)

    # Plot results
    model_output.evolution_plot_3p(orientation="h")

    # Print elapsed time
    print("%s seconds" % (time.time() - start_time))

    return model_output


if __name__ == "__main__":
    main_model_output = main("testlocation")
