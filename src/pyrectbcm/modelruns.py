from pyrectbcm.rectangular_input_generator import ModelData
from pyrectbcm.rectangular_model import rec_model


def run_model(seed=None):

    # Prepare input data
    Input = ModelData("testlocation", seed=seed)

    # Run the model
    Output = rec_model(Input, silent=1)
    return Input, Output
