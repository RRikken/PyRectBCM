from pyrectbcm.rectangular_input_generator import ModelData
from pyrectbcm.rectangular_model import rec_model


def main(location="testkees"):
    Input = ModelData(location, seed=10)
    Output = rec_model(Input, silent=True)
    Output.evolution_plot_3p(orientation="h")
    return Output


if __name__ == "__main__":
    Output = main("testlocation")
