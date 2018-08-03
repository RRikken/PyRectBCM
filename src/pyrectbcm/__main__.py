import time
from pyrectbcm.rectangular_input_generator import ModelData
from pyrectbcm.rectangular_model import rec_model

def main():
    start_time = time.time()
    Input = ModelData('testkees')
    Output = rec_model(Input)
    print("%s seconds" % (time.time() - start_time))

    Output.amplitude_plot()
    Output.evolution_plot()

if __name__ == "__main__":
    main()
