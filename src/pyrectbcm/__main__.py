import time
import numexpr as ne
import numpy as np
from pyrectbcm.rectangular_input_generator import rig
from pyrectbcm.rectangular_model import rec_model
from pyrectbcm.rectangular_basin_amplitude_plot import amplitude_plot
# from rectangular_input_generator import rig
# from rectangular_model import rec_model
# from rectangular_basin_amplitude_plot import amplitude_plot

def main():
    start_time = time.time()
    Input = rig('testkees')
    Output = rec_model(Input)

    print("%s seconds" % (time.time() - start_time))
    amplitude_plot(Output)

if __name__ == "__main__":
    main()
