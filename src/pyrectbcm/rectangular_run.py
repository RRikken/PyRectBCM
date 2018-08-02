%load_ext autoreload
%reset -f
%autoreload 2

# %%
import time
import numexpr as ne
ne.set_num_threads(4)
import numpy as np

from rectangular_input_generator import rig
from rectangular_model import rec_model
from rectangular_basin_amplitude_plot import amplitude_plot

start_time = time.time()
Input = rig('testkees')
Output = rec_model(Input)

print("%s seconds" % (time.time() - start_time))
amplitude_plot(Output)

# # %%
# %load_ext line_profiler
# %lprun -T prof.txt -m rectangular_model rec_model(Input)
