from pyrectbcm.rectangular_input_generator import rig
from pyrectbcm.rectangular_model import rec_model
from pyrectbcm.rectangular_basin_amplitude_plot import amplitude_plot
from multiprocessing import Pool
import pickle
Inputs = []
Outputs = []

def run_model():
    Input = rig('testkees')
    Output = rec_model(Input, silent = 1)
    return Input, Output

def runs(N = 0):
    if N ==0:
        raise NameError("Can't do zero runs")

    pool = Pool(processes=4)
    results = []
    for n in range(N):
        results.append(pool.apply_async(run_model, args = ()))
        print(n)

    for r in results:
        Input, Output = r.get()
        Inputs.append(Input)
        Outputs.append(Output)
        print(Output.Inlets.widths)
    pool.close()
    pool.join()

if __name__ == '__main__':
    runs(4)
    with open('Input.pkl', 'wb') as output:
        pickle.dump(Inputs, output, pickle.HIGHEST_PROTOCOL)
    with open('Output.pkl', 'wb') as output:
        pickle.dump(Outputs, output, pickle.HIGHEST_PROTOCOL)
