from pyrectbcm.modelruns import run_model
from multiprocessing import Pool
import numpy as np
import pickle
from scipy import sparse

dspath = "D:/Koen/Documents/UT/PhD/Repository/PyRectBCM/Output/"


def runs(N=0, spath=dspath):
    if N == 0:
        raise NameError("Can't do zero runs")

    # Prepare individual workers
    cpus = 4
    pool = Pool(processes=4)

    # Split number of model runs in cycles with the same size as the number of workers
    cycles = np.ceil(N / cpus)
    subNrange = range(cpus)

    # Start model runs
    rn = 0
    for c in range(cycles.astype(int)):
        # Create jobs
        results = []
        for n in subNrange:
            ix = n + c * cpus

            results.append(pool.apply_async(run_model, kwds={"seed": ix}))
            print(ix)

        # Gather results and write to harddisk
        for r in results:
            try:
                Input, Output = r.get()
                Output.Inlets.wit = sparse.coo_matrix(Output.Inlets.wit)

                with open("{}Input_run{}.pkl".format(spath, rn), "wb") as output:
                    pickle.dump(Input, output, pickle.HIGHEST_PROTOCOL)
                with open("{}Output_run{}.pkl".format(spath, rn), "wb") as output:
                    pickle.dump(Output, output, pickle.HIGHEST_PROTOCOL)
                rn += 1
            except:
                pass
    pool.close()
    pool.join()
    return True


if __name__ == "__main__":
    runs(10)
