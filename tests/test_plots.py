from pyrectbcm.rectangular_input_generator import ModelData
import py.test
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def test_amp_plot():
    Input = ModelData('testkees')
    uj = np.zeros(Input.Basin.numinlets) + 1
    Input.Inlets.uj = uj[np.newaxis, :]
    Input.Basin.mub = 1-1j*(8/(3*np.pi)*Input.Basin.cd*0.4)/(Input.Ocean.tidefreq*Input.Basin.depth)
    plt.ion()
    assert Input.amplitude_plot(silent = 1)[0]
    assert Input.amplitude_plot()[0]
    plt.close('all')

def test_evo_plot():
    Input = ModelData('testkees')
    Input.Inlets.wit = np.repeat(Input.Inlets.widths, 2, axis = 0)
    plt.ion()
    assert Input.evolution_plot(silent = 1)[0]
    assert Input.evolution_plot()[0]
    plt.close('all')

def test_evo_plot_3p():
    Input = ModelData('testkees')
    Input.Inlets.wit = np.repeat(Input.Inlets.widths, 2, axis = 0)
    plt.ion()
    assert Input.evolution_plot_3p(orientation = 'h', silent = 1)[0]
    assert Input.evolution_plot_3p(orientation = 'v', silent = 1)[0]
    assert Input.evolution_plot_3p(orientation = 'h')[0]
    assert Input.evolution_plot_3p(orientation = 'h')[0]
    plt.close('all')
