from pyrectbcm.rectangular_input_generator import rig
import py.test

def test_import():
    Input = rig('testkees')
    assert hasattr(Input, 'Basin')
    assert hasattr(Input, 'Inlets')
    assert hasattr(Input, 'Ocean')
    assert hasattr(Input, 'Pars')
