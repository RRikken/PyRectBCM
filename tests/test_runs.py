import py.test
import pyrectbcm.multirun as multirun
from pyrectbcm.rectangular_model import rec_model
from pyrectbcm.rectangular_input_generator import ModelData
from numpy import nan

def test_single_run():
    Input = ModelData("testkees")
    Input, Output = multirun.run_model()
    assert hasattr(Input, "Basin")
    assert hasattr(Input, "Inlets")
    assert hasattr(Input, "Ocean")
    assert hasattr(Input, "Pars")
    assert hasattr(Output, "Basin")
    assert hasattr(Output, "Inlets")
    assert hasattr(Output, "Ocean")
    assert hasattr(Output, "Pars")

    Output = rec_model(Input)
    assert hasattr(Output, "Basin")
    assert hasattr(Output, "Inlets")
    assert hasattr(Output, "Ocean")
    assert hasattr(Output, "Pars")

    Input = ModelData("testkees")
    Input.Inlets.widths = [nan, nan, nan]
    with py.test.raises(NameError):
        Output = rec_model(Input)

def test_multi_run():
    with py.test.raises(NameError):
        assert multirun.runs()

    assert multirun.runs(1)
