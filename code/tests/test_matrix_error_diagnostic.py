import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'paper'))
from matrix_error_diagnostic import balanced_error


def test_fixed_scaling_is_invariant_to_entry_unit_changes():
    errors=np.array([[1.,2.,3.,4.,5.,6.],[3.,2.,1.,-4.,-5.,6.]])
    scales=np.array([2.,3.,4.,5.,6.,7.])
    units=np.array([180/np.pi,.1,2.,1.,180/np.pi,10.])
    np.testing.assert_allclose(balanced_error(errors,scales),
                               balanced_error(errors*units,scales*units))


def test_equal_scaled_errors_have_expected_rms():
    scales=np.array([2.,3.,4.,5.,6.,7.])
    np.testing.assert_allclose(balanced_error(np.stack([scales,-2*scales]),scales),[1.,2.])
