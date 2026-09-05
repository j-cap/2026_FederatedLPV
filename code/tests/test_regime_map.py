import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from experiment_6a_regime_map import fleet, remap, design
from experiment_4f_structured_identification import StructuredModel
from federated_lpv import sample_fleet
from federated_lpv.vehicle import family_centers

def test_nominal_separation_reproduces_original_fleet():
    for old,new in zip(sample_fleet(21),fleet(21,1)):
        np.testing.assert_allclose(list(vars(old.parameters).values()),list(vars(new.parameters).values()),rtol=1e-14)

def test_collapsed_centers_preserve_scatter_without_family_shift():
    centers=family_centers(); nominal=centers['nominal']
    for old,new in zip(sample_fleet(21),fleet(21,0)):
        for key in ('mass','yaw_inertia','front_stiffness','rear_stiffness'):
            np.testing.assert_allclose(getattr(new.parameters,key)/getattr(nominal,key),
                getattr(old.parameters,key)/getattr(centers[old.family],key))

def test_speed_mapping_and_design_cover_new_envelope():
    params={f:np.array([50.,55.,.6]) for f in ('global','nominal','heavy','handling')}
    models={m:StructuredModel('global' if m in ('M1','M3') else 'family',
        'constant' if m in ('M1','M2') else 'lpv',params) for m in ('M1','M2','M3','M4')}
    for bounds in ((18,22),(15,25),(10,30)):
        np.testing.assert_allclose(remap(np.array([12,20,28]),bounds),[bounds[0],20,bounds[1]])
        controllers=design(models,bounds)
        np.testing.assert_allclose(controllers['M4'].speeds[[0,-1]],bounds)
        np.testing.assert_allclose(controllers['M2'].speeds,[20])
