"""Record-level validation for a fixed, externally fitted family prior."""
import numpy as np
from .output_error import fit, prepare, predict


def select_strength(records, prior, strengths, prior_scale=0.05):
    """Leave each independent episode out; caller must exclude it from prior.

    The prior is fitted to other clients only in 7B. This function never takes
    evaluation trajectories or ground-truth parameters as arguments.
    """
    strengths=sorted(set(strengths))
    if len(records)<2 or not strengths or strengths[0]<0:
        raise ValueError('need independent validation records and valid strengths')
    rows=[]
    for fold,validation in enumerate(records):
        training=[r for j,r in enumerate(records) if j!=fold]
        heldout=prepare([validation])
        for strength in strengths:
            p,diag=fit(training,prior,strength,prior_scale)
            if not diag['success']:raise RuntimeError('unconverged validation fit')
            loss=float(np.mean(((predict(p,heldout)-heldout['y'])/np.deg2rad([.05,.1]))**2))
            rows.append(dict(fold=fold,strength=strength,validation_loss=loss,
                             cf_over_m=p[0],cr_over_m=p[1],m_over_iz=p[2],**diag))
    means={s:np.mean([r['validation_loss'] for r in rows if r['strength']==s]) for s in strengths}
    selected=min(strengths,key=lambda s:(means[s],s))
    return selected,rows
