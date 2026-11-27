"""New draws from the CFL-FF manuscript distribution, mapped to tanh slopes.

This is a documented adaptation, not a replay of the original Pacejka fleet.
Only the lateral blocks are sampled; no longitudinal random draws are consumed.
"""
from __future__ import annotations
import numpy as np


def draw_vehicle(family, rng, specification):
    a = specification['anchors'][family]
    s = specification['variation']
    dm, di = rng.multivariate_normal([0., 0.],
        s['family_log_sd']**2*np.array([[1., s['mass_inertia_correlation']],
                                      [s['mass_inertia_correlation'], 1.]]))
    mass = a['mass_kg']*np.exp(dm)
    inertia = a['inertia_kg_m2']*np.exp(di)
    lf, lr = np.array(a['geometry_m'])*np.exp(rng.normal(0., s['family_log_sd']))
    tire = np.array(a['lateral_BCE']) + rng.normal(size=(2, 3))*s['BCE_additive_sd']
    gamma = rng.uniform(*s['payload_fraction'])
    x = rng.uniform(*s['payload_position_wheelbase'])*(lf+lr)
    # x>0 is forward of the unloaded CG. Translate both masses to the new CG.
    shift = gamma*x/(1.+gamma)
    inertia += mass*gamma/(1.+gamma)*x*x
    mass *= 1.+gamma
    lf -= shift
    lr += shift
    tire[:, 0] *= rng.uniform(*s['tire_condition'], size=2)
    micro = np.exp(rng.normal(0., s['client_log_sd'], size=6))
    mass *= micro[0]; inertia *= micro[1]
    lf *= micro[2]; lr *= micro[3]; tire[:, 0] *= micro[4:]
    mu = specification['friction']
    loads = mass*9.81*np.array([lr, lf])/(lf+lr)
    stiffness = tire[:, 0]*tire[:, 1]*mu*loads
    theta = [stiffness[0]/mass, stiffness[1]/mass, mass/inertia, mu,
             *specification['relaxation_lengths_m']]
    if min(mass, inertia, lf, lr, *theta) <= 0:
        raise ValueError('Nonphysical draw; no rejection/resampling rule is permitted')
    return dict(geometry_m=[float(lf), float(lr)], physical_parameters=list(map(float, theta)),
        physical_draw=dict(mass_kg=float(mass), inertia_kg_m2=float(inertia),
            lateral_BCE=tire.tolist(), stiffness_N_rad=stiffness.tolist(),
            payload_fraction=float(gamma), payload_position_m=float(x)))


def nominal_center(specification):
    """Engineering population center from independent design draws, never test fits."""
    rng = np.random.default_rng(specification['nominal_design_seed'])
    logs = [np.log(draw_vehicle(f, rng, specification)['physical_parameters'])
            for f in specification['anchors']
            for _ in range(specification['nominal_design_draws_per_family'])]
    return np.exp(np.mean(logs, axis=0)).tolist()


def generate_fleet(seed, specification, speeds):
    """Six clients/family; two per speed. Opaque IDs and randomized record order."""
    rng = np.random.default_rng(np.random.SeedSequence([specification['namespace'], seed]))
    clients = []
    for family_index, family in enumerate(specification['anchors']):
        for repeat in range(2):
            for speed_index, speed in enumerate(speeds):
                c = draw_vehicle(family, rng, specification)
                c.update(family=family, training_speed=speed,
                    # Three six-client groups, each with 2/family and 2/speed.
                    # This is a stratified mixed control, not label-free clustering.
                    mixed_group=(family_index+speed_index+repeat) % 3)
                clients.append(c)
    clients = [clients[i] for i in rng.permutation(len(clients))]
    for i, c in enumerate(clients):c['client'] = f'vehicle_{i:02d}'
    return dict(seed=seed, clients=clients)
