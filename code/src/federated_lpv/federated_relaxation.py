"""Data-local nonlinear OE and federated physical-parameter sharing.

The server receives numerical messages only. Client endpoints own the yaw/input
records. No true parameters, family labels or evaluation records enter learning.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from .tire_relaxation_pilot import observed_records, response_jacobian, batch_predict


class PublicPredictionCache:
    """Simulation accelerator for the common, prescribed public steering probe.

    Caches model predictions/derivatives only, never observations or residuals.
    Only enabled during shared-parameter Global optimization. It is equivalent
    to clients independently simulating the same broadcast model/input/speed.
    """
    def __init__(self, public_input, speeds, geometry, settings):
        self.u = np.asarray(public_input)
        self.speeds = list(speeds)
        self.geometry = geometry
        self.settings = settings
        self.cache = OrderedDict()
        self.simulations = 0

    def predict(self, q, speed):
        key = np.asarray(q,dtype=float).tobytes()
        if key not in self.cache:
            inputs = np.array([self.u]*len(self.speeds))
            y,j = response_jacobian(np.asarray(q),np.exp(q),list(range(6)),inputs,
                self.speeds,self.geometry,self.settings)
            self.cache[key]=(y,j.reshape(len(self.speeds),len(self.u)+1,6))
            self.simulations += 1
            if len(self.cache)>8: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        y,j=self.cache[key];index=self.speeds.index(float(speed))
        return y[index:index+1],j[index]


class Client:
    def __init__(self, records, geometry, settings):
        self.u, self.y, self.speeds = observed_records(records)
        self.geometry = geometry
        self.settings = settings
        self.sd = np.deg2rad(settings['yaw_noise_std_deg_s'])
        self.cache = {}
        self.public_predictor = None
        self.prediction_calls = 0
        self.jacobian_calls = 0

    def residual(self, q, jacobian=False):
        q = np.asarray(q, float)
        if 'q' not in self.cache or not np.array_equal(q, self.cache['q']):
            self.cache = dict(q=q.copy())
        if self.public_predictor is not None and 'j' not in self.cache:
            if (len(self.speeds)!=1 or not np.array_equal(self.u[0],self.public_predictor.u)
                    or not np.array_equal(self.geometry,self.public_predictor.geometry)):
                raise ValueError('Public cache requires the identical prescribed input')
            y,j=self.public_predictor.predict(q,float(self.speeds[0]))
            self.cache.update(r=(y-self.y).ravel()/self.sd,j=j/self.sd)
        if jacobian and 'j' not in self.cache:
            y, j = response_jacobian(q, np.exp(q), list(range(6)), self.u, self.speeds,
                                     self.geometry, self.settings)
            self.cache.update(r=(y-self.y).ravel()/self.sd, j=j/self.sd)
            self.jacobian_calls += 1
        elif 'r' not in self.cache:
            y = batch_predict(np.exp(q), self.u, self.speeds, self.geometry,
                self.settings['sample_time'], self.settings['integration_max_step'])[0, :, :, 1]
            self.cache['r'] = (y-self.y).ravel()/self.sd
            self.prediction_calls += 1
        return (self.cache['r'], self.cache['j']) if jacobian else self.cache['r']

    def loss_message(self, q):
        r = self.residual(q)
        return float(r@r)

    def information_message(self, q):
        r, j = self.residual(q, True)
        h = j.T@j
        return dict(loss=float(r@r), gradient=(j.T@r).tolist(),
                    upper_hessian=h[np.triu_indices(6)].tolist())

    def fit(self, start, center=None, weights=None, max_nfev=None):
        """Exact nonlinear residual, optionally with a quadratic log prior."""
        cfg = self.settings; bounds = np.log(np.asarray(cfg['bounds']).T)
        start = np.asarray(start, float)
        w = np.zeros(6) if weights is None else np.asarray(weights, float)
        eta = np.zeros(6) if center is None else np.asarray(center, float)
        def evaluate(q, jacobian):
            prior = w*(q-eta)
            if jacobian:
                _, j = self.residual(q, True)
                return np.vstack([j, np.diag(w)])
            r, _ = self.residual(q, True)
            return np.r_[r, prior]
        try:
            initial_loss = float(np.sum(evaluate(start, False)**2))
            fit = least_squares(lambda q:evaluate(q, False), start,
                jac=lambda q:evaluate(q, True), bounds=bounds, x_scale='jac',
                max_nfev=max_nfev or cfg['max_nfev'], ftol=cfg['ftol'],
                xtol=cfg['xtol'], gtol=cfg['gtol'])
            q = fit.x; loss = float(fit.fun@fit.fun)
            r, j = self.residual(q, True)
            s = np.linalg.svd(j, compute_uv=False)
            return dict(available=bool(np.isfinite(loss) and loss <= initial_loss+1e-7*max(1.,initial_loss)),
                optimizer_converged=bool(fit.success), status=int(fit.status), message=str(fit.message),
                log_parameters=q.tolist(), parameters=np.exp(q).tolist(),
                objective=loss, data_sse=float(r@r), prior_penalty=float(np.sum((w*(q-eta))**2)),
                initial_objective=initial_loss, nfev=int(fit.nfev), optimality=float(fit.optimality),
                near_bound=(np.minimum(q-bounds[0], bounds[1]-q)<1e-3).tolist(),
                data_singular_values=s.tolist(), data_relative_rank=int(np.sum(s>s[0]*1e-8)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return dict(available=False, optimizer_converged=False, reason=str(exc))


def select_local(attempts):
    valid = [(a['objective'],i) for i,a in enumerate(attempts)
             if a['available'] and a['optimizer_converged']]
    if not valid:
        return dict(available=False, reason='No converged local start')
    _, i = min(valid)
    return dict(attempts[i], selected_start=i)


def aggregate(messages):
    """Reconstruct summed Gauss–Newton information from 28 floats/client."""
    h = np.zeros((6,6)); g = np.zeros(6); loss = 0.
    for msg in messages:
        if set(msg) != {'loss','gradient','upper_hessian'}:
            raise ValueError('Unexpected server message fields')
        upper = np.zeros((6,6)); upper[np.triu_indices(6)] = msg['upper_hessian']
        h += upper+upper.T-np.diag(np.diag(upper))
        g += msg['gradient']; loss += msg['loss']
    return loss, g, h


def initialize_global(start):
    return dict(log_parameters=list(start), iteration=0, damping=1e-4, history=[],
                status='running', communication_float64=0)


def global_step(clients, state, settings):
    """One bounded damped GN step, using only loss/gradient/information messages."""
    q = np.asarray(state['log_parameters']); lower,upper = np.log(np.asarray(settings['bounds']).T)
    loss,g,h = aggregate([c.information_message(q) for c in clients])
    count = state['communication_float64']+len(clients)*(6+28)
    scale = np.sqrt(np.maximum(np.diag(h), 1e-12))
    active = ((q <= lower+1e-9)&(g>0))|((q >= upper-1e-9)&(g<0))
    projected = np.where(active,0.,g)
    stationarity = float(np.max(abs(projected)/scale))
    hs = h/scale[:,None]/scale[None,:]; gs = g/scale
    hs[active,:] = 0; hs[:,active] = 0; gs[active] = 0
    damp = state['damping']
    step = np.linalg.solve(hs+damp*np.eye(6),-gs)/scale
    step *= min(1.,settings['global_max_log_step']/max(np.max(abs(step)),1e-30))
    accepted = False; trial_loss = loss; trial = q.copy(); alpha = 0.
    for alpha in settings['global_line_search']:
        candidate = np.clip(q+alpha*step,lower,upper)
        candidate_loss = sum(c.loss_message(candidate) for c in clients)
        count += len(clients)*7
        if candidate_loss < loss:
            accepted=True;trial=candidate;trial_loss=candidate_loss;break
    movement = float(np.max(abs(trial-q)))
    relative = (loss-trial_loss)/max(1.,loss)
    converged = stationarity <= settings['global_gradient_tolerance'] or (
        accepted and relative <= settings['global_relative_tolerance'] and movement <= settings['global_step_tolerance'])
    iteration = state['iteration']+1
    status = 'converged' if converged else ('budget_exhausted' if iteration >= settings['global_max_steps'] else 'running')
    history = state['history']+[dict(iteration=iteration,objective=trial_loss,accepted=accepted,
        projected_scaled_gradient=stationarity,max_log_step=movement,damping=damp)]
    return dict(log_parameters=trial.tolist(),objective=trial_loss,iteration=iteration,
        damping=max(damp/3,1e-8) if accepted else min(damp*10,1e12),history=history,
        status=status,communication_float64=count)


def initialize_personal(local_logs, groups, seed, weights):
    z = np.asarray(local_logs); w = np.asarray(weights)
    if groups == 1:
        labels = np.zeros(len(z),int)
    else:
        labels = KMeans(n_clusters=groups, n_init=1, random_state=seed).fit_predict(z*w)
    centers = np.array([z[labels==g].mean(axis=0) if np.any(labels==g) else z[g % len(z)] for g in range(groups)])
    return dict(log_parameters=z.tolist(),centers=centers.tolist(),assignments=labels.tolist(),
        round=0,history=[],status='running',pending={},communication_float64=len(z)*6)


def client_round_message(fit):
    """Seven float64 values; no trajectory, family label or simulator parameter."""
    if not fit['available']: raise ValueError('Unavailable local update')
    return dict(log_parameters=fit['log_parameters'], data_sse=fit['data_sse'])


def finish_personal_round(state, messages, weights, settings, local_converged):
    if any(set(m)!={'log_parameters','data_sse'} for m in messages):
        raise ValueError('Unexpected personalized server message fields')
    z = np.array([m['log_parameters'] for m in messages]); centers=np.array(state['centers']);w=np.array(weights)
    distance = np.sum(((z[:,None,:]-centers[None,:,:])*w)**2,axis=2)
    labels=np.argmin(distance,axis=1)
    for g in range(len(centers)):
        if np.any(labels==g): centers[g]=z[labels==g].mean(axis=0)
    penalty=float(np.sum(((z-centers[labels])*w)**2));data=float(sum(m['data_sse'] for m in messages));obj=data+penalty
    history=state['history'];old=history[-1]['objective'] if history else None
    if old is not None and obj > old+1e-6*max(1.,old):
        raise ValueError('Alternating objective increased; checkpoint preserved')
    movement=float(np.max(abs(z-np.array(state['log_parameters']))))
    same=labels.tolist()==state['assignments']
    stable=old is not None and abs(old-obj)/max(1.,abs(old))<settings['personal_relative_tolerance']
    converged=stable and same and movement<settings['personal_step_tolerance'] and local_converged
    round_=state['round']+1
    status='converged' if converged else ('budget_exhausted' if round_>=settings['personal_max_rounds'] else 'running')
    record=dict(round=round_,objective=obj,data_sse=data,penalty=penalty,max_log_step=movement,
        assignments_changed=not same,all_local_optimizers_converged=bool(local_converged),active_groups=len(set(labels.tolist())))
    return dict(log_parameters=z.tolist(),centers=centers.tolist(),assignments=labels.tolist(),
        round=round_,history=history+[record],status=status,pending={},
        communication_float64=state['communication_float64']+len(z)*13)


def coverage_region(speed, local_speed, fleet_speeds):
    if speed == local_speed: return 'local_observed'
    if speed in fleet_speeds: return 'peer_observed'
    if min(fleet_speeds) < speed < max(fleet_speeds): return 'fleet_interpolation'
    return 'fleet_extrapolation'
