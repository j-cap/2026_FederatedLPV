"""Data isolation, exact message algebra, numerical fitting and durable workflow."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import paper_federated_relaxation_coverage as run
from federated_lpv import federated_relaxation as fl
from federated_lpv.tire_relaxation_pilot import batch_predict


@pytest.fixture
def cfg():return run.preflight()


def synthetic_clients(cfg):
    p=np.array([50.,55.,.6,.9,.4,.7]);u=.006*np.sin(np.arange(80)*.08)
    clients=[]
    for speed in [13,21,29]:
        y=batch_predict(p,[u],[speed],[1.2,1.6],.01,.001)[0,0,:,1]
        clients.append(fl.Client([dict(speed=speed,applied_input=u,measured_yaw=y)],[1.2,1.6],run.settings(cfg)))
    return p,clients


def test_message_algebra_and_loss_gradient(cfg):
    p,clients=synthetic_clients(cfg);q=np.log(p*1.01)
    loss,g,h=fl.aggregate([c.information_message(q) for c in clients])
    parts=[c.residual(q,True) for c in clients];r=np.concatenate([v[0] for v in parts]);j=np.vstack([v[1] for v in parts])
    np.testing.assert_allclose(loss,r@r);np.testing.assert_allclose(g,j.T@r);np.testing.assert_allclose(h,j.T@j)
    for index in range(6):
        d=np.eye(6)[index]*1e-5
        numeric=(sum(c.loss_message(q+d) for c in clients)-sum(c.loss_message(q-d) for c in clients))/2e-5
        np.testing.assert_allclose(numeric,2*g[index],atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError,match='Unexpected'):fl.aggregate([dict(loss=0,gradient=[0]*6,upper_hessian=[0]*21,family='nominal')])


def test_public_prediction_cache_preserves_private_residuals(cfg):
    p,clients=synthetic_clients(cfg);q=np.log(p*1.01)
    expected=[c.information_message(q) for c in clients]
    cache=fl.PublicPredictionCache(clients[0].u[0],[13,21,29],[1.2,1.6],run.settings(cfg))
    for c,old in zip(clients,expected):
        c.cache={};c.public_predictor=cache
        msg=c.information_message(q)
        for name in msg:np.testing.assert_allclose(msg[name],old[name],rtol=1e-12,atol=1e-12)
    assert cache.simulations==1
    altered=fl.Client([dict(speed=13,applied_input=clients[0].u[0],measured_yaw=clients[0].y[0]+.001)],[1.2,1.6],run.settings(cfg))
    altered.public_predictor=cache
    assert altered.loss_message(q)!=clients[0].loss_message(q)
    assert cache.simulations==1 and not hasattr(cache,'y')


def test_global_server_uses_only_messages(cfg):
    p,clients=synthetic_clients(cfg)
    class Endpoint:
        __slots__=('information_message','loss_message')
        def __init__(self,c):self.information_message=c.information_message;self.loss_message=c.loss_message
    endpoints=[Endpoint(c) for c in clients]
    state=fl.initialize_global(np.log(p*1.01).tolist());settings=dict(cfg['sharing'],bounds=cfg['fit']['bounds'])
    first=sum(c.loss_message(state['log_parameters']) for c in clients)
    for _ in range(10):
        state=fl.global_step(endpoints,state,settings)
        if state['status']!='running':break
    assert state['objective']<first*1e-5
    assert np.all(np.diff([v['objective'] for v in state['history']])<=1e-10)
    assert state['communication_float64']>0


def test_local_fit_and_fixed_prior(cfg):
    p,clients=synthetic_clients(cfg);q=np.log(p*1.01)
    a=clients[0].fit(q);b=clients[0].fit(q,np.log(cfg['nominal_parameters']),cfg['sharing']['weights'])
    assert a['available'] and a['optimizer_converged'] and a['data_sse']<1e-7
    assert b['available'] and b['objective']<=b['initial_objective']
    w=np.array(cfg['sharing']['weights']);z=np.array(b['log_parameters']);eta=np.log(cfg['nominal_parameters'])
    np.testing.assert_allclose(b['prior_penalty'],np.sum((w*(z-eta))**2))
    eta[3]+=10
    np.testing.assert_allclose(b['prior_penalty'],np.sum((w*(z-eta))**2))


def test_personalized_round_monotonic_and_label_free(cfg):
    p,clients=synthetic_clients(cfg);logs=[np.log(p*np.array([1+.01*i,1,.99,.9,1,1])).tolist() for i in range(3)]
    weights=cfg['sharing']['weights'];state=fl.initialize_personal(logs,1,0,weights)
    for _ in range(2):
        fits=[c.fit(q,state['centers'][g],weights,max_nfev=15) for c,q,g in zip(clients,state['log_parameters'],state['assignments'])]
        state=fl.finish_personal_round(state,[fl.client_round_message(f) for f in fits],weights,cfg['sharing'],all(f['optimizer_converged'] for f in fits))
    assert state['history'][1]['objective']<=state['history'][0]['objective']+1e-7
    assert state['assignments']==[0,0,0]
    assert state['communication_float64']==3*6+2*3*13


def test_hidden_measurements_rejected(cfg):
    record=dict(speed=21,applied_input=np.zeros(20),measured_yaw=np.zeros(21),true_parameters=np.ones(6))
    with pytest.raises(ValueError,match='accepts yaw'):fl.Client([record],[1.2,1.6],run.settings(cfg))


def test_coverage_and_balanced_assignment(cfg):
    assert fl.coverage_region(13,13,[13,21,29])=='local_observed'
    assert fl.coverage_region(29,13,[13,21,29])=='peer_observed'
    assert fl.coverage_region(17,13,[13,21,29])=='fleet_interpolation'
    assert fl.coverage_region(33,13,[13,21,29])=='fleet_extrapolation'
    for fleet in cfg['fleets']:
        for family in ['nominal','heavy','handling']:
            assert sorted(c['training_speed'] for c in fleet['clients'] if c['family']==family)==[13,21,29]
    assert all(not any('previous' in k for k in c) for f in cfg['fleets'] for c in f['clients'])


def test_selected_attempt_respects_status_and_training_loss():
    attempts={'0':dict(status='budget_exhausted',objective=2.,log_parameters=[0]*6),
              '1':dict(status='converged',objective=3.,log_parameters=[1]*6),
              '2':dict(status='local_update_failed',objective=0.)}
    result=run.choose_sharing(attempts)
    assert result['selected_attempt']=='0' and result['optimization_status']=='budget_exhausted'
    assert result['objective']==2.


def test_restart_restore_and_html(tmp_path,monkeypatch,cfg):
    """Tiny synthetic orchestration with stub fits; real acquisition/evaluation."""
    cfg=copy.deepcopy(cfg);cfg['duration_s']=.15;cfg['evaluation_speeds']=[13,21]
    cfg['starts']=cfg['starts'][:1];cfg['sharing']['group_initialization_seeds']=[17]
    cfg['sharing']['personal_max_rounds']=1;cfg['sharing']['global_max_steps']=1
    monkeypatch.setattr(run,'preflight',lambda:cfg)
    monkeypatch.setattr(run,'OUT',tmp_path/'tables');monkeypatch.setattr(run,'RAW',tmp_path/'raw');monkeypatch.setattr(run,'REPORT',tmp_path/'report.html')
    counts=[]
    def fit(self,start,center=None,weights=None,max_nfev=None):
        counts.append(1);q=np.asarray(start);prior=0. if center is None else float(np.sum((np.array(weights)*(q-center))**2))
        return dict(available=True,optimizer_converged=True,log_parameters=q.tolist(),parameters=np.exp(q).tolist(),
            objective=1.+prior,data_sse=1.,prior_penalty=prior,initial_objective=1.+prior,nfev=1,optimality=0.,near_bound=[False]*6)
    monkeypatch.setattr(fl.Client,'fit',fit)
    def global_step(clients,state,settings):
        return dict(state,status='budget_exhausted',objective=9.,iteration=1,history=[dict(objective=9.)],communication_float64=34*len(clients))
    monkeypatch.setattr(fl,'global_step',global_step)
    for f in cfg['fleets']:run.worker(f['seed'])
    previous=len(counts);p=run.storage.read_checkpoint(run.path(1301));p['status']='running'
    p['evaluation'].pop(next(iter(p['evaluation'])));run.storage.atomic_json(run.path(1301),p)
    run.worker(1301);assert len(counts)==previous
    p=run.storage.read_checkpoint(run.path(1301));run.validate(p,cfg,1301)
    damaged=copy.deepcopy(p);damaged['evaluation'].pop(next(iter(damaged['evaluation'])))
    with pytest.raises(AssertionError):run.validate(damaged,cfg,1301)
    dest=run.RAW/'1301'/'vehicle_00_training.npz';dest.unlink();run.restore(cfg,[1301]);assert dest.exists()
    run.analyze(cfg);html=run.REPORT.read_text(encoding='utf-8')
    assert html.count('<svg')==1 and '\ufffd' not in html
