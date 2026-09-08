import numpy as np
from experiment_9a_learned_groups import fit_partition,cluster_models


def config():
    return {'candidate_clusters':[2,3,4],'minimum_silhouette':.25,'minimum_cluster_size':5,'kmeans_restarts':10}


def test_partition_recovers_separated_groups_and_unknown_k():
    rng=np.random.default_rng(1);z=np.vstack([rng.normal(-.7,.03,(20,3)),rng.normal(0,.03,(20,3)),rng.normal(.7,.03,(20,3))])
    labels,centers,k,records=fit_partition(z,config(),seed=2)
    assert k==3 and len(np.unique(labels))==3 and centers.shape==(3,3)
    assert len(records)==3


def test_partition_falls_back_to_one_group_without_admissible_split():
    cfg=config();cfg['minimum_cluster_size']=40
    labels,centers,k,_=fit_partition(np.random.default_rng(3).normal(0,.01,(20,3)),cfg,seed=3)
    assert k==1 and np.all(labels==0) and centers.shape==(1,3)


def test_cluster_models_returns_positive_physical_ratios():
    z=np.array([[-.5,0,.5],[-.3,.1,.4],[.5,0,-.5],[.3,-.1,-.4]])
    models=cluster_models(z,np.array([0,0,1,1]),2,np.array([[40,70],[35,75],[.45,.8]]))
    assert set(models)=={0,1} and all(np.all(v>0) for v in models.values())
