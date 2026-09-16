from federated_lpv.communication import (
    initialization_scalars,em_upload_scalars_per_group,
    mixture_broadcast_scalars_per_group,manifold_upload_scalars_per_group,
    mixture_backbone_cost,fixed_k_cost,
)


def test_three_parameter_message_dimensions():
    assert initialization_scalars()==10
    assert em_upload_scalars_per_group()==16
    assert mixture_broadcast_scalars_per_group()==10
    assert manifold_upload_scalars_per_group()==10


def test_candidate_search_cost_exceeds_selected_model_cost():
    discovery=mixture_backbone_cost(180,36,30,range(1,7),3)
    fixed=fixed_k_cost(180,36,30,3)
    assert discovery.upload_scalars>fixed.upload_scalars
    assert discovery.download_scalars>fixed.download_scalars
    assert discovery.manifold_upload_scalars==fixed.manifold_upload_scalars
