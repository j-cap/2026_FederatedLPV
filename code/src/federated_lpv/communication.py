"""Scalar-level communication accounting for federated mixture/backbone learning."""
from dataclasses import dataclass,asdict


def symmetric_entries(d):
    return d*(d+1)//2


def initialization_scalars(d=3):
    """Count, first moment, and symmetric second moment."""
    return 1+d+symmetric_entries(d)


def em_upload_scalars_per_group(d=3):
    """Mass, symmetric precision, information vector, and symmetric scatter."""
    return 1+symmetric_entries(d)+d+symmetric_entries(d)


def mixture_broadcast_scalars_per_group(d=3):
    """Mixing weight, mean, and symmetric intrinsic covariance."""
    return 1+d+symmetric_entries(d)


def model_selection_scalars(k):
    """Final responsibility masses plus one local log-likelihood contribution."""
    return k+1


def manifold_upload_scalars_per_group(d=3):
    """Hard-group count, first moment, and symmetric second moment."""
    return 1+d+symmetric_entries(d)


@dataclass(frozen=True)
class CommunicationCost:
    upload_scalars: int
    download_scalars: int
    client_transmissions: int
    discovery_upload_scalars: int
    manifold_upload_scalars: int

    def bytes(self,float_bytes=4):
        values=asdict(self)
        values.update(upload_bytes=self.upload_scalars*float_bytes,
                      download_bytes=self.download_scalars*float_bytes)
        return values


def mixture_backbone_cost(n_clients,cohort_size,rounds,candidate_k,selected_k,d=3):
    """Implementation-matched cost of discovering K and aggregating a backbone.

    Every candidate order is trained independently, as in
    ``select_federated_mixture``. Initialization uses additive first/second
    moments, each EM round uploads sufficient statistics and downloads the
    mixture state, all clients contribute final BIC statistics, and all clients
    contribute selected-group moments for the final backbone.
    """
    candidate_k=tuple(candidate_k);initial=initialization_scalars(d);em=em_upload_scalars_per_group(d);broadcast=mixture_broadcast_scalars_per_group(d);manifold=manifold_upload_scalars_per_group(d)
    discovery_upload=sum(cohort_size*initial+rounds*cohort_size*em*k+n_clients*model_selection_scalars(k) for k in candidate_k)
    manifold_upload=n_clients*manifold*selected_k
    download=sum(rounds*cohort_size*broadcast*k for k in candidate_k)
    transmissions=len(candidate_k)*(rounds+1)*cohort_size+len(candidate_k)*n_clients+n_clients
    return CommunicationCost(discovery_upload+manifold_upload,download,transmissions,discovery_upload,manifold_upload)


def fixed_k_cost(n_clients,cohort_size,rounds,k,d=3):
    """Cost when a previously selected K is reused without candidate search."""
    return mixture_backbone_cost(n_clients,cohort_size,rounds,[k],k,d)
