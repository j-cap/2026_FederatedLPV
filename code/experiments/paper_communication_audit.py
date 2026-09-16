"""Recompute paper-facing communication with model-order discovery included."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from federated_lpv.communication import mixture_backbone_cost,fixed_k_cost

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables'
CONFIG=ROOT/'code/config/experiment_9j.json'


def load_protocol(seeds):
    return pd.concat([pd.read_csv(OUT/f'experiment_9j_seed{seed}_protocol.csv.gz') for seed in seeds],ignore_index=True)


def summarize_phase(phase,seeds,cfg):
    protocol=load_protocol(seeds);rows=[];n_clients=sum(cfg['train_counts'])
    for record in protocol.itertuples(index=False):
        cohort=int(np.ceil(record.participation*n_clients));discovery=mixture_backbone_cost(n_clients,cohort,cfg['federated_rounds'],cfg['candidate_clusters'],int(record.selected_k)).bytes(cfg['float_bytes']);fixed=fixed_k_cost(n_clients,cohort,cfg['federated_rounds'],int(record.selected_k)).bytes(cfg['float_bytes'])
        rows.append(dict(phase=phase,seed=record.seed,method=record.method,participation=record.participation,selected_k=record.selected_k,legacy_upload_bytes=record.upload_bytes,legacy_download_bytes=record.download_bytes,discovery_upload_bytes=discovery['upload_bytes'],discovery_download_bytes=discovery['download_bytes'],fixed_k_upload_bytes=fixed['upload_bytes'],fixed_k_download_bytes=fixed['download_bytes'],discovery_client_transmissions=discovery['client_transmissions']))
    return rows


def main():
    cfg=json.loads(CONFIG.read_text());rows=summarize_phase('development',cfg['development_seeds'],cfg)+summarize_phase('confirmation',cfg['reserved_confirmation_seeds'],cfg);raw=pd.DataFrame(rows);summary=raw.groupby(['phase','method','participation']).agg(selected_k=('selected_k','mean'),legacy_upload_bytes=('legacy_upload_bytes','mean'),legacy_download_bytes=('legacy_download_bytes','mean'),discovery_upload_bytes=('discovery_upload_bytes','mean'),discovery_download_bytes=('discovery_download_bytes','mean'),fixed_k_upload_bytes=('fixed_k_upload_bytes','mean'),fixed_k_download_bytes=('fixed_k_download_bytes','mean'),discovery_client_transmissions=('discovery_client_transmissions','mean')).reset_index();raw.to_csv(OUT/'paper_communication_audit_by_seed.csv',index=False);summary.to_csv(OUT/'paper_communication_audit.csv',index=False);conclusions={'accounting_version':'candidate-order discovery plus selected backbone','candidate_k':cfg['candidate_clusters'],'dimensions':{'initialization_scalars':10,'em_upload_scalars_per_group':16,'mixture_broadcast_scalars_per_group':10,'final_selection_scalars':'K+1','manifold_upload_scalars_per_group':10},'excludes':['transport headers','cryptographic overhead','retransmissions','new-client model download'],'provenance':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [CONFIG,ROOT/'code/src/federated_lpv/communication.py',ROOT/'code/experiments/paper_communication_audit.py']}};(OUT/'paper_communication_accounting.json').write_text(json.dumps(conclusions,indent=2)+'\n');print(summary.to_string(index=False));print(json.dumps(conclusions,indent=2))


if __name__=='__main__':main()
