"""Export scenario arrays on the execution machine without fitting or rollouts."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import paper_smooth_local_lqi as frozen
from paper_smooth_replay_diagnosis import metadata


def main():
    frozen.preflight()
    cfg=frozen.preflight()[1]
    scenarios=frozen.parent.scenarios(cfg)
    matches={}
    for seed in range(411,421):
        saved=frozen.read_checkpoint(seed)
        matches[str(seed)]={rec['scenario']:{k: frozen.array_hashes(dict(speed=scenarios[rec['scenario']]['speed'],reference=scenarios[rec['scenario']]['speed']*scenarios[rec['scenario']]['curvature']))[k]==rec['hashes'][k] for k in ['speed','reference']} for rec in saved['clients'][0]['records']}
    payload=dict(environment=metadata(),source_sha256=frozen.preflight()[3],
        original_input_hash_matches=matches,
        scenarios={k:{n:np.asarray(v).tolist() for n,v in sc.items()} for k,sc in scenarios.items()},
        plant_parameters={str(seed):{c.client_id:asdict(c.parameters) for c in frozen.sample_fleet(seed)} for seed in range(411,421)})
    path=frozen.OUT/'paper_smooth_replay_execution_reference.json'
    if path.exists():raise FileExistsError('Preserve the existing execution-reference export; do not overwrite it')
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path),all_original_input_hashes_match=all(v for a in matches.values() for b in a.values() for v in b.values()))))


if __name__=='__main__':main()
