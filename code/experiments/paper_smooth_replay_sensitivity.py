"""One-client numerical sensitivity probe, not reconstruction of old inputs."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import paper_smooth_local_lqi as frozen


def main():
    cfg=frozen.preflight()[1];saved=frozen.read_checkpoint(411)
    item=next(a for a in saved['clients'] if a['client']=='heavy_01')
    client=next(c for c in frozen.sample_fleet(411) if c.client_id==item['client'])
    schedule=tuple(np.asarray(v) for v in item['schedule'])
    sc=frozen.parent.scenarios(cfg)['moderate'];rows=[]
    with threadpool_limits(limits=1):
        for mode in ['native','repeat','speed_up_ulp','speed_down_ulp']:
            a={k:v.copy() for k,v in sc.items()}
            if mode.startswith('speed_'):
                a['speed']=np.nextafter(a['speed'],np.inf if mode=='speed_up_ulp' else -np.inf)
            metrics,arrays=frozen.simulate_with_arrays(client,schedule,a,cfg)
            rows.append(dict(mode=mode,rmse=metrics['tracking_rmse_deg_s'],feasible=metrics['feasible'],
                max_speed_difference=float(np.max(np.abs(a['speed']-sc['speed']))),hashes=frozen.array_hashes(arrays)))
    assert rows[0]['hashes']==rows[1]['hashes']
    payload=dict(seed=411,client='heavy_01',scenario='moderate',
        scope='representative sensitivity test; not recovery of original arrays',
        source_sha256=frozen.preflight()[3],probe_sha256=frozen.digest(Path(__file__)),rows=rows)
    (frozen.OUT/'paper_smooth_replay_ulp_probe.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('Identical-input repeat matches; ULP sensitivity probe saved')


if __name__=='__main__':main()
