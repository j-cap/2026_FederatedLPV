"""Read-only verification of the frozen Smooth Local LQI supplement."""
from pathlib import Path
import json
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/experiments')]
import paper_smooth_local_lqi as run
from federated_lpv.local_scheduling import curvature_factor
from experiment_10a_basis_coverage_audit import raw_basis
from experiment_10b_oracle_complementary_gate import predict


def main():
    cfg, ctl, basis, lock=run.preflight()
    manifest=run.load(run.OUT/f'{run.PREFIX}_manifest.json')
    assert manifest['source_sha256']==lock
    for name, sha in {**manifest['checkpoints'], **manifest['outputs']}.items():
        assert run.digest(run.OUT/name)==sha, name
    frames=[]; diagnostics=[]; max_normal=0.; max_parent=0.; max_schedule=0.
    for seed in cfg['seeds']:
        saved=run.read_checkpoint(seed)
        frame=pd.DataFrame(saved['rows'])
        assert len(frame)==360 and not frame.duplicated(['client','method','scenario']).any()
        original=frame[frame.method!='SmoothLocal']
        expected=pd.read_csv(run.OUT/f'experiment_10f_confirmation_seed{seed}.csv.gz')
        run.check_parent(original,expected,cfg['parent_tolerance'])
        a=original.set_index(['client','method','scenario']).sort_index()
        b=expected.set_index(['client','method','scenario']).sort_index()
        max_parent=max(max_parent,float(np.max(np.abs(a.tracking_rmse_deg_s-b.tracking_rmse_deg_s))))
        inputs={x['client']:x for x in saved['inputs']}
        for item in saved['clients']:
            inp=inputs[item['client']]
            phi=raw_basis(np.asarray(inp['speeds']),7,basis)
            target=np.asarray(inp['targets']); theta=np.asarray(item['model']); q=curvature_factor()
            # Normal-equation residual, scaled for a high-penalty augmented solve.
            gram=phi.T@phi/len(phi)+1e6*q.T@q+1e-16*np.eye(7)
            rhs=phi.T@target/len(phi)
            residual=np.linalg.norm(gram@theta-rhs)/(np.linalg.norm(gram)*np.linalg.norm(theta)+np.linalg.norm(rhs))
            assert residual<1e-10
            max_normal=max(max_normal,float(residual))
            schedule=run.parent.design_schedule(theta,np.asarray(ctl['speed_grid']),saved['weights'],basis)
            if item['schedule'] is None:
                assert schedule is None
                continue
            for actual, frozen in zip(schedule,item['schedule']):
                delta=float(np.max(np.abs(actual-np.asarray(frozen))/(1+np.abs(frozen))))
                max_schedule=max(max_schedule,delta)
                assert delta<1e-5, delta
            schedule=tuple(np.asarray(a) for a in item["schedule"])
            rho=run.parent.stability_radius(schedule,np.asarray(item['exact_model']),ctl,basis)
            rows=frame[(frame.client==item['client'])&(frame.method=='SmoothLocal')]
            np.testing.assert_allclose(rows.rho,rho,rtol=1e-9,atol=1e-9)
            prediction=predict(theta,ctl['speed_grid'],basis)
            exact=predict(np.asarray(item['exact_model']),ctl['speed_grid'],basis)
            diagnostics.append(dict(seed=seed,client=item['client'],rho=rho,
                full_matrix_error=float(np.mean(np.linalg.norm(prediction-exact,axis=1)/np.linalg.norm(exact,axis=1))),
                b2_sign_changes=int(np.sum(np.sign(prediction[:,5])!=np.sign(exact[:,5]))),
                infeasible_maneuvers=int((~rows.feasible).sum())))
        frames.append(frame)
    raw=pd.concat(frames,ignore_index=True)
    saved_units=pd.read_csv(run.OUT/f'{run.PREFIX}_seed_summary.csv')
    actual=raw.groupby(['seed','method','scenario']).tracking_rmse_deg_s.mean().sort_index()
    expected=saved_units.set_index(['seed','method','scenario']).tracking.sort_index()
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    # Recreate summaries in temporary storage without overwriting executed outputs.
    original_out=run.OUT
    with tempfile.TemporaryDirectory() as temp:
        for name in ["paper_local_smoothness_selection.json"] + [f"{run.PREFIX}_seed{s}.json" for s in cfg["seeds"]]:
            shutil.copyfile(original_out/name,Path(temp)/name)
        run.OUT=Path(temp)
        original_read=run.read_checkpoint
        run.read_checkpoint=lambda seed:run.load(original_out/f'{run.PREFIX}_seed{seed}.json')
        try:
            run.summarize()
            for name in ('clients','seed_summary','comparisons'):
                pd.testing.assert_frame_equal(pd.read_csv(run.OUT/f'{run.PREFIX}_{name}.csv'),
                                              pd.read_csv(original_out/f'{run.PREFIX}_{name}.csv'))
        finally:
            run.OUT=original_out; run.read_checkpoint=original_read
    # Portability check: preserve saved outcomes and report replay differences.
    saved=run.read_checkpoint(411)
    clients={c.client_id:c for c in run.sample_fleet(411)}
    replay=[]
    for item in saved['clients']:
        if item['schedule'] is None:
            continue
        schedule=tuple(np.asarray(a) for a in item['schedule'])
        for rec in item['records']:
            metrics,arrays=run.simulate_with_arrays(clients[item['client']],schedule,
                run.parent.scenarios(ctl)[rec['scenario']],ctl)
            frozen=next(v for v in saved['rows'] if v['client']==item['client']
                and v['method']=='SmoothLocal' and v['scenario']==rec['scenario'])
            replay.append(dict(client=item['client'],scenario=rec['scenario'],
                exact_hash_match=run.array_hashes(arrays)==rec['hashes'],
                rmse_difference=metrics['tracking_rmse_deg_s']-frozen['tracking_rmse_deg_s'],
                feasibility_match=metrics['feasible']==frozen['feasible']))
    replay=pd.DataFrame(replay)
    replay.to_csv(original_out/f'{run.PREFIX}_replay_audit.csv',index=False)
    smooth=raw[raw.method=='SmoothLocal']
    d=pd.DataFrame(diagnostics)
    payload=dict(status='saved_maps_and_summaries_verified',fleets=10,rows=len(raw),smooth_rollouts=len(smooth),
        parent_rmse_max_difference=max_parent,normalized_stationarity_max=max_normal,normalized_schedule_difference_max=max_schedule,
        unstable_frozen_point_clients=int((d.rho>=1).sum()),rho_max=float(d.rho.max()),
        b2_sign_changed_clients=int((d.b2_sign_changes>0).sum()),
        replay_seed411=dict(rollouts=len(replay),exact_hash_matches=int(replay.exact_hash_match.sum()),
            max_absolute_rmse_difference=float(replay.rmse_difference.abs().max()),
            feasibility_matches=int(replay.feasibility_match.sum())),
        scenarios={name:dict(rmse=float(g.tracking_rmse_deg_s.mean()),infeasible=int((~g.feasible).sum()),
             beta_peak=float(g.beta_peak_deg.max()),mean_saturation=float(g.saturation_fraction.mean()),
             synthesis_or_rollout_failures=int((g.status!='complete').sum())) for name,g in smooth.groupby('scenario')},
        source_sha256=lock,execution_manifest_sha256=run.digest(original_out/f'{run.PREFIX}_manifest.json'))
    d.to_csv(original_out/f'{run.PREFIX}_map_audit.csv',index=False)
    (original_out/f'{run.PREFIX}_audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
