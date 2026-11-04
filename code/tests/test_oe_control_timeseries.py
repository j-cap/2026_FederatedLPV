"""Selection, replay guards, and figure-layout tests using synthetic fixtures."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
import prepare_oe_control_timeseries as plot


def fixture():
    cfg = plot.run.load(plot.run.CONFIG)
    spec = plot.run.load(plot.CONFIG)
    t = np.linspace(0, 12, 1201)
    profile = dict(time=t, speed=20+8*np.sin(t/3), reference=.3*np.sin(t))
    traces = {}
    for family, scale in [('heavy', 1.), ('handling', .8)]:
        traces[family] = {}
        for j, method in enumerate(spec['methods']):
            error = scale * .008 * np.sin(t*2+j*.1)
            traces[family][method] = dict(**profile, state=np.c_[.01*np.sin(t), profile['reference']+error],
                                         steering=.06*np.sin(t[:-1]+j*.1))
    return cfg, spec, profile, traces


def test_selection_is_scale_invariant_and_ties_are_deterministic():
    values = np.array([[1., 5., 2.], [2., 5., 4.], [3., 5., 6.]])
    got = plot.rank_typical(values, [11, 12, 13])
    assert got['selected_id'] == 12
    assert got['active_components'] == [True, False, True]
    scaled = plot.rank_typical(values*np.array([1e3, 1., .1]), [11, 12, 13])
    np.testing.assert_allclose([r['distance'] for r in got['candidates']],
                               [r['distance'] for r in scaled['candidates']])
    assert plot.rank_typical([[3, 3], [3, 3]], [9, 4])['selected_id'] == 4
    with pytest.raises(ValueError, match='Nonfinite'):
        plot.rank_typical([[1., np.nan]], [1])


def test_selection_uses_both_scenarios_and_does_not_filter_for_wins():
    spec = dict(methods=['NominalLTI', 'Global', 'Federated'], selection_scenarios=['moderate', 'hard'],
                families=['heavy', 'handling'])
    payloads = []
    for seed, offset in [(1, .1), (2, .2), (3, .3)]:
        clients = []
        for index in range(30):
            family = ['nominal', 'heavy', 'handling'][index//10]
            rows = []
            for scenario in spec['selection_scenarios']:
                for j, method in enumerate(spec['methods']):
                    # Federated deliberately worse everywhere: selection must still succeed.
                    metric = offset + (index % 10)*.01 + j*.1 + (scenario == 'hard')*.5
                    rows.append(dict(method=method, scenario=scenario,
                                     outcome=dict(status='complete', metrics={'tracking_rmse_deg_s': metric})))
            clients.append(dict(index=index, client=f'client{index:02d}', family=family,
                                observed_speeds=[10, 12, 14], rows=rows))
        payloads.append(dict(seed=seed, clients=clients))
    chosen = plot.select_cases(payloads, spec)
    assert chosen['fleet']['selected_id'] == 2
    assert len(chosen['component_order']) == 6
    assert [c['family'] for c in chosen['cases']] == ['heavy', 'handling']
    assert plot.select_cases(list(reversed(payloads)), spec) == chosen
    broken = deepcopy(payloads)
    broken[0]['clients'][0]['rows'].pop()
    with pytest.raises(ValueError, match='Missing or duplicate'):
        plot.select_cases(broken, spec)


def test_replay_rejects_metric_and_raw_array_drift():
    arrays = {'time': np.arange(3.), 'state': np.zeros((3, 2))}
    outcome = {'status': 'complete', 'metrics': {'rmse': 0.}}
    saved = dict(outcome=deepcopy(outcome), arrays=plot.run.array_hashes(arrays))
    plot.verify_replay(outcome, arrays, saved, 'fixture')
    changed = deepcopy(outcome); changed['metrics']['rmse'] = 1e-14
    with pytest.raises(ValueError, match='metrics/status mismatch'):
        plot.verify_replay(changed, arrays, saved, 'fixture')
    arrays['state'][1, 0] = 1e-14
    with pytest.raises(ValueError, match='array hash mismatch'):
        plot.verify_replay(outcome, arrays, saved, 'fixture')


def test_layout_has_shared_axes_unclipped_labels_and_separate_panels(tmp_path):
    cfg, spec, profile, traces = fixture()
    panels = plot.make_panels(profile, traces, cfg, spec)
    assert len(panels) == 6
    np.testing.assert_array_equal(panels['reference'][1].get_yticks(), [-40, -20, 0, 20])
    for signal in ['error', 'steering']:
        assert panels['heavy_'+signal][1].get_ylim() == panels['handling_'+signal][1].get_ylim()
    for fig, ax in panels.values():
        assert ax.get_title() == ''
        assert ax.get_legend() is None
        assert fig.get_figwidth() == 3.48
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer), ax.yaxis.label.get_window_extent(renderer),
                 ax.xaxis.label.get_window_extent(renderer)]
        for bbox in boxes:
            assert bbox.x0 >= -1 and bbox.y0 >= -1
            assert bbox.x1 <= fig.bbox.width+1 and bbox.y1 <= fig.bbox.height+1
        plot.plt.close(fig)
    legend = plot.make_legend(spec)
    legend.canvas.draw()
    assert [t.get_text() for t in legend.legends[0].get_texts()] == [plot.LABELS[m] for m in spec['methods']]
    bounds = legend.legends[0].get_window_extent(legend.canvas.get_renderer())
    assert 0 <= bounds.x0 < bounds.x1 <= legend.bbox.width
    assert 0 <= bounds.y0 < bounds.y1 <= legend.bbox.height
    plot.plt.close(legend)
    selection = dict(cases=[dict(seed=0, client=f'{f}_fixture', family=f) for f in spec['families']])
    outputs = plot.render(profile, traces, cfg, spec, selection, [], tmp_path, tmp_path/'preview.html')
    assert len([p for p in outputs if p.suffix == '.pdf']) == 7
    assert (tmp_path/f'{plot.PREFIX}_legend.pdf').is_file()
    assert (tmp_path/'preview.html').read_text(encoding='utf-8').count('<svg') == 7
    assert 'Heavy recipient' in (tmp_path/'preview.html').read_text(encoding='utf-8')
    assert 'Handling recipient' in (tmp_path/'preview.html').read_text(encoding='utf-8')
    csv_path = tmp_path/'traces.csv'
    plot.write_traces(csv_path, traces, selection, spec)
    import csv
    with csv_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6*1201
    assert sum(r['steering_deg'] == '' for r in rows) == 6
    assert all(r['steering_deg'] == '' for r in rows if r['time_s'] == '12.0')


def test_replay_uses_saved_schedules_without_synthesis(monkeypatch):
    cfg = plot.run.load(plot.run.CONFIG)
    spec = plot.run.load(plot.CONFIG)
    frozen = plot.run.load(ROOT/'code/config/paper_fixed_lti_identification.json')['controller_frozen']
    schedule = dict(status='designed', nodes=[dict(gain=frozen['gain'], prefilter=frozen['prefilter'])
                                            for _ in cfg['speed_grid']])
    t = np.arange(21)*cfg['sample_time']
    profile = dict(time=t, speed=np.full(len(t),20.), reference=.01*np.sin(3*t))
    parameters = dict(mass=1500., yaw_inertia=2500., front_length=1.2, rear_length=1.6,
                      front_stiffness=80000., rear_stiffness=80000.)
    with plot.threadpool_limits(limits=1):
        outcome, arrays = plot.run.simulate(plot.run.VehicleParameters(**parameters), schedule, profile, cfg)
    rows = [dict(method=m, scenario='hard', outcome=outcome, arrays=plot.run.array_hashes(arrays))
            for m in spec['methods']]
    clients = [dict(index=i, client=f'{f}_fixture', family=f, parameters=parameters,
                    schedules={m:schedule for m in spec['methods']}, rows=rows)
               for i,f in enumerate(spec['families'])]
    selection = dict(cases=[dict(seed=0,index=c['index'],client=c['client'],family=c['family']) for c in clients])
    monkeypatch.setattr(plot.run,'profiles',lambda config: {'hard':profile})
    def forbidden(*args,**kwargs):
        raise AssertionError('Controller synthesis must not occur during replay')
    monkeypatch.setattr(plot.run,'design_nodes',forbidden)
    result, traces, checks = plot.replay(spec,cfg,[dict(seed=0,clients=clients)],selection)
    assert len(checks)==6 and all(c['exact_match'] for c in checks)
    assert set(traces)==set(spec['families'])
