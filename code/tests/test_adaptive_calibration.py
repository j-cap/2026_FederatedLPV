import json
from experiment_9d_adaptive_calibration import CONFIG,calibration_seconds,speed_plan


def test_speed_plan_starts_local_then_adds_complementary_speeds():
    cfg=json.loads(CONFIG.read_text());plan=speed_plan(0,cfg)
    assert plan[0]==cfg['coverage_speeds'][0]
    assert len(plan)==cfg['maximum_records']-cfg['initial_records']+1
    assert len(set(plan))==len(plan)


def test_adaptive_budget_counts_short_and_long_records():
    cfg=json.loads(CONFIG.read_text())
    assert calibration_seconds(3,cfg)==0.75
    assert calibration_seconds(4,cfg)==1.75
    assert calibration_seconds(5,cfg)==2.75
