import importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_unified_ablation_is_frozen_to_9k():
    path=ROOT/'code/experiments/experiment_9m_unified_ablation.py';spec=importlib.util.spec_from_file_location('experiment_9m',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.assert_frozen(json.loads(module.CONFIG.read_text()))
