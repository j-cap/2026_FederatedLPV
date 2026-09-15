import importlib.util
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "code/experiments/experiment_9k_blind_confirmation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_9k", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_configuration_is_frozen_against_9j():
    module = load_module()
    config = json.loads(module.CONFIG.read_text())
    module.assert_frozen(config)


def test_confirmation_seeds_are_disjoint_from_development_seeds():
    reference = json.loads((ROOT / "code/config/experiment_9j.json").read_text())
    confirmation = json.loads((ROOT / "code/config/experiment_9k.json").read_text())
    assert set(reference["development_seeds"]).isdisjoint(confirmation["confirmation_seeds"])
