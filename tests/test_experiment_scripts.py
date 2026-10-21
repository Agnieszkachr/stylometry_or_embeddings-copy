import importlib.util
import warnings

import numpy as np

from conftest import REPO_ROOT


def _load_script(relpath, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp5_meter_multiclass_probe_runs():
    # Regression: LogisticRegression(multi_class=...) raises TypeError on
    # scikit-learn >= 1.8, which crashed this probe before Table 7 was saved.
    exp5 = _load_script("experiments/exp5_interpretability/run.py", "exp5_run")
    rng = np.random.RandomState(0)
    meters = ['Я', 'Х', 'Д', 'Ан', 'Аф']
    poems = [{'prosody': {'meter': meters[i % 5]}} for i in range(600)]
    embeddings = rng.randn(600, 8)
    embeddings[np.arange(600), np.arange(600) % 5] += 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = exp5.probe_meter_multiclass(poems, embeddings)

    assert result is not None
    assert result['accuracy'] > 0.5


def test_permutation_p_value_formatting():
    perm = _load_script("experiments/exp7_waterfall/run_permutation_test.py", "perm_test")
    assert perm.format_p(0.0, 1000) == "< 0.001"
    assert perm.format_p(0.0, 1000, relation=True) == "< 0.001"
    assert perm.format_p(0.058, 1000) == "0.058"
    assert perm.format_p(0.058, 1000, relation=True) == "= 0.058"
