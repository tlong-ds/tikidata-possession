from src import steps
from src.core import decode_switch as dsw


def test_grid_replaces_declared_keys_and_keeps_others():
    st = steps.make({"use": "decode.wide", "scale": [0.01, 0.05], "window": [3.0, 5]})
    g = st.grid({"scale": [0.1], "lambda_emit": [2.0], "eps_gate": [0.01]})
    assert g["scale"] == [0.01, 0.05] and g["eps_gate"] == [0.01]
    assert g["window"] == [3, 5] and all(isinstance(x, int) for x in g["window"])
    assert g["lambda_emit"] == [0.3, 0.5, 1.0]


def test_default_grid_contains_baseline_choice_and_builds_params():
    st = steps.make({"use": "decode.wide"})
    base = dict(dsw.DEFAULT_GRID, gate_scale=[0.5, 1.0], window=[0, 3])
    params = dsw.grid_params(st.grid(base))
    assert len(params) == 3 ** 5
    chosen = {"scale": 0.05, "lambda_emit": 1.0, "window": 3, "gate_scale": 0.5, "floor": 1e-6}
    assert any(all(getattr(p, k) == v for k, v in chosen.items()) for p in params)
