"""Focused regression tests; bench.check records the complete native coverage."""
import numpy as np
import pytest
from bench.api import Native, NativeError, Shape, inputs, oracle_points, error_metrics
from bench.check import contract_checks, positions


@pytest.fixture(scope='module')
def native():
    try:
        return Native()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_hand_oracle():
    s = Shape(cin=1, cout=1, groups=1, h=2, w=3, r=1, s=1, ph=0, pw=0)
    x = np.arange(1, 7, dtype=np.float32).reshape(s.input)
    w = np.full(s.weight, 2, np.float32)
    assert np.array_equal(oracle_points(s, x, w, positions(s, True)), np.arange(1, 7) * 2)


def test_contract(native):
    assert all(r['status'] == 'PASS' for r in contract_checks(native))


@pytest.mark.parametrize('impl', [0, 1, 2])
def test_group_isolation_and_impulse(native, impl):
    s = Shape(cin=4, cout=4, groups=2, h=3, w=5)
    x = np.zeros(s.input, np.float32)
    x[0, 0, 1, 2] = 1
    w = np.ones(s.weight, np.float32)
    y = np.empty(s.output, np.float32)
    native.cpu(s, x, w, y, impl, 1)
    assert np.all(y[:, 2:] == 0)
    expected = np.zeros(s.output, np.float32)
    expected[:, :2, :, 1:4] = 1
    assert np.array_equal(y, expected)


def test_nonfinite_metrics():
    for value in [np.nan, np.inf, -np.inf]:
        assert not error_metrics(np.array([value]), np.array([0.]))['passed']


def test_cuda_nondefault_stream(native):
    if not native.lib.gc_cuda_available():
        pytest.skip('Native CUDA unavailable; NOT_RUN')
    torch = pytest.importorskip('torch')
    if not native.lib.gc_cuda_available() or not torch.cuda.is_available():
        pytest.skip('Real native CUDA backend/device and CUDA torch required; NOT_RUN')
    s = Shape(cin=4, cout=4, groups=2, h=9, w=17)
    x, w = inputs(s)
    stream = torch.cuda.Stream()
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        dx, dw = torch.from_numpy(x).to(stream.device), torch.from_numpy(w).to(stream.device)
        dy = torch.empty(s.output, device=stream.device)
        native.cuda(s, dx, dw, dy, 2, stream.cuda_stream)
        event = torch.cuda.Event()
        event.record(stream)
        event.synchronize()
        actual = dy.cpu().numpy()
    p = positions(s, True)
    assert error_metrics(np.array([actual[i] for i in p]), oracle_points(s, x, w, p))['passed']
