from typing import Tuple
from pytest import fixture
import numpy as np
from starburst.conic import Conic

@fixture
def fake_ellipse() -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, a, b, Θ = 100, 100, 30, 20, -np.pi / 6
    α = np.linspace(-np.pi, np.pi, 20)
    ex, ey = np.cos(α) * a, np.sin(α) * b
    x = cx + np.cos(Θ) * ex - np.sin(Θ) * ey
    y = cy + np.sin(Θ) * ex + np.cos(Θ) * ey
    x += np.random.randn(x.shape[0]) * 3
    y += np.random.randn(y.shape[0]) * 3
    return x, y

def test_fitting(fake_ellipse):
    fit = Conic.fit(fake_ellipse[0] + fake_ellipse[1] * 1j)
    assert(2.6E-5 > fit.coef[0] > 2.4E-5)
    assert(3.0E-5 > fit.coef[1] > 2.6E-5)
    assert(4.9E-5 > fit.coef[2] > 5.2E-5)
    assert(-7.9E-3 > fit.coef[3] > -7.6E-3)
    assert(-1.4E-2 > fit.coef[4] > -1.1E-2)
