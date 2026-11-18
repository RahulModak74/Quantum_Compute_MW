"""scikit-learn API: estimator contract, pipelines, statevector == MPS kernel."""
import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from mwqc.sklearn_api import MWSurrogateRegressor, QuantumFeatureMap, QuantumKernelClassifier


def test_kernel_classifier_in_cv():
    X, y = make_moons(80, noise=0.15, random_state=0)
    clf = QuantumKernelClassifier(reps=2, C=5.0)
    assert clone(clf).get_params()["C"] == 5.0
    s = cross_val_score(clf, X, y, cv=3)
    assert s.mean() > 0.85          # measured 0.926; RBF-SVC gets 0.975 on the same split


def test_mps_kernel_equals_statevector_kernel():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (6, 8))
    y = np.array([0, 1] * 3)
    a = QuantumKernelClassifier(reps=2, method="statevector").fit(X, y)
    b = QuantumKernelClassifier(reps=2, method="mps", chi_max=64).fit(X, y)
    assert np.allclose(a.kernel(X), b.kernel(X), atol=1e-10)


def test_feature_map_pipeline():
    X, y = make_moons(60, noise=0.1, random_state=1)
    pipe = make_pipeline(QuantumFeatureMap(reps=2), LogisticRegression(max_iter=500))
    # measured 0.85, identical to LogisticRegression on the raw features: works, no gain
    assert cross_val_score(pipe, X, y, cv=3).mean() > 0.8


def test_regressor_std_and_score():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (200, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    r = MWSurrogateRegressor(n_members=2, epochs=120, random_state=0).fit(X, y)
    mu, sd = r.predict(X[:20], return_std=True)
    assert mu.shape == (20,) and np.all(sd > 0)
    assert r.score(X, y) > 0.9
