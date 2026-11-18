"""
scikit-learn estimators on top of mwqc — fit / predict / score, usable in Pipeline,
GridSearchCV, cross_val_score.

    from mwqc.sklearn_api import MWSurrogateRegressor, QuantumKernelClassifier, QuantumFeatureMap

MWSurrogateRegressor     Bayesian physics surrogate (deep ensemble) with predict(X, return_std=True)
QuantumKernelClassifier  SVM on the fidelity kernel |<phi(x)|phi(x')>|^2 of a quantum feature map
                         (ZZ map of Havlicek et al., Nature 2019). Statevector for <= 20 features,
                         MPS beyond that (the ZZ map with linear entanglement has bounded bond dim
                         per layer), so 50-200 feature kernels run on a laptop.
QuantumFeatureMap        Transformer: x -> single-qubit expectations <X_q>, <Y_q>, <Z_q> after the map.

Honest note: a quantum kernel is only useful if it beats classical kernels on your data.
On two-moons (80 points, 3-fold CV) RBF-SVC scores 0.975, this kernel 0.926 (linear ZZ) and 0.963
with entanglement="none", i.e. a product-state, classically trivial map. Features are scaled to
[0, pi/2]: larger ranges make the kernel too oscillatory (0.54-0.70). Treat it as an experiment.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .circuit import MWCircuit
from .mps import MPSState
from .statevector import StatevectorState
from .surrogate import MWSurrogate, mlp_factory


# ── regressor ────────────────────────────────────────────────────────────────
class MWSurrogateRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_members=5, epochs=300, hidden=64, depth=2, lr=3e-3, batch_size=32,
                 weight_decay=1e-4, likelihood="gaussian", calibration_fraction=0.15,
                 device="cpu", random_state=0):
        self.n_members, self.epochs, self.hidden, self.depth = n_members, epochs, hidden, depth
        self.lr, self.batch_size, self.weight_decay = lr, batch_size, weight_decay
        self.likelihood, self.calibration_fraction = likelihood, calibration_fraction
        self.device, self.random_state = device, random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self._1d = y.ndim == 1
        Y = y.reshape(len(y), -1).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        nval = int(len(X) * self.calibration_fraction) if self.calibration_fraction else 0
        val, fit = idx[:nval], idx[nval:]
        self.model_ = MWSurrogate(mlp_factory(self.hidden, self.depth), n_members=self.n_members,
                                  epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
                                  weight_decay=self.weight_decay, likelihood=self.likelihood,
                                  device=self.device, seed=self.random_state)
        self.model_.fit(X[fit], Y[fit])
        if nval >= 5:
            self.model_.calibrate(X[val], Y[val])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "model_")
        X = check_array(X)
        mu, sd = self.model_.predict(X)
        if self._1d:
            mu, sd = mu[:, 0], sd[:, 0]
        return (mu, sd) if return_std else mu


# ── quantum feature map / kernel ─────────────────────────────────────────────
def zz_feature_map(x: np.ndarray, reps: int = 2, entanglement: str = "linear") -> MWCircuit:
    """Havlicek ZZ feature map: [H, P(2 x_i), CX P(2(pi-x_i)(pi-x_j)) CX] x reps."""
    n = len(x)
    qc = MWCircuit(n, name="zz_map")
    pairs = [(i, i + 1) for i in range(n - 1)]
    if entanglement == "none":
        pairs = []
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * x[i], i)
        for i, j in pairs:
            qc.cx(i, j)
            qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
            qc.cx(i, j)
    return qc


class _MapBase(BaseEstimator):
    def _states(self, X):
        n = X.shape[1]
        use_sv = self.method == "statevector" or (self.method == "auto" and n <= 20)
        out = []
        for x in X:
            qc = zz_feature_map(x, self.reps, self.entanglement)
            if use_sv:
                out.append(StatevectorState(n).run(qc))
            else:
                out.append(MPSState(n, chi_max=self.chi_max).run(qc))
        return out, use_sv

    def _scale(self, X, fit=False):
        if fit:
            self.lo_, self.hi_ = X.min(0), X.max(0)
        rng = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        return np.clip((X - self.lo_) / rng, -0.5, 1.5) * self.feature_range


class QuantumKernelClassifier(ClassifierMixin, _MapBase):
    def __init__(self, reps=2, entanglement="linear", C=1.0, method="auto", chi_max=32,
                 feature_range=np.pi / 2):
        self.reps, self.entanglement, self.C = reps, entanglement, C
        self.method, self.chi_max, self.feature_range = method, chi_max, feature_range

    @staticmethod
    def _gram(A, B, sv):
        if sv:
            VA = np.array([s.to_vector() for s in A])
            VB = np.array([s.to_vector() for s in B])
            return np.abs(VA.conj() @ VB.T) ** 2
        K = np.empty((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                K[i, j] = abs(a.overlap(b)) ** 2
        return K

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Xs = self._scale(X.astype(float), fit=True)
        self.train_states_, self.sv_ = self._states(Xs)
        K = self._gram(self.train_states_, self.train_states_, self.sv_)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        self.n_features_in_ = X.shape[1]
        return self

    def kernel(self, X):
        check_is_fitted(self, "svc_")
        st, sv = self._states(self._scale(check_array(X).astype(float)))
        return self._gram(st, self.train_states_, sv)

    def predict(self, X):
        return self.svc_.predict(self.kernel(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self.kernel(X))


class QuantumFeatureMap(TransformerMixin, _MapBase):
    def __init__(self, reps=2, entanglement="linear", method="auto", chi_max=32, feature_range=np.pi / 2):
        self.reps, self.entanglement = reps, entanglement
        self.method, self.chi_max, self.feature_range = method, chi_max, feature_range

    def fit(self, X, y=None):
        X = check_array(X).astype(float)
        self._scale(X, fit=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "lo_")
        st, _ = self._states(self._scale(check_array(X).astype(float)))
        return np.array([np.concatenate([s.bloch_vector(q) for q in range(s.n)]) for s in st])
