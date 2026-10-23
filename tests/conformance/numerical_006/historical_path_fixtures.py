"""Old default-start recipes only freeze the old residual-test theta.

These are not candidate initializers or likelihood equations. Any floors here
belong to the historical fixture recipe, never to the new mathematical kernel.
The original test overwrites every mean coefficient before the tested requests.
"""
from types import SimpleNamespace
import numpy as np

def historical_GARCH_start(self, data: np.ndarray) -> np.ndarray:
    self._data = data

    min_positive   = 1e-8
    persist_target = 0.90
    beta_share     = 0.80

    # omega: small fraction of sample variance, scaled by (1 - persistence)
    sample_var = np.var(data)
    omega = max(min_positive, 0.05 * sample_var * (1 - persist_target))

    total_beta  = persist_target * beta_share
    total_alpha = persist_target * (1.0 - beta_share)

    alpha = ([max(min_positive, total_alpha / self.p)] * self.p) if self.p else []
    beta  = ([max(min_positive, total_beta  / self.q)] * self.q) if self.q else []

    return np.array([omega] + alpha + beta)


def historical_GJRGARCH_start(self, data: np.ndarray) -> np.ndarray:
    self._data = data

    min_positive   = 1e-8
    persist_target = 0.90
    beta_share     = 0.80

    total_beta  = persist_target * beta_share
    total_alpha = persist_target * (1.0 - beta_share) * 0.6  # 60% to alpha
    total_gamma = persist_target * (1.0 - beta_share) * 0.4  # 40% to gamma

    sample_var = np.var(data)
    start_persistence = total_alpha + 0.5 * total_gamma + total_beta
    omega = max(min_positive, sample_var * (1.0 - start_persistence))

    alpha = ([max(min_positive, total_alpha / self.p)] * self.p) if self.p else []
    gamma = ([max(min_positive, total_gamma / self.p)] * self.p) if self.p else []
    beta  = ([max(min_positive, total_beta  / self.q)] * self.q) if self.q else []

    return np.array([omega] + alpha + gamma + beta)


def historical_EGARCH_start(self, data: np.ndarray) -> np.ndarray:
    self._data = data
    sample_var = max(float(np.var(data)), 1e-8)
    beta_total = 0.90
    alpha_total = 0.10
    beta = np.full(self.q, beta_total / self.q, dtype=np.float64)
    alpha = np.full(self.p, alpha_total / self.p, dtype=np.float64)
    gamma = np.zeros(self.p, dtype=np.float64)
    gamma[0] = -0.05
    omega = (1.0 - float(beta.sum())) * np.log(sample_var)
    return np.concatenate([[omega], alpha, gamma, beta]).astype(np.float64)


def historical_APARCH_start(self, data: np.ndarray) -> np.ndarray:
    self._data = data
    sample_var = max(float(np.var(data)), 1e-8)
    alpha = np.full(self.p, 0.08 / self.p, dtype=np.float64)
    gamma = np.zeros(self.p, dtype=np.float64)
    beta = np.full(self.q, 0.85 / self.q, dtype=np.float64)
    omega = max(1e-8, 0.07 * sample_var)
    return np.concatenate(([omega], alpha, gamma, beta, [2.0]))


def repaired_sample_start(family,data,columns,p,q):
    recipe=(historical_GARCH_start,historical_GJRGARCH_start,historical_EGARCH_start,historical_APARCH_start)[family]
    declared=SimpleNamespace(p=p,q=q)
    volatility=recipe(declared,data)
    return np.r_[np.zeros(columns),volatility,10.]
