"""Simulated markets with known truth (slides 8-9, Appendix A5).

For stock i on one day, with the decision at minute 0, the return of minute k = 1..K is

    d_k = sum_j B[j, k-1] z_j                 signal known at the decision
        + sum_j sum_{l<k} B[j, k-1-l] u_{j,l}  future news, moving returns with the same decay
        + eta_k                                pure noise

Each row of B is one signal's clock. z_j is seen only through noisy proxies X = z + proxy_sd * v, next
to pure-noise columns. An AR(1) state with persistence phi is the profile B[k-1] = a * phi^(k-1) with
news variance 1 - phi^2. All labels are partial sums of one realised path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import brentq

K = 90            # minutes after the decision


@dataclass(frozen=True, eq=False)
class World:
    B: np.ndarray                          # (signals, K) clock of each signal
    news_var: np.ndarray                   # per-minute news variance of each signal
    Sz: np.ndarray                         # covariance of the signals
    noise_var: np.ndarray                  # per-minute variance of eta
    proxies: int = 4                       # noisy copies of each signal
    proxy_sd: float = 1.0
    nuisance: int = 36                     # pure-noise features
    noise_type: str = "iid"                # iid | arma | garch | common
    arma: tuple[float, float] = (0.0, 0.0)
    garch: tuple[float, float] = (0.08, 0.90)
    vol_dispersion: float = 0.0            # sd of the log stock-day volatility multiplier
    common_var: float = 0.0                # per-minute variance of a common factor
    beta_sd: float = 0.5                   # cross-sectional sd of common-factor loadings
    lags: int = 0                          # pre-decision return lags added to X (arma noise)
    drift: np.ndarray | None = None        # multiplies B outside training

    @property
    def signals(self) -> int:
        return self.B.shape[0]

    @property
    def features(self) -> int:
        return self.signals * self.proxies + self.lags + self.nuisance

    def out_of_sample(self) -> "World":
        return self if self.drift is None else replace(self, B=self.B * self.drift[:, None], drift=None)


def make_world(profiles: list[np.ndarray], news_var: list[float], Sz: np.ndarray | None = None,
               noise_var: np.ndarray | None = None, **options) -> World:
    B = np.vstack(profiles).astype(float)
    return World(B=B, news_var=np.asarray(news_var, float), Sz=np.eye(len(B)) if Sz is None else np.asarray(Sz, float),
                 noise_var=np.ones(K) if noise_var is None else np.asarray(noise_var, float), **options)


# ------------------------------------------------------------------------------- signal clocks
def ar1_profile(phi: float) -> np.ndarray:
    """b_k = phi^(k-1): phi = 0.5 is priced within minutes (FAST), 0.95 across the window (SLOW)."""
    return phi ** np.arange(K)


def drift_profile() -> np.ndarray:
    return np.ones(K)


def delayed_profile(tau: float = 5.0) -> np.ndarray:
    """Gamma-shaped response peaking at minute tau."""
    k = np.arange(1, K + 1)
    return (k / tau) * np.exp(1 - k / tau)


def reversal_profile(phi_fast: float = .6, phi_slow: float = .97, reversal: float = .4) -> np.ndarray:
    """Fast response, of which the fraction `reversal` is undone by minute K."""
    fast, slow = ar1_profile(phi_fast), ar1_profile(phi_slow)
    return fast - reversal * fast.sum() / slow.sum() * slow


def matching_ar1(first: float, cumulative: float) -> float:
    """phi of the AR(1) profile with this first response and this cumulative response."""
    return brentq(lambda p: (1 - p ** K) / (1 - p) - cumulative / first, -0.99, 0.9999)


# ------------------------------------------------------------------------------- population moments
def _arma_autocov(phi: float, theta: float, variance: float, lags: int) -> np.ndarray:
    g = np.zeros(lags + 1)
    g[0] = variance * (1 + 2 * phi * theta + theta ** 2) / (1 - phi ** 2)
    g[1] = variance * (1 + phi * theta) * (phi + theta) / (1 - phi ** 2)
    for j in range(2, lags + 1):
        g[j] = phi * g[j - 1]
    return g


def noise_cov(world: World, stocks: int | None = None) -> np.ndarray:
    """Covariance of eta_1..eta_K for one cross-sectionally demeaned stock."""
    if world.noise_type == "arma":
        return toeplitz(_arma_autocov(*world.arma, float(np.mean(world.noise_var)), K - 1))
    omega = np.diag(world.noise_var)
    if world.noise_type == "common":
        omega = omega + np.eye(K) * world.common_var * world.beta_sd ** 2 * (1 if stocks is None else 1 - 1 / stocks)
    return omega


def _news_matrix(kernel: np.ndarray) -> np.ndarray:
    """G[l, k] = kernel[k - l - 1] for k > l: a shock in minute l + 1 moves later minutes."""
    G = np.zeros((K, K))
    for l in range(K):
        G[l, l + 1:] = kernel[:K - l - 1]
    return G


def news_cov(world: World) -> np.ndarray:
    total = np.zeros((K, K))
    for j in range(world.signals):
        if world.news_var[j] > 0:
            G = _news_matrix(world.B[j])
            total += world.news_var[j] * G.T @ G
    return total


def moments(world: World, stocks: int | None = None) -> dict:
    """Sigma = Cov(X), Gamma = Cov(X, d_k) and C = Cov(d_k, d_l), all known exactly."""
    m, p, d = world.signals, world.proxies, world.features
    H = np.kron(np.eye(m), np.ones((p, 1)))
    s = m * p
    sigma, gamma = np.zeros((d, d)), np.zeros((d, K))
    sigma[:s, :s] = H @ world.Sz @ H.T + world.proxy_sd ** 2 * np.eye(s)
    gamma[:s] = H @ world.Sz @ world.B
    if world.lags:
        g = _arma_autocov(*world.arma, float(np.mean(world.noise_var)), K + world.lags)
        L = world.lags
        sigma[s:s + L, s:s + L] = toeplitz(g[:L])
        for i in range(L):                  # lag i is eta_{-i}; Cov(eta_{-i}, eta_k) = gamma_{k+i}
            gamma[s + i] = g[np.arange(1, K + 1) + i]
    sigma[s + world.lags:, s + world.lags:] = np.eye(world.nuisance)
    C = world.B.T @ world.Sz @ world.B + news_cov(world) + noise_cov(world, stocks)
    return {"sigma": sigma, "gamma": gamma, "C": C}


def calibrate(world: World, target_ic: float = .08, delta: int = K) -> World:
    """Scale all signals by one number so that the best possible IC on Y(delta) equals target_ic.

    Scaling B by s gives IC*^2(s) = s^2 P / (s^2 S + N), with P the predictable part, S the signal and
    news variance (news follows B) and N the pure noise. Solve for s.
    """
    mom = moments(world)
    g = mom["gamma"][:, :delta].sum(axis=1)
    P = g @ np.linalg.solve(mom["sigma"], g)
    S = (world.B.T @ world.Sz @ world.B + news_cov(world))[:delta, :delta].sum()
    N = noise_cov(world)[:delta, :delta].sum()
    return replace(world, B=world.B * np.sqrt(target_ic ** 2 * N / (P - target_ic ** 2 * S)))


# ------------------------------------------------------------------------------- simulation
def _noise(world: World, rows: int, rng: np.random.Generator, loadings: np.ndarray,
           days: int, stocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Pure noise (rows, K) and the pre-decision lags (rows, L), generated in time order."""
    variance = world.noise_var
    if world.noise_type == "arma":
        phi, theta = world.arma
        burn, L = 60, world.lags
        e = rng.standard_normal((rows, burn + L + K)).astype("float32") * np.sqrt(float(np.mean(variance)))
        eta = np.empty_like(e)
        eta[:, 0] = e[:, 0]
        for t in range(1, eta.shape[1]):
            eta[:, t] = phi * eta[:, t - 1] + e[:, t] + theta * e[:, t - 1]
        lags = eta[:, burn:burn + L][:, ::-1] if L else np.zeros((rows, 0), "float32")
        return eta[:, burn + L:], lags.copy()
    if world.noise_type == "garch":
        a, b = world.garch
        e = rng.standard_normal((rows, K)).astype("float32")
        scale = np.exp(world.vol_dispersion * rng.standard_normal(rows) - world.vol_dispersion ** 2)
        eta = np.empty_like(e)
        s2 = np.full(rows, variance[0], dtype="float64")
        for k in range(K):
            if k:
                s2 = variance[k] * (1 - a - b) + a * eta[:, k - 1].astype("float64") ** 2 / scale ** 2 + b * s2
            eta[:, k] = np.sqrt(s2) * e[:, k] * scale
        return eta, np.zeros((rows, 0), "float32")
    eta = (rng.standard_normal((rows, K)) * np.sqrt(variance)).astype("float32")
    if world.noise_type == "common":
        factor = rng.standard_normal((days, K)) * np.sqrt(world.common_var)
        eta += (np.repeat(factor, stocks, axis=0) * np.tile(loadings, days)[:, None]).astype("float32")
    return eta, np.zeros((rows, 0), "float32")


def _block(world: World, days: int, stocks: int, rng: np.random.Generator, loadings: np.ndarray) -> dict:
    rows = days * stocks
    m, p = world.signals, world.proxies
    z = rng.multivariate_normal(np.zeros(m), world.Sz, size=rows).astype("float32")
    proxies = np.repeat(z, p, axis=1) + world.proxy_sd * rng.standard_normal((rows, m * p)).astype("float32")
    nuisance = rng.standard_normal((rows, world.nuisance)).astype("float32")
    # Everything below happens after the decision, so X never sees it (arma lags are pre-decision noise)
    eta, lags = _noise(world, rows, rng, loadings, days, stocks)
    returns = (z @ world.B.astype("float32"))
    news = np.zeros_like(returns)
    for j in range(m):
        if world.news_var[j] > 0:
            u = rng.standard_normal((rows, K)).astype("float32") * np.sqrt(world.news_var[j])
            news += u @ _news_matrix(world.B[j]).astype("float32")
    return {"x": np.concatenate([proxies, lags.astype("float32"), nuisance], axis=1), "returns": returns + news + eta}


def simulate(world: World, stocks: int, train_days: int, validation_days: int, test_days: int, seed: int) -> dict:
    """A balanced panel with rows sorted by day. Test days use the drifted world, if any.

    The slides use only train and test. The validation block stays because the published seeds run
    through it, so removing it would change every test draw.
    """
    rng = np.random.default_rng(seed)
    loadings = 1 + world.beta_sd * rng.standard_normal(stocks)
    train = _block(world, train_days, stocks, rng, loadings)
    outside = world.out_of_sample()
    _block(outside, validation_days, stocks, rng, loadings)
    test = _block(outside, test_days, stocks, rng, loadings)
    return {"train": train, "test": test, "stocks": stocks}
