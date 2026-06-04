#!/usr/bin/env python3
"""
Reproduce the stochastic discounted-sum monitoring appendix experiments.

This script is a lightly wrapped version of the original stochastic-monitoring
notebook.  The numerical monitor logic and plotting code are intentionally kept
close to the notebook, while the surrounding wrapper turns the notebook cells
into a reproducible command-line experiment.

Two Beta-process parameterisations are used deliberately:

* PHASES_SIM uses the high-variance ``(a,b)`` parameters for the single displayed
  run and for the first Monte Carlo table block.
* PHASES_MC uses the lower-variance ``(10a,10b)`` parameters for the second Monte
  Carlo table block.

Default outputs are written to ``experimental-results/stochastic-monitoring/``.
The main generated artefacts are Figures 5--12, the Monte Carlo table, and the
CSV files underlying the plots.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
import warnings
from scipy.special import hyp1f1
from scipy.optimize import minimize_scalar


def beta_subgaussian_proxy(alpha, beta, L=100):
    """
    Compute a sub-Gaussian proxy for a Beta(alpha, beta) random variable.

    Parameters
    ----------
    alpha, beta : float
        Shape parameters of the Beta distribution.
    L : float, optional
        Symmetric search radius for the MGF parameter lambda.

    Logic
    -----
    The function maximises the centred log-MGF ratio over positive and negative
    lambda values. The optimiser is applied to the negative objective because
    ``minimize_scalar`` performs minimisation.

    Returns
    -------
    float
        A numerical proxy sigma such that the centred Beta variable is treated as
        sigma-sub-Gaussian by the monitor bounds.
    """
    mu = alpha / (alpha + beta)

    def objective(lam):
        """
        Evaluate the negative centred log-MGF ratio at one lambda value.

        Parameters
        ----------
        lam : float
            MGF parameter used in the sub-Gaussian proxy optimisation.

        Logic
        -----
        Near zero, the expression is replaced by the Beta variance limit. Away from
        zero, the centred MGF is computed with the confluent hypergeometric function.

        Returns
        -------
        float
            Negative value of ``2 log E exp(lambda (X-mu)) / lambda^2``.
        """
        if abs(lam) < 1e-8:
            return -alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        log_mgf_centered = np.log(hyp1f1(alpha, alpha + beta, lam)) - lam * mu
        return -(2 * log_mgf_centered / lam ** 2)

    res_pos = minimize_scalar(objective, bounds=(1e-8, L), method="bounded")
    res_neg = minimize_scalar(objective, bounds=(-L, -1e-8), method="bounded")

    sigma2 = max(-res_pos.fun, -res_neg.fun)
    return np.sqrt(sigma2)


warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "figure.dpi": 130})

# =====================================================================
# 0.  GLOBAL CONFIGURATION
# =====================================================================

# Beta process phases: (n_steps, alpha, beta)  →  mean = α / (α + β)
PHASES_MC = [
    (150, 10, 90),  # mean ≈ 0.20  ⊥
    (150, 50, 50),  # mean = 0.50  ⊤
    (150, 80, 20),  # mean = 0.90  ⊥
    (150, 40, 60),  # mean = 0.40  ⊤
]

PHASES_SIM = [
    (150, 1, 9),  # mean ≈ 0.20  ⊥
    (150, 5, 5),  # mean = 0.50  ⊤
    (150, 8, 2),  # mean = 0.90  ⊥
    (150, 4, 6),  # mean = 0.40  ⊤
]

PHASES = PHASES_SIM

N_STEPS = sum(p[0] for p in PHASES)

SIGMA = max([beta_subgaussian_proxy(p[1], p[2]) for p in PHASES])

R_INF, R_SUP = 0.0, 1.0  # value range  (Beta ⊆ [0, 1])
D_R = R_SUP - R_INF  # diameter = 1
# SIGMA        = 0.1                 # sub-Gaussian proxy: σ = d_R / 2  (Hoeffding)
K1 = 2 ** 0.25 + 2 ** (-0.25) / np.sqrt(2.0)  # ≈ 1.784 (Howard et al.)

TARGET_L, TARGET_U = 0.40, 0.60
EPSILON = 0.1
DELTA = 0.10
R_PAST = 0.0  # past discount  (r = 0 → purely forward-looking)
S_FUT = 0.92  # future discount

LAMBDA = 1.0 + R_PAST / (1.0 - R_PAST) + S_FUT / (1.0 - S_FUT)

SIGMA_AVR = SIGMA / LAMBDA
D_R_AVR = D_R / LAMBDA

# Time points to inspect closely
PROBE_TIMES = list(range(1, 601, 1))  # list(range(25,601, 25)) # [25, 75, 175, 300, 400, 520]

SOUNDNESSES = ["pointwise", "local", "uniform"]
PAL = {"pointwise": "#1976D2", "local": "#E65100", "uniform": "#6A1B9A"}
PAL_LIGHT = {"pointwise": "#BBDEFB", "local": "#FFE0B2", "uniform": "#E1BEE7"}

N_MC = 1000  # Monte Carlo runs
RNG_SEED = 42

# ---------------------------------------------------------------------
# Scale toggle
# ---------------------------------------------------------------------

SCALE_MODE = "average"  # "average" or "absolute"

TARGET_L_AVG, TARGET_U_AVG = 0.35, 0.65
EPSILON_AVG = 0.05
DELTA = 0.01

R_PAST = 0.95
S_FUT = 0.95

LAMBDA = 1.0 + R_PAST / (1.0 - R_PAST) + S_FUT / (1.0 - S_FUT)

if SCALE_MODE == "average":
    SCALE_DENOM = LAMBDA
    TARGET_L, TARGET_U = TARGET_L_AVG, TARGET_U_AVG
    EPSILON = EPSILON_AVG
elif SCALE_MODE == "absolute":
    SCALE_DENOM = 1.0
    TARGET_L, TARGET_U = TARGET_L_AVG * LAMBDA, TARGET_U_AVG * LAMBDA
    EPSILON = EPSILON_AVG * LAMBDA
else:
    raise ValueError("SCALE_MODE must be 'average' or 'absolute'.")

R_INF_MON = R_INF / SCALE_DENOM
R_SUP_MON = R_SUP / SCALE_DENOM
D_R_MON = D_R / SCALE_DENOM


# =====================================================================
# 1.  Helpers
# =====================================================================

def verdict_is_incorrect(verdict, s_true, L, U, eps):
    """
    Check whether a decisive verdict violates approximate soundness.

    Parameters
    ----------
    verdict : int
        ``+1`` for inside, ``-1`` for outside, and ``0`` for inconclusive.
    s_true : float
        Oracle expected discounted value at the monitored time.
    L, U : float
        Lower and upper endpoints of the target interval.
    eps : float
        Approximation tolerance around the target boundary.

    Logic
    -----
    An inside verdict is correct only inside ``[L-eps, U+eps]``. An outside verdict
    is incorrect if the oracle value lies in the shrunken interval ``[L+eps,U-eps]``.

    Returns
    -------
    bool
        ``True`` exactly when the decisive verdict is unsound for ``s_true``.
    """
    """Approximate-soundness error for a decisive verdict."""
    if verdict == +1:
        return not (L - eps <= s_true <= U + eps)
    if verdict == -1:
        return L + eps <= s_true <= U - eps
    return False


def _mean_pm_std(x, digits=3):
    """
    Format a sample mean with an empirical standard deviation for LaTeX tables.

    Parameters
    ----------
    x : array-like
        Values to aggregate; non-numeric and missing entries are ignored.
    digits : int, optional
        Number of decimal places in both reported quantities.

    Logic
    -----
    The input is converted to a numeric pandas Series. The standard deviation uses
    ``ddof=1`` when at least two values are available and zero for a singleton.

    Returns
    -------
    str
        A LaTeX-ready string of the form ``mean plus/minus sd`` or ``--`` for no data.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return "--"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{digits}f} $\\pm$ {sd:.{digits}f}"


def export_mc_metrics_latex(metrics_df, path="monitor_mc_metrics.tex", digits=2):
    """
    Write the original one-block Monte Carlo summary table.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-run Monte Carlo metrics with one row per soundness level and run.
    path : str or pathlib.Path, optional
        Destination of the generated ``.tex`` table.
    digits : int, optional
        Number of decimal places in each mean and standard deviation.

    Logic
    -----
    The metrics are grouped by soundness, formatted as mean plus standard
    deviation, and embedded into a small booktabs LaTeX table.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        The table data as a DataFrame and the raw LaTeX source written to ``path``.
    """
    cols = [
        ("coverage_violation_rate", "CI viol."),
        ("any_coverage_violation", "Any CI"),
        ("release_fraction", "Released"),
        ("mean_delay", "Delay"),
        # ("incorrect_first_verdicts", "Wrong"),
        ("incorrect_first_rate", "Wrong rate"),
        ("any_incorrect", "Any wrong"),
    ]

    rows = []
    for snd, sub in metrics_df.groupby("soundness", sort=False):
        vals = [_mean_pm_std(sub[col], digits) for col, _ in cols]
        rows.append([snd.capitalize()] + vals)

    caption = (
        "The table reports Monte Carlo averages with standard deviations over independent runs. The Coverage columns measure failures of the statistical uncertainty intervals: CI viol. is the average fraction of checked (t,n) pairs for which the interval does not contain the oracle expected discounted sum, while Any CI is the fraction of runs in which at least one such failure occurs. The Release columns measure decisiveness: Released is the fraction of monitored time points that eventually receive a decisive verdict, and Delay is the average number of additional observations n−t needed until the first verdict. The Verdicts columns measure soundness-relevant mistakes of the first decisive verdicts: Wrong is the average number of incorrect first verdicts, Wrong rate normalises this by the number of released verdicts, and Any wrong is the fraction of runs containing at least one incorrect verdict."
    )

    header_1 = (
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Coverage} "
        "& \\multicolumn{2}{c}{Release} "
        "& \\multicolumn{2}{c}{Verdicts} \\\\\n"
    )
    header_2 = (
        "Sound. & CI viol. & Any CI & Released & Delay "
        "& Wrong rate & Any wrong \\\\\n"
        "\\midrule\n"
    )

    body = "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)

    latex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{" + caption + "}\n"
                                     "\\label{tab:mc_metrics}\n"
                                     "\\begin{tabular}{lcccccc}\n"
            + header_1
            + header_2
            + body
            + "\n\\bottomrule\n"
              "\\end{tabular}\n"
              "\\end{table}\n"
    )

    with open(path, "w") as f:
        f.write(latex)

    return pd.DataFrame(rows, columns=["Soundness"] + [c[1] for c in cols]), latex


# =====================================================================
# 1.  BETA PROCESS
# =====================================================================

def generate_process(phases=PHASES, rng=None):
    """
    Generate one piecewise-stationary Beta process realisation.

    Parameters
    ----------
    phases : sequence of tuple[int, float, float]
        Tuples ``(length, alpha, beta)`` describing each stationary phase.
    rng : numpy.random.Generator, optional
        Random generator used to sample observations. If omitted, ``RNG_SEED`` is
        used for reproducibility.

    Logic
    -----
    For each phase, the latent conditional mean is alpha/(alpha+beta) and the
    observations are i.i.d. Beta(alpha,beta). Phase bounds are tracked for plotting.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, list[tuple[int, int]], list[float]]
        Observations, latent means, inclusive phase bounds, and per-phase means.
    """
    """
    Piecewise-stationary Beta process.

    Returns
    -------
    X  : (N,) observations  X_t ~ Beta(α_t, β_t)
    P  : (N,) latent biases  P_t = α_t / (α_t + β_t) = E_{t-1}[X_t]
    bounds : list of (start, end) indices per phase
    means  : list of per-phase means
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    X_parts, P_parts = [], []
    bounds, means = [], []
    pos = 0
    for length, alpha, beta in phases:
        mean = alpha / (alpha + beta)
        P_parts.append(np.full(length, mean))
        X_parts.append(rng.beta(alpha, beta, length))
        bounds.append((pos, pos + length - 1))
        means.append(mean)
        pos += length
    return (np.concatenate(X_parts), np.concatenate(P_parts),
            bounds, means)


# =====================================================================
# 2.  STATISTICAL PRIMITIVES
# =====================================================================


def omega(t, n, r, s):
    """
    Compute the squared discounted-weight norm omega_{t,n}^{r,s}.

    Parameters
    ----------
    t, n : int
        Monitored time and current observation horizon with ``n >= t``.
    r, s : float
        Past and future discount factors.

    Logic
    -----
    The past and future geometric sums of squared weights are evaluated in closed
    form, with edge cases for zero discount factors.

    Returns
    -------
    float
        Sum of squared monitor weights up to horizon ``n``.
    """
    r2, s2 = r ** 2, s ** 2
    past = (r2 * (1 - r2 ** t) / (1 - r2)) if (r > 0 and t > 0) else 0.0
    nf = n - t + 1
    fut = (1 - s2 ** nf) / (1 - s2) if s > 0 else 1.0
    return past + fut


def beta_bound(t, n, r, s, sigma, delta, soundness):
    """
    Compute the statistical half-width beta for one soundness notion.

    Parameters
    ----------
    t, n : int
        Monitored time and current observation horizon.
    r, s : float
        Past and future discount factors.
    sigma : float
        Sub-Gaussian proxy for the centred observations.
    delta : float
        Error probability allocated to the requested guarantee.
    soundness : {'pointwise', 'local', 'uniform'}
        Statistical soundness level.

    Logic
    -----
    Pointwise uses the fixed-time Hoeffding--Azuma width. Local uses the stitched
    sub-Gaussian boundary. Uniform deflates delta by the summable time-index budget
    before applying the local boundary.

    Returns
    -------
    float
        Statistical confidence half-width for the finite discounted sum.
    """
    om = omega(t, n, r, s)
    V_base = sigma ** 2 * om
    V = max(1.0, V_base)

    if soundness == "pointwise":
        return np.sqrt(2.0 * sigma ** 2 * om * np.log(2.0 / delta))

    def _le(d):
        """
        Evaluate the local stitched sub-Gaussian boundary for a given delta.

        Parameters
        ----------
        d : float
            Error level used by the local boundary.

        Logic
        -----
        The precomputed variance proxy ``V_base`` and clipped scale ``V`` from the outer
        function are inserted into the Howard-style stitched expression.

        Returns
        -------
        float
            Local anytime-valid statistical half-width.
        """
        inner = (
                2.0 * np.log(np.log2(V) + 1.0)
                + np.log(2.0 * np.pi ** 2 / (6.0 * d))
        )
        return K1 * np.sqrt(V_base * inner)

    if soundness == "local":
        return _le(delta)

    delta_t = 6.0 * delta / (np.pi ** 2 * (t + 1) ** 2)
    return _le(delta_t)


def uncertainty_interval(s_hat, beta, gamma, R_inf, R_sup):
    """
    Convert an observed discounted sum into a full uncertainty interval.

    Parameters
    ----------
    s_hat : float
        Observed finite discounted sum.
    beta : float
        Statistical half-width around ``s_hat``.
    gamma : float
        Deterministic tail weight left outside the observed horizon.
    R_inf, R_sup : float
        Lower and upper bounds of the monitored value range.

    Logic
    -----
    The statistical interval ``s_hat ± beta`` is enlarged by the worst-case
    remaining deterministic tail ``gamma * [R_inf, R_sup]``.

    Returns
    -------
    tuple[float, float]
        Lower and upper endpoints of the monitor uncertainty interval.
    """
    return s_hat - beta + gamma * R_inf, s_hat + beta + gamma * R_sup


def det_tail(t, n, r, s):
    """
    Compute the deterministic unseen-tail weight gamma_{t,n}^{r,s}.

    Parameters
    ----------
    t, n : int
        Monitored time and observation horizon.
    r, s : float
        Past and future discount factors.

    Logic
    -----
    The missing pre-history contributes ``r^(t+1)/(1-r)`` and the missing future
    contributes ``s^(n-t+1)/(1-s)`` whenever the corresponding discount is nonzero.

    Returns
    -------
    float
        Total deterministic tail weight.
    """
    """γ^{r,s}_{t,n} = r^{t+1}/(1-r) + s^{n-t+1}/(1-s)  (Eq. 7)."""
    g = 0.0
    if r > 0:
        g += r ** (t + 1) / (1.0 - r)
    if s > 0:
        g += s ** (n - t + 1) / (1.0 - s)
    return g


def half_width(t, n, r, s, sigma, delta, d_R, soundness):
    """
    Compute the total monitor half-width beta plus deterministic tail.

    Parameters
    ----------
    t, n : int
        Monitored time and current horizon.
    r, s : float
        Discount factors.
    sigma, delta : float
        Statistical proxy and error probability.
    d_R : float
        Diameter of the monitored value range.
    soundness : str
        Soundness level passed to ``beta_bound``.

    Logic
    -----
    The statistical half-width is added to ``d_R`` times the deterministic tail.

    Returns
    -------
    float
        Total symmetric uncertainty half-width.
    """
    """Total symmetric half-width  β + d_R · γ."""
    return (beta_bound(t, n, r, s, sigma, delta, soundness)
            + d_R * det_tail(t, n, r, s))


def _verdict(ci_lo, ci_hi, L, U, eps):
    """
    Map an uncertainty interval to a three-valued monitor verdict.

    Parameters
    ----------
    ci_lo, ci_hi : float
        Lower and upper endpoints of the uncertainty interval.
    L, U : float
        Target interval endpoints.
    eps : float
        Approximation tolerance.

    Logic
    -----
    The function releases an inside verdict if the whole interval is inside the
    expanded target and an outside verdict if it is disjoint from the shrunken target.

    Returns
    -------
    int
        ``+1`` for inside, ``-1`` for outside, and ``0`` for inconclusive.
    """
    """
    +1 = ⊤  (CI ⊆ I+ε),   -1 = ⊥  (CI ∩ I-ε = ∅),   0 = ?
    """
    if ci_lo >= L - eps and ci_hi <= U + eps:
        return +1
    if ci_hi < L + eps or ci_lo > U - eps:
        return -1
    return 0


def true_expected_sum(t, P, r, s, phases):
    """
    Compute the oracle infinite expected discounted sum for one time index.

    Parameters
    ----------
    t : int
        Monitored time.
    P : numpy.ndarray
        Latent conditional means over the simulated finite horizon.
    r, s : float
        Past and future discount factors.
    phases : sequence of tuple[int, float, float]
        Phase specification used to infer the asymptotic final-phase mean.

    Logic
    -----
    Observed finite past and future latent means are summed exactly. The unobserved
    future is analytically continued with the mean of the final Beta phase.

    Returns
    -------
    float
        Oracle infinite expected discounted sum at time ``t``.
    """
    """
    Oracle S^{r,s}_t(W): finite sum over known P plus analytical future tail
    using the last phase's mean as the asymptotic continuation.
    """
    N = len(P)
    past = (sum(r ** i * P[t - i] for i in range(1, t + 1))
            if r > 0 and t > 0 else 0.0)
    fut_obs = sum(s ** i * P[t + i] for i in range(N - t))
    p_inf = phases[-1][1] / (phases[-1][1] + phases[-1][2])  # last-phase mean
    tail = s ** (N - t) / (1.0 - s) * p_inf if s > 0 else 0.0
    return past + fut_obs + tail


# =====================================================================
# 3.  MONITOR RUN
# =====================================================================

def run_monitor(X, P, r, s, sigma, delta, d_R, L, U, eps,
                probe_times, phases, soundnesses=SOUNDNESSES,
                scale_denom=1.0, R_inf=R_INF, R_sup=R_SUP):
    """
    Run the stochastic discounted-sum monitor on one simulated trajectory.

    Parameters
    ----------
    X, P : numpy.ndarray
        Observed values and latent conditional means.
    r, s : float
        Past and future discounts.
    sigma, delta, d_R : float
        Statistical proxy, error level, and scaled range diameter.
    L, U, eps : float
        Target interval and approximation tolerance.
    probe_times : sequence[int]
        Time indices for which monitor records are produced.
    phases : sequence[tuple[int, float, float]]
        Beta phase specification used for the oracle expected sum.
    soundnesses : sequence[str], optional
        Soundness levels to evaluate.
    scale_denom : float, optional
        Normalisation constant when plotting discounted averages.
    R_inf, R_sup : float, optional
        Scaled lower and upper range bounds.

    Logic
    -----
    For every probe time and soundness level, the finite observed discounted sum is
    updated incrementally over ``n >= t``. Each row records the statistical width,
    deterministic tail, confidence interval, verdict, and oracle coverage status.

    Returns
    -------
    dict[tuple[int, str], pandas.DataFrame]
        Monitor traces keyed by ``(probe_time, soundness)``.
    """
    """
    Run the stochastic discounted-sum monitor.

    For each probe time t and soundness level, records at every
    observation step n ≥ t:

        s_hat  – observed discounted sum  Ŝ^{r,s}_t(W_{0:n})
        s_exp  – expected discounted sum  S^{r,s}_t(W_{0:n})  (uses P)
        s_true – oracle infinite expected sum  S^{r,s}_t(W)
        beta   – statistical half-width β
        det    – deterministic tail  d_R · γ
        hw     – total half-width  β + d_R·γ
        ci_lo / ci_hi – uncertainty interval endpoints
        verdict    : +1 / 0 / -1
        violation  : bool  (s_true outside CI)

    Returns dict  (t, soundness) → pd.DataFrame
    """
    N = len(X)
    results = {}

    for t in probe_times:
        if t >= N:
            continue
        # ── Past sums (fixed for given t) ──────────────────────────
        past_hat = (sum(r ** i * X[t - i] for i in range(1, t + 1))
                    if r > 0 and t > 0 else 0.0)
        past_exp = (sum(r ** i * P[t - i] for i in range(1, t + 1))
                    if r > 0 and t > 0 else 0.0)
        s_true = true_expected_sum(t, P, r, s, phases) / scale_denom

        for snd in soundnesses:
            rows = []
            s_hat_run = (past_hat + X[t]) / scale_denom
            s_exp_run = (past_exp + P[t]) / scale_denom

            for n in range(t, N):
                if n > t:
                    i = n - t
                    s_hat_run += s ** i * X[n] / scale_denom
                    s_exp_run += s ** i * P[n] / scale_denom

                b = beta_bound(t, n, r, s, sigma, delta, snd) / scale_denom
                g = det_tail(t, n, r, s)
                d = d_R * g
                lo, hi = uncertainty_interval(s_hat_run, b, g, R_inf, R_sup)
                hw = max(s_hat_run - lo, hi - s_hat_run)

                verd = _verdict(lo, hi, L, U, eps)
                viol = bool(s_true < lo or s_true > hi)

                rows.append(dict(
                    t=t, n=n, soundness=snd,
                    s_hat=s_hat_run, s_exp=s_exp_run, s_true=s_true,
                    beta=b, det=d, hw=hw,
                    ci_lo=lo, ci_hi=hi,
                    verdict=verd, violation=viol,
                ))
            results[(t, snd)] = pd.DataFrame(rows)

    return results


# =====================================================================
# 4.  MONTE CARLO VIOLATION STUDY
# =====================================================================
def monte_carlo(r, s, sigma, delta, d_R, L, U, eps,
                phases=PHASES, probe_times=PROBE_TIMES,
                soundnesses=SOUNDNESSES, n_runs=N_MC, seed=RNG_SEED,
                scale_denom=1.0, R_inf=R_INF, R_sup=R_SUP):
    """
    Estimate coverage and first-verdict error rates over repeated simulations.

    Parameters
    ----------
    r, s, sigma, delta, d_R : float
        Monitor discount factors, statistical proxy, error level, and range diameter.
    L, U, eps : float
        Target interval and approximation tolerance.
    phases : sequence[tuple[int, float, float]], optional
        Beta process phases used for this Monte Carlo block.
    probe_times : sequence[int], optional
        Monitored time indices.
    soundnesses : sequence[str], optional
        Soundness levels to compare.
    n_runs : int, optional
        Number of independent simulations.
    seed : int, optional
        Seed for the Monte Carlo generator.
    scale_denom, R_inf, R_sup : float, optional
        Scaling and bounded-range parameters for average-vs-absolute mode.

    Logic
    -----
    The function precomputes deterministic and statistical widths, then simulates
    independent Beta trajectories. It counts pointwise, local, and uniform coverage
    violations and records release delays and first-verdict mistakes.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Pointwise violation rates, local violation rates, uniform violation rates,
        and raw per-run summary metrics.
    """
    N = sum(p[0] for p in phases)
    probe_times = [t for t in probe_times if t < N]
    rng = np.random.default_rng(seed)

    _b, _g = {}, {}
    for t in probe_times:
        for n in range(t, N):
            _g[(t, n)] = det_tail(t, n, r, s)
            for snd in soundnesses:
                _b[(t, n, snd)] = beta_bound(t, n, r, s, sigma, delta, snd) / scale_denom

    pw_count = {
        (t, snd, n): 0
        for t in probe_times for snd in soundnesses
        for n in range(t, N)
    }
    loc_count = {(t, snd): 0 for t in probe_times for snd in soundnesses}
    unif_count = {snd: 0 for snd in soundnesses}

    metric_rows = []
    progress_step = max(1, n_runs // 10)

    for rcount in range(n_runs):
        if rcount % progress_step == 0:
            print(f"{rcount}/{n_runs}")

        X, P, _, _ = generate_process(phases, rng)

        for snd in soundnesses:
            run_violated = False

            run_grid_points = 0
            run_ci_violations = 0

            first_verdicts = 0
            incorrect_first_verdicts = 0
            delays = []

            for t in probe_times:
                past_hat = (
                    sum(r ** i * X[t - i] for i in range(1, t + 1))
                    if r > 0 and t > 0 else 0.0
                )
                s_true = true_expected_sum(t, P, r, s, phases) / scale_denom
                s_hat_run = (past_hat + X[t]) / scale_denom

                t_violated = False
                first_seen = False

                for n in range(t, N):
                    if n > t:
                        s_hat_run += s ** (n - t) * X[n] / scale_denom

                    lo, hi = uncertainty_interval(
                        s_hat_run,
                        _b[(t, n, snd)],
                        _g[(t, n)],
                        R_inf,
                        R_sup,
                    )

                    viol = bool(s_true < lo or s_true > hi)
                    verd = _verdict(lo, hi, L, U, eps)

                    run_grid_points += 1

                    if viol:
                        pw_count[(t, snd, n)] += 1
                        run_ci_violations += 1
                        t_violated = True
                        run_violated = True

                    if verd != 0 and not first_seen:
                        first_seen = True
                        first_verdicts += 1
                        delays.append(n - t)

                        if verdict_is_incorrect(verd, s_true, L, U, eps):
                            incorrect_first_verdicts += 1

                if t_violated:
                    loc_count[(t, snd)] += 1

            if run_violated:
                unif_count[snd] += 1

            metric_rows.append(dict(
                run=rcount,
                soundness=snd,
                coverage_violation_rate=run_ci_violations / run_grid_points,
                any_coverage_violation=float(run_ci_violations > 0),
                first_verdicts=first_verdicts,
                release_fraction=first_verdicts / len(probe_times),
                mean_delay=np.mean(delays) if len(delays) else np.nan,
                median_delay=np.median(delays) if len(delays) else np.nan,
                incorrect_first_verdicts=incorrect_first_verdicts,
                incorrect_first_rate=(
                    incorrect_first_verdicts / first_verdicts
                    if first_verdicts > 0 else np.nan
                ),
                any_incorrect=float(incorrect_first_verdicts > 0),
            ))

    pw_rows = [
        dict(t=t, soundness=snd, n=n, rate=pw_count[(t, snd, n)] / n_runs)
        for t in probe_times for snd in soundnesses
        for n in range(t, N)
    ]
    loc_rows = [
        dict(t=t, soundness=snd, rate=loc_count[(t, snd)] / n_runs)
        for t in probe_times for snd in soundnesses
    ]
    unif_rows = [
        dict(soundness=snd, rate=unif_count[snd] / n_runs)
        for snd in soundnesses
    ]

    return (
        pd.DataFrame(pw_rows),
        pd.DataFrame(loc_rows),
        pd.DataFrame(unif_rows),
        pd.DataFrame(metric_rows),
    )


# =====================================================================
# 5.  PLOTTING
# =====================================================================

# ── 5a.  Fig 1: Process overview ────────────────────────────────────


def plot_process(X, P, phase_bounds, phase_means, L, U, eps,
                 r, s, phases, scale_denom=1.0):
    """
    Plot the simulated Beta process and its discounted expected values.

    Parameters
    ----------
    X, P : numpy.ndarray
        Observed process values and latent phase means.
    phase_bounds, phase_means : sequence
        Phase intervals and labels used for shaded backgrounds and the phase bar.
    L, U, eps : float
        Target interval and tolerance displayed on the plot.
    r, s : float
        Discount factors used for empirical and expected discounted sums.
    phases : sequence[tuple[int, float, float]]
        Phase specification used by the oracle expected sum.
    scale_denom : float, optional
        Normalisation factor for discounted averages.

    Logic
    -----
    The figure overlays observations, latent means, empirical discounted sums, and
    oracle expected discounted sums. A lower panel visualises the phase sequence.

    Returns
    -------
    matplotlib.figure.Figure
        The process overview figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 4),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax = axes[0]
    t_idx = np.arange(len(X))

    S_true = np.array([
        true_expected_sum(t, P, r, s, phases) / scale_denom
        for t in t_idx
    ])

    S_emp = np.array([
        (
                (sum(r ** i * X[t - i] for i in range(1, t + 1))
                 if r > 0 and t > 0 else 0.0)
                + sum(s ** i * X[t + i] for i in range(len(X) - t))
        ) / scale_denom
        for t in t_idx
    ])

    phase_cols = ["#E3F2FD", "#FFF8E1", "#FCE4EC", "#E8F5E9"]
    for (s0, s1), c in zip(phase_bounds, phase_cols):
        ax.axvspan(s0, s1, color=c, alpha=0.6, zorder=0)

    ax.scatter(t_idx, X, s=3, alpha=0.35, color="#546E7A",
               label="Observation $X_t$", zorder=2)
    ax.plot(t_idx, P, color="#D32F2F", lw=1.6,
            label="Latent bias $P_t$", zorder=3)
    ax.plot(t_idx, S_emp, color="#00838F", lw=1.4, ls=":",
            label=r"Empirical discounted sum $\hat{S}_t$", zorder=4)
    ax.plot(t_idx, S_true, color="#6A1B9A", lw=1.8, ls="-.",
            label=r"True discounted expected sum $S_t$", zorder=5)

    ax.axhspan(L, U, color="#A5D6A7", alpha=0.35,
               label=f"Target $I=[{L},{U}]$")
    ax.axhspan(L - eps, L, color="#FFCC80", alpha=0.3)
    ax.axhspan(U, U + eps, color="#FFCC80", alpha=0.3,
               label=f"Buffer ±ε={eps}")
    ax.axhline(L, color="#388E3C", lw=0.8, ls="--")
    ax.axhline(U, color="#388E3C", lw=0.8, ls="--")

    ax.set_ylabel("Value")
    y_max = max(1.05, np.max(S_true) * 1.05, np.max(S_emp) * 1.05)
    ax.set_ylim(-0.05, 1.05 if scale_denom != 1.0 else y_max)
    ax.legend(loc="upper right", ncol=6)
    ax.set_title("Stochastic Beta process with piecewise-stationary phases", fontsize=10, y=1.005)

    ax2 = axes[1]
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    labels = [f"Ph {i + 1}\nμ={m:.2f}" for i, m in enumerate(phase_means)]
    bar_cols = ["#1565C0", "#2E7D32", "#B71C1C", "#1B5E20"]
    for (s0, s1), lab, bc in zip(phase_bounds, labels, bar_cols):
        ax2.barh(0.5, s1 - s0, left=s0, height=0.8,
                 color=bc, alpha=0.75, align="center")
        ax2.text((s0 + s1) / 2, 0.5, lab, ha="center", va="center",
                 color="white", fontsize=7, fontweight="bold")
    ax2.set_xlabel("Time step  $t$")

    for a in axes:
        for (s0, s1) in phase_bounds[1:]:
            a.axvline(s0, color="#455A64", lw=0.6, ls=":")

    fig.tight_layout(h_pad=0.4)
    return fig


def plot_CI_process(X, P, phase_bounds, phase_means, L, U, eps,
                    r, s, phases, results, scale_denom=1.0, min_interval=False):
    """
    Plot process trajectories with one uncertainty interval per monitored time.

    Parameters
    ----------
    X, P : numpy.ndarray
        Observed values and latent conditional means.
    phase_bounds, phase_means : sequence
        Phase layout used in the background and summary bar.
    L, U, eps : float
        Target interval and tolerance.
    r, s : float
        Discount factors.
    phases : sequence[tuple[int, float, float]]
        Phase specification for oracle expected sums.
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor output returned by ``run_monitor``.
    scale_denom : float, optional
        Normalisation factor for average mode.
    min_interval : bool, optional
        If ``False``, use the interval at the first decisive verdict; if ``True``,
        use the narrowest interval reached by the finite run.

    Logic
    -----
    For each soundness level, the process curves are shown together with vertical
    confidence intervals selected from the monitor trace for each probe time.

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing verdict-time or end-of-run uncertainty intervals.
    """
    n_rows = len(SOUNDNESSES) + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11, 2.6 * len(SOUNDNESSES) + 1.0),
        gridspec_kw={"height_ratios": [3] * len(SOUNDNESSES) + [1]},
        sharex=True
    )

    t_idx = np.arange(len(X))

    S_true = np.array([
        true_expected_sum(t, P, r, s, phases) / scale_denom
        for t in t_idx
    ])

    S_emp = np.array([
        (
                (sum(r ** i * X[t - i] for i in range(1, t + 1))
                 if r > 0 and t > 0 else 0.0)
                + sum(s ** i * X[t + i] for i in range(len(X) - t))
        ) / scale_denom
        for t in t_idx
    ])

    phase_cols = ["#E3F2FD", "#FFF8E1", "#FCE4EC", "#E8F5E9"]

    for row, snd in enumerate(SOUNDNESSES):
        ax = axes[row]

        for (s0, s1), c in zip(phase_bounds, phase_cols):
            ax.axvspan(s0, s1, color=c, alpha=0.6, zorder=0)

        ax.scatter(t_idx, X, s=3, alpha=0.25, color="#546E7A",
                   label="Observation $X_t$", zorder=2)
        ax.plot(t_idx, P, color="#D32F2F", lw=1.3,
                label="Latent bias $P_t$", zorder=3)
        ax.plot(t_idx, S_emp, color="#00838F", lw=1.2, ls=":",
                label=r"Empirical discounted sum $\hat{S}_t$", zorder=4)
        ax.plot(t_idx, S_true, color="#6A1B9A", lw=1.5, ls="-.",
                label=r"True discounted expected sum $S_t$", zorder=5)

        # Interval at first verdict; if no verdict, take smallest interval.

        if min_interval:
            # Always use the smallest CI for each t.
            xs, los, his = [], [], []
            for (t, snd_key), df in results.items():
                if snd_key != snd or df.empty:
                    continue

                width = df["ci_hi"] - df["ci_lo"]
                rec = df.loc[width.idxmin()]

                xs.append(t)
                los.append(rec["ci_lo"])
                his.append(rec["ci_hi"])
        else:
            xs, los, his = [], [], []
            for (t, snd_key), df in results.items():
                if snd_key != snd or df.empty:
                    continue

                decisive = df[df["verdict"] != 0]
                rec = decisive.iloc[0] if len(decisive) else df.loc[(df["ci_hi"] - df["ci_lo"]).idxmin()]

                xs.append(t)
                los.append(rec["ci_lo"])
                his.append(rec["ci_hi"])

        ax.vlines(xs, los, his, color=PAL[snd], lw=0.8, alpha=0.35,
                  label="CI at verdict / smallest CI", zorder=6)

        ax.axhspan(L, U, color="#A5D6A7", alpha=0.35,
                   label=f"Target $I=[{L},{U}]$")
        ax.axhspan(L - eps, L, color="#FFCC80", alpha=0.3)
        ax.axhspan(U, U + eps, color="#FFCC80", alpha=0.3,
                   label=f"Buffer ±ε={eps}")
        ax.axhline(L, color="#388E3C", lw=0.8, ls="--")
        ax.axhline(U, color="#388E3C", lw=0.8, ls="--")

        for s0, _ in phase_bounds[1:]:
            ax.axvline(s0, color="#455A64", lw=0.6, ls=":")

        ax.set_ylabel("Value")
        ax.set_title(f"{snd.capitalize()} soundness", color=PAL[snd], fontweight="bold")

        y_max = max(1.05, np.max(S_true) * 1.05, np.max(S_emp) * 1.05, max(his) * 1.05 if his else 1.05)
        y_min = min(-0.05, min(los) * 1.05 if los else -0.05)
        ax.set_ylim(y_min, 1.05 if scale_denom != 1.0 else y_max)

        if row == 0:
            ax.legend(loc="upper right", ncol=6)

    ax2 = axes[-1]
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    labels = [f"Ph {i + 1}\nμ={m:.2f}" for i, m in enumerate(phase_means)]
    bar_cols = ["#1565C0", "#2E7D32", "#B71C1C", "#1B5E20"]

    for (s0, s1), lab, bc in zip(phase_bounds, labels, bar_cols):
        ax2.barh(0.5, s1 - s0, left=s0, height=0.8,
                 color=bc, alpha=0.75, align="center")
        ax2.text((s0 + s1) / 2, 0.5, lab, ha="center", va="center",
                 color="white", fontsize=7, fontweight="bold")

    for s0, _ in phase_bounds[1:]:
        ax2.axvline(s0, color="#455A64", lw=0.6, ls=":")

    ax2.set_xlabel("Time step  $t$")

    fig.suptitle("Stochastic Beta process with verdict-time uncertainty intervals",
                 fontsize=10, y=1.005)
    fig.tight_layout(h_pad=0.4)
    return fig


# ── 5b.  Fig 2: Uncertainty-interval evolution ───────────────────────

def plot_ci_evolution(results, probe_times_plot, L, U, eps, phase_bounds):
    """
    Plot uncertainty interval evolution for selected probe times.

    Parameters
    ----------
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor records produced by ``run_monitor``.
    probe_times_plot : sequence[int]
        Probe times to display as rows.
    L, U, eps : float
        Target interval and tolerance boundaries.
    phase_bounds : sequence[tuple[int, int]]
        Phase boundaries drawn as vertical reference lines.

    Logic
    -----
    Each subplot shows the confidence band, observed finite sum, expected finite
    sum, oracle true value, and target/tolerance thresholds as the horizon grows.

    Returns
    -------
    matplotlib.figure.Figure
        Grid of uncertainty-evolution subplots.
    """
    """
    For each probe time (row) and each soundness (column), plot the CI
    band, observed sum, expected sum, and true value as n grows.
    """
    n_rows = len(probe_times_plot)
    n_cols = len(SOUNDNESSES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.8 * n_rows),
                             sharex=False, sharey=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Uncertainty intervals vs. observation count  n",
                 fontsize=10, y=1.005)

    for row, t in enumerate(probe_times_plot):
        for col, snd in enumerate(SOUNDNESSES):
            ax = axes[row, col]
            df = results[(t, snd)]
            n_arr = df["n"].values

            # CI band
            ax.fill_between(n_arr, df["ci_lo"], df["ci_hi"],
                            color=PAL_LIGHT[snd], alpha=0.7,
                            label="CI band")
            ax.plot(n_arr, df["s_hat"], color=PAL[snd], lw=1.2,
                    label=r"$\hat{S}_t$ (obs)")
            ax.plot(n_arr, df["s_exp"], color="#37474F", lw=1.0,
                    ls="--", label=r"$S_t^{\mathrm{exp}}$ (finite)")
            ax.axhline(df["s_true"].iloc[0], color="#B71C1C", lw=1.2,
                       ls=(0, (4, 2)), label=r"$S_t^\infty$ (oracle)")

            # Target interval and epsilon tolerance bounds
            ax.axhline(L - eps, color="#F57C00", lw=1.2, ls=":", alpha=0.95,
                       label=r"$L-\epsilon$")
            ax.axhline(L, color="#2E7D32", lw=1.5, ls="--", alpha=0.95,
                       label=r"$L$")
            ax.axhline(U, color="#2E7D32", lw=1.5, ls="--", alpha=0.95,
                       label=r"$U$")
            ax.axhline(U + eps, color="#F57C00", lw=1.2, ls=":", alpha=0.95,
                       label=r"$U+\epsilon$")

            if L + eps <= U - eps:
                ax.axhline(L + eps, color="#EF6C00", lw=1.0, ls="-.", alpha=0.9,
                           label=r"$L+\epsilon$")
                ax.axhline(U - eps, color="#EF6C00", lw=1.0, ls="-.", alpha=0.9,
                           label=r"$U-\epsilon$")

            # Verdict overlay
            for row_d in df.itertuples():
                c = {1: "#4CAF5020", -1: "#F4433620", 0: None}[row_d.verdict]
                if c:
                    ax.axvline(row_d.n, color=c, lw=0.3, alpha=0.4)

            # Probe-time marker
            ax.axvline(t, color="#FF6F00", lw=0.9, ls=":", alpha=0.7)

            if row == 0:
                ax.set_title(snd.capitalize(), color=PAL[snd], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"t = {t}", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Observation step  n")
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

            ax.set_xlim(t, df["n"].max())
            ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# ── 5c.  Fig 3: Verdict heatmap ─────────────────────────────────────

def plot_verdict_heatmap(results, probe_times, phase_bounds):
    """
    Plot the three-valued verdict timeline as a heatmap.

    Parameters
    ----------
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor output keyed by probe time and soundness.
    probe_times : sequence[int]
        Monitored times used as heatmap rows.
    phase_bounds : sequence[tuple[int, int]]
        Phase boundaries drawn over the heatmap.

    Logic
    -----
    For each soundness level, verdict values are written into a matrix indexed by
    probe time and observation horizon. Colour encodes outside, inconclusive, inside.

    Returns
    -------
    matplotlib.figure.Figure
        Verdict heatmap figure.
    """
    N = sum(p[0] for p in PHASES)

    max_height = 8.27  # A4 landscape height in inches
    min_height = 3.0
    row_height = 0.28
    fig_height = min(max_height, max(min_height, row_height * len(probe_times)))

    fig, axes = plt.subplots(
        1, len(SOUNDNESSES),
        figsize=(11.69, fig_height),  # A4 landscape width
        sharey=True
    )
    if len(SOUNDNESSES) == 1:
        axes = [axes]

    fig.suptitle("Verdict timeline heatmap  (⊤ inside / ? uncertain / ⊥ outside)",
                 fontsize=10, y=1.005)

    cmap = ListedColormap(["#EF5350", "#BDBDBD", "#66BB6A"])  # ⊥ ? ⊤
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    max_labels = 18
    label_step = max(1, int(np.ceil(len(probe_times) / max_labels)))

    for col, snd in enumerate(SOUNDNESSES):
        ax = axes[col]
        matrix = np.full((len(probe_times), N), np.nan)
        for row, t in enumerate(probe_times):
            if (t, snd) not in results:
                continue
            df = results[(t, snd)]
            for rec in df.itertuples():
                matrix[row, rec.n] = rec.verdict

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=[0, N, len(probe_times), 0],
            interpolation="nearest",
            rasterized=True,
        )

        yticks = np.arange(0, len(probe_times), label_step) + 0.5
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"t={probe_times[i]}" for i in range(0, len(probe_times), label_step)],
            fontsize=7
        )

        ax.set_xlabel("Observation step  n")
        ax.set_title(snd.capitalize(), color=PAL[snd], fontweight="bold")

        # Phase boundaries
        for s0, _ in phase_bounds[1:]:
            ax.axvline(s0, color="white", lw=0.8, ls="--", alpha=0.6)

        if col == len(SOUNDNESSES) - 1:
            cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], pad=0.01)
            cbar.ax.set_yticklabels(["⊥", "?", "⊤"], fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── 5d.  Fig 4: Bound-width decomposition ───────────────────────────

def plot_bound_decomposition(results, probe_times_plot, phase_bounds):
    """
    Plot statistical and deterministic contributions to monitor width.

    Parameters
    ----------
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor records produced by ``run_monitor``.
    probe_times_plot : sequence[int]
        Probe times to display as rows.
    phase_bounds : sequence[tuple[int, int]]
        Phase boundaries shown as visual references.

    Logic
    -----
    The deterministic tail is stacked below the statistical half-width and the
    total half-width is overlaid as a line for each soundness level.

    Returns
    -------
    matplotlib.figure.Figure
        Bound decomposition grid.
    """
    """
    For each probe time (row) and each soundness (column), show how the
    statistical part β and deterministic tail d_R·γ of the half-width evolve.
    """
    n_rows = len(probe_times_plot)
    n_cols = len(SOUNDNESSES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.8 * n_rows),
                             sharex=False, sharey=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Bound-width decomposition",
                 fontsize=10, y=1.005)

    for row, probe_t in enumerate(probe_times_plot):
        for col, snd in enumerate(SOUNDNESSES):
            ax = axes[row, col]
            df = results[(probe_t, snd)]
            n_arr = df["n"].values

            ax.fill_between(
                n_arr, 0, df["det"],
                color="#90A4AE", alpha=0.8,
                label=r"Det. tail $d_R\cdot\gamma$"
            )
            ax.fill_between(
                n_arr, df["det"], df["hw"],
                color=PAL[snd], alpha=0.75,
                label=r"Stat. $\beta$"
            )
            ax.plot(n_arr, df["hw"], color=PAL[snd], lw=1.1,
                    label="Total half-width")

            ax.axhline(
                df["beta"].iloc[-1],
                color=PAL[snd], lw=0.9, ls="--", alpha=0.65,
                label=r"$\beta$ limit"
            )

            for s0, _ in phase_bounds[1:]:
                ax.axvline(s0, color="#455A64", lw=0.6, ls=":", alpha=0.5)

            if row == 0:
                ax.set_title(snd.capitalize(), color=PAL[snd], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"t = {probe_t}", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Observation step  n")
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

            ax.set_xlim(probe_t, df["n"].max())
            ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# ── 5e.  Fig 5: Violation rates (Monte Carlo) ───────────────────────

def plot_violation_rates(pw_df, loc_df, unif_df, delta, probe_times):
    """
    Visualise Monte Carlo violation rates for all soundness notions.

    Parameters
    ----------
    pw_df, loc_df, unif_df : pandas.DataFrame
        Pointwise, local, and uniform violation-rate summaries from ``monte_carlo``.
    delta : float
        The theoretical error level drawn as a dashed reference line.
    probe_times : sequence[int]
        Probe times included in the plot.

    Logic
    -----
    The first row plots pointwise rates over observation horizons. The second row
    shows local per-time rates and the uniform per-run rate.

    Returns
    -------
    matplotlib.figure.Figure
        Monte Carlo violation-rate figure.
    """
    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle("Empirical violation rates  (Monte Carlo,  "
                 f"n_runs = {N_MC})   dashed line = δ = {delta}",
                 fontsize=10)

    # ── Row 0: pointwise violation rate vs n, one subplot per soundness ──
    for col, snd in enumerate(SOUNDNESSES):
        ax = fig.add_subplot(gs[0, col])
        sub = pw_df[pw_df["soundness"] == snd]
        for t in probe_times:
            d = sub[sub["t"] == t]
            ax.plot(d["n"], d["rate"], lw=0.9, label=f"t={t}", alpha=0.8)
        ax.axhline(delta, color="black", lw=1.0, ls="--", label=f"δ={delta}")
        ax.set_title(f"Pointwise rate – {snd}", color=PAL[snd], fontweight="bold")
        ax.set_xlabel("n")
        ax.set_ylabel("P(violation at n)")
        ax.set_yscale("log")
        ax.set_ylim(-0.01, max(delta * 3, pw_df["rate"].max() * 1.1))
        ax.legend().remove()
        # if col == 0:
        #    ax.legend(fontsize=5, ncol=2)

    # ── Row 1 left: local violation rate per (t, soundness) ─────────────
    ax_loc = fig.add_subplot(gs[1, :2])
    t_labels = [f"t={t}" for t in probe_times]
    x = np.arange(len(probe_times))
    width = 0.25
    for i, snd in enumerate(SOUNDNESSES):
        rates = [loc_df[(loc_df["soundness"] == snd) & (loc_df["t"] == t)]["rate"].values
                 for t in probe_times]
        rates = [r[0] if len(r) > 0 else 0.0 for r in rates]
        ax_loc.bar(x + i * width, rates, width, label=snd.capitalize(),
                   color=PAL[snd], alpha=0.8)
    ax_loc.axhline(delta, color="black", lw=1.2, ls="--", label=f"δ={delta}")
    # ax_loc.set_xticks(x + width)
    # ax_loc.set_xticklabels(t_labels, fontsize=7)
    ax_loc.set_ylabel("P(ever violated for t)")
    ax_loc.set_title("Local violation rate  ∀n ≥ t", fontweight="bold")
    ax_loc.legend(fontsize=7)

    # ── Row 1 right: uniform violation rate (one bar per soundness) ──────
    ax_unif = fig.add_subplot(gs[1, 2])
    snd_labels = [s.capitalize() for s in SOUNDNESSES]
    colors = [PAL[s] for s in SOUNDNESSES]
    rates_u = [unif_df[unif_df["soundness"] == s]["rate"].values[0]
               for s in SOUNDNESSES]
    ax_unif.bar(snd_labels, rates_u, color=colors, alpha=0.85)
    ax_unif.axhline(delta, color="black", lw=1.2, ls="--", label=f"δ={delta}")
    ax_unif.set_ylabel("P(ever violated, any t)")
    ax_unif.set_title("Uniform violation rate  ∀t, ∀n ≥ t", fontweight="bold")
    ax_unif.legend(fontsize=7)
    ax_unif.set_ylim(0, max(delta * 2, max(rates_u) * 1.2))

    return fig


import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ====================================================================
# 1.  MATHEMATICAL PRIMITIVES
# ====================================================================

def normalization_factor(r, s):
    """
    Compute the discounted-sum normalisation constant.

    Parameters
    ----------
    r, s : float
        Past and future discount factors.

    Logic
    -----
    The present has weight one, while past and future geometric tails contribute
    ``r/(1-r)`` and ``s/(1-s)`` when nonzero.

    Returns
    -------
    float
        Lambda used to convert discounted sums into discounted averages.
    """
    lam = 1.0
    if r > 0: lam += r / (1.0 - r)
    if s > 0: lam += s / (1.0 - s)
    return lam


def T_min(r, eps, d_R=1.0):
    """
    Compute the smallest start time with sufficiently small past residual.

    Parameters
    ----------
    r : float
        Past discount factor.
    eps : float
        Tolerance target for the residual.
    d_R : float, optional
        Diameter of the value range.

    Logic
    -----
    The closed-form inequality ``r^(T+1)/(1-r) <= eps/d_R`` is solved and rounded
    up to the next integer time.

    Returns
    -------
    int
        Minimal admissible ``T``; zero when the past discount is zero.
    """
    """Smallest T s.t. past residual r^{T+1}/(1-r) <= eps/d_R."""
    if r == 0: return 0
    target = eps * (1.0 - r) / d_R
    if target <= 0: return 0
    return max(0, int(np.ceil(np.log(target) / np.log(r))) - 1)


def tau_star(r, s, eps, T, d_R=1.0):
    """
    Compute the deterministic approximate-monitoring delay horizon.

    Parameters
    ----------
    r, s : float
        Past and future discount factors.
    eps : float
        Approximation tolerance.
    T : int
        First monitored time.
    d_R : float, optional
        Diameter of the monitored value range.

    Logic
    -----
    This implements the closed-form future waiting time after subtracting the past
    residual at time ``T``. An impossible precision returns infinity.

    Returns
    -------
    int or float
        The finite delay horizon, or ``np.inf`` if the tolerance is unattainable.
    """
    """Theorem 12 horizon tau*(R,r,s,eps,T), Eq.(9)."""
    if s == 0: return 0
    rhs = 2.0 * eps / d_R
    if r > 0: rhs -= r ** (T + 1) / (1.0 - r)
    inner = rhs * (1.0 - s)
    if inner <= 0: return np.inf
    return int(np.ceil(np.log(inner) / np.log(s))) - 1


def det_tail(t, n, r, s):
    """
    Compute the deterministic tail weight for the RQ4 helper functions.

    Parameters
    ----------
    t, n : int
        Monitored time and observation horizon.
    r, s : float
        Past and future discount factors.

    Logic
    -----
    The function mirrors the monitor tail formula and is redefined here to keep the
    notebook-derived RQ4 block self-contained.

    Returns
    -------
    float
        Deterministic unseen-tail weight.
    """
    """gamma^{r,s}_{t,n} = r^{t+1}/(1-r) + s^{n-t+1}/(1-s), Eq.(7)."""
    g = 0.0
    if r > 0: g += r ** (t + 1) / (1.0 - r)
    if s > 0: g += s ** (n - t + 1) / (1.0 - s)
    return g


def weight_norm_sq(t, n, r, s):
    """
    Compute the squared norm of finite discounted weights.

    Parameters
    ----------
    t, n : int
        Monitored time and observation horizon.
    r, s : float
        Past and future discount factors.

    Logic
    -----
    Closed-form geometric sums are used for past and present/future squared weights.

    Returns
    -------
    float
        Finite-horizon value of omega.
    """
    """omega^{r,s}_{t,n} = sum r^{2i} + sum s^{2i}."""
    past = r ** 2 * (1.0 - r ** (2 * t)) / (1.0 - r ** 2) if (r > 0 and t > 0) else 0.0
    fut = (1.0 - s ** (2 * (n - t + 1))) / (1.0 - s ** 2)
    return past + fut


# ====================================================================
# 2.  STATISTICAL ERROR BOUNDS
# ====================================================================

def pe_bound(t, n, r, s, sigma, delta):
    """
    Compute the pointwise Hoeffding--Azuma half-width.

    Parameters
    ----------
    t, n, r, s, sigma, delta
        Time index, horizon, discounts, sub-Gaussian proxy, and error level.

    Logic
    -----
    The fixed-sample bound is evaluated from the squared weight norm.

    Returns
    -------
    float
        Pointwise statistical half-width.
    """
    """Pointwise PE: sqrt(2 sigma^2 omega log(2/delta)). Hoeffding-Azuma."""
    omega = weight_norm_sq(t, n, r, s)
    return np.sqrt(2.0 * sigma ** 2 * omega * np.log(2.0 / delta))


def le_bound(t, n, r, s, sigma, delta):
    """
    Compute the local anytime stitched half-width.

    Parameters
    ----------
    t, n, r, s, sigma, delta
        Time index, horizon, discounts, sub-Gaussian proxy, and error level.

    Logic
    -----
    The Howard-style stitched expression is evaluated with the finite weight norm.

    Returns
    -------
    float
        Local statistical half-width.
    """
    """Local LE: Howard et al. [24] stitched sub-Gaussian boundary."""
    k1 = 2 ** 0.25 + 2 ** (-0.25) / np.sqrt(2.0)
    omega = weight_norm_sq(t, n, r, s)
    V = sigma ** 2 * omega
    log2V = np.log2(max(V, 2.0))
    inner = 2.0 * np.log(max(log2V, 1.0)) + 1.0 + np.log(2.0 * np.pi ** 2 / (6.0 * delta))
    return k1 * np.sqrt(V * max(inner, 1e-12))


def ue_bound(t, n, r, s, sigma, delta):
    """
    Compute the uniform half-width via time-index delta deflation.

    Parameters
    ----------
    t, n, r, s, sigma, delta
        Time index, horizon, discounts, sub-Gaussian proxy, and global error level.

    Logic
    -----
    The global delta is split as ``6 delta/(pi^2 (t+1)^2)`` before applying the
    local stitched bound.

    Returns
    -------
    float
        Uniform statistical half-width.
    """
    """Uniform UE: LE with delta_t = 6*delta/(pi^2*(t+1)^2)."""
    delta_t = 6.0 * delta / (np.pi ** 2 * (t + 1) ** 2)
    return le_bound(t, n, r, s, sigma, delta_t)


_BOUND = {"pointwise": pe_bound, "local": le_bound, "uniform": ue_bound}


def error_decomp(t, n, r, s, sigma, delta, d_R, soundness="pointwise"):
    """
    Return the total, statistical, and deterministic uncertainty widths.

    Parameters
    ----------
    t, n, r, s, sigma, delta, d_R
        Time index, horizon, discounts, statistical parameters, and range diameter.
    soundness : str, optional
        Bound type selected from ``_BOUND``.

    Logic
    -----
    The selected statistical bound is added to ``d_R`` times the deterministic tail.

    Returns
    -------
    tuple[float, float, float]
        Total, statistical, and deterministic half-widths.
    """
    """(total, stat_err, det_tail_val) half-widths."""
    det = d_R * det_tail(t, n, r, s)
    stat = _BOUND[soundness](t, n, r, s, sigma, delta)
    return stat + det, stat, det


# =====================================================================
# EXPERIMENT PLOTTING HELPERS — STYLE MATCHING THE MONITOR FIGURES
# =====================================================================

COMPONENTS = ["total", "statistical", "tail"]
COMPONENT_PAL = {
    "total": "#263238",
    "statistical": "#1976D2",
    "tail": "#90A4AE",
}
COMPONENT_DASHES = {
    "total": "",
    "statistical": (4, 2),
    "tail": (1, 2),
}

R_PAST = 0.0

SCALE_MODES = ["average", "absolute"]


def _fix_suptitle_layout(g, top=0.88):
    """
    Reserve vertical space for a seaborn FacetGrid title.

    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid
        Grid returned by a seaborn plotting function.
    top : float, optional
        Top subplot boundary passed to ``subplots_adjust``.

    Logic
    -----
    The figure layout is adjusted in place and the same grid is returned for
    call-chaining by the plotting wrappers.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        The adjusted grid.
    """
    g.fig.subplots_adjust(top=top)
    return g


def _scaled_params(r, s, sigma, d_R, scale_mode):
    """
    Scale statistical and deterministic parameters for sum or average mode.

    Parameters
    ----------
    r, s : float
        Discount factors defining the normalisation factor.
    sigma, d_R : float
        Raw sub-Gaussian proxy and range diameter.
    scale_mode : {'average', 'absolute'}
        Whether to divide by lambda or keep absolute discounted sums.

    Logic
    -----
    Average mode divides both uncertainty contributors by the normalisation factor;
    absolute mode leaves them unchanged.

    Returns
    -------
    tuple[float, float]
        Scaled ``sigma`` and ``d_R``.
    """
    if scale_mode == "average":
        lam = normalization_factor(r, s)
        return sigma / lam, d_R / lam
    if scale_mode == "absolute":
        return sigma, d_R
    raise ValueError("scale_mode must be 'average' or 'absolute'.")


def omega_limit(t, r, s):
    """
    Compute the limiting squared weight norm as n tends to infinity.

    Parameters
    ----------
    t : int
        Monitored time.
    r, s : float
        Past and future discount factors.

    Logic
    -----
    The finite past term is kept and the infinite future squared-weight sum is
    computed in closed form.

    Returns
    -------
    float
        Limiting omega value.
    """
    past = (
        r ** 2 * (1.0 - r ** (2 * t)) / (1.0 - r ** 2)
        if (r > 0 and t > 0) else 0.0
    )
    future = 1.0 / (1.0 - s ** 2) if s > 0 else 1.0
    return past + future


def det_tail_limit(t, r, s):
    """
    Compute the limiting deterministic tail as n tends to infinity.

    Parameters
    ----------
    t : int
        Monitored time.
    r, s : float
        Past and future discount factors. The future tail vanishes in the limit.

    Logic
    -----
    Only the unobserved past residual remains after the observation horizon goes to
    infinity.

    Returns
    -------
    float
        Limiting deterministic tail weight.
    """
    return r ** (t + 1) / (1.0 - r) if r > 0 else 0.0


def beta_bound_limit(t, r, s, sigma, delta, soundness):
    """
    Compute the limiting statistical half-width as n tends to infinity.

    Parameters
    ----------
    t : int
        Monitored time.
    r, s : float
        Past and future discount factors.
    sigma : float
        Sub-Gaussian proxy.
    delta : float
        Error level.
    soundness : {'pointwise', 'local', 'uniform'}
        Bound family.

    Logic
    -----
    The finite omega in ``beta_bound`` is replaced by ``omega_limit``. Uniform mode
    again applies the time-index delta split before the local bound.

    Returns
    -------
    float
        Limiting statistical half-width.
    """
    om = omega_limit(t, r, s)
    V_base = sigma ** 2 * om
    V = max(1.0, V_base)

    if soundness == "pointwise":
        return np.sqrt(2.0 * sigma ** 2 * om * np.log(2.0 / delta))

    def _le(d):
        """
        Evaluate the limiting local stitched boundary for one delta value.

        Parameters
        ----------
        d : float
            Error probability used by the local boundary.

        Logic
        -----
        The outer function supplies the limiting variance proxy and clipped scale.

        Returns
        -------
        float
            Limiting local statistical half-width.
        """
        inner = (
                2.0 * np.log(np.log2(V) + 1.0)
                + np.log(2.0 * np.pi ** 2 / (6.0 * d))
        )
        return K1 * np.sqrt(V_base * inner)

    if soundness == "local":
        return _le(delta)

    if soundness == "uniform":
        delta_t = 6.0 * delta / (np.pi ** 2 * (t + 1) ** 2)
        return _le(delta_t)

    raise ValueError("Unknown soundness.")


def bound_decomposition(t, n, r, s, sigma, delta, d_R, soundness):
    """
    Decompose a finite-horizon uncertainty half-width.

    Parameters
    ----------
    t, n, r, s, sigma, delta, d_R, soundness
        Time, horizon, discounts, uncertainty parameters, and soundness level.

    Logic
    -----
    The statistical term comes from ``beta_bound`` and the deterministic term from
    ``det_tail`` multiplied by the range diameter.

    Returns
    -------
    dict[str, float]
        Entries ``total``, ``statistical``, and ``tail``.
    """
    stat = beta_bound(t, n, r, s, sigma, delta, soundness)
    tail = d_R * det_tail(t, n, r, s)
    return {"total": stat + tail, "statistical": stat, "tail": tail}


def bound_decomposition_limit(t, r, s, sigma, delta, d_R, soundness):
    """
    Decompose the limiting uncertainty half-width.

    Parameters
    ----------
    t, r, s, sigma, delta, d_R, soundness
        Time, discounts, uncertainty parameters, and soundness level.

    Logic
    -----
    This is the infinite-horizon analogue of ``bound_decomposition``.

    Returns
    -------
    dict[str, float]
        Entries ``total``, ``statistical``, and ``tail``.
    """
    stat = beta_bound_limit(t, r, s, sigma, delta, soundness)
    tail = d_R * det_tail_limit(t, r, s)
    return {"total": stat + tail, "statistical": stat, "tail": tail}


# =====================================================================
# EXPERIMENT A — LIMIT WIDTH VS FUTURE DISCOUNT
# =====================================================================

def run_limit_discount_experiment(
        t=1,
        r=R_PAST,
        s_grid=None,
        sigmas=(0.1, 1.0),
        delta=DELTA,
        d_R=D_R,
):
    """
    Generate data for the limit-width-vs-discount experiment.

    Parameters
    ----------
    t : int, optional
        Monitored time.
    r : float, optional
        Past discount.
    s_grid : array-like, optional
        Future-discount values; a dense default grid emphasises values near one.
    sigmas : sequence[float], optional
        Raw sub-Gaussian proxies to compare.
    delta : float, optional
        Error probability.
    d_R : float, optional
        Range diameter.

    Logic
    -----
    For every scale mode, future discount, sigma, and soundness level, the limiting
    statistical, tail, and total widths are computed.

    Returns
    -------
    pandas.DataFrame
        Long-form data underlying Figure 11.
    """
    if s_grid is None:
        s_grid = np.concatenate([
            np.linspace(0.001, 0.89, 150),
            np.linspace(0.90, 0.999, 250),
        ])

    rows = []
    for scale_mode in SCALE_MODES:
        for s in s_grid:
            for sigma_raw in sigmas:
                sigma, d_R_scaled = _scaled_params(r, s, sigma_raw, d_R, scale_mode)

                for soundness in SOUNDNESSES:
                    vals = bound_decomposition_limit(
                        t, r, s, sigma, delta, d_R_scaled, soundness
                    )
                    for component, value in vals.items():
                        rows.append(dict(
                            time=t,
                            past_discount=r,
                            future_discount=s,
                            sigma=sigma_raw,
                            delta=delta,
                            scale_mode=scale_mode,
                            soundness=soundness,
                            component=component,
                            value=value,
                        ))
    return pd.DataFrame(rows)


def plot_limit_discount_experiment(df, scale_mode="average"):
    """
    Plot limit uncertainty width as a function of the future discount.

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by ``run_limit_discount_experiment``.
    scale_mode : str, optional
        Scale mode to plot.

    Logic
    -----
    A seaborn FacetGrid separates soundness levels and sigma values, with colour and
    style identifying total/statistical/tail components.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Figure 11 plotting grid.
    """
    sub = df[df["scale_mode"] == scale_mode].copy()

    g = sns.relplot(
        kind="line",
        data=sub,
        x="future_discount",
        y="value",
        hue="component",
        style="component",
        col="soundness",
        row="sigma",
        col_order=SOUNDNESSES,
        row_order=sorted(sub["sigma"].unique()),
        palette=COMPONENT_PAL,
        dashes=COMPONENT_DASHES,
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.25,
    )

    g.set_axis_labels("Future discount $s$", r"Limit half-width $(n\to\infty)$")
    g.set_titles(row_template="{row_name}", col_template=r"$\sigma$ = {col_name}")
    g.fig.suptitle(
        f"Limit uncertainty width vs. future discount  [{scale_mode}]",
        y=1.02,
        fontsize=11,
    )
    _fix_suptitle_layout(g)
    return g


# =====================================================================
# EXPERIMENT B — UNIFORM LIMIT WIDTH VS TIME
# =====================================================================

def run_uniform_time_experiment(
        r=R_PAST,
        s_values=(0.95, 0.99, 0.999),
        sigmas=(0.001, 0.01, 0.1, 1.0),
        delta=DELTA,
        d_R=D_R,
        t_max=10_000,
):
    """
    Generate data for the uniform limit-width-vs-time experiment.

    Parameters
    ----------
    r : float, optional
        Past discount.
    s_values : sequence[float], optional
        Future discounts to compare.
    sigmas : sequence[float], optional
        Raw sub-Gaussian proxies.
    delta : float, optional
        Global uniform error level.
    d_R : float, optional
        Range diameter.
    t_max : int, optional
        Largest monitored time included in the grid.

    Logic
    -----
    Only the uniform soundness level is evaluated. For every time, the limiting
    width decomposition is recorded in long form.

    Returns
    -------
    pandas.DataFrame
        Data underlying Figure 12.
    """
    rows = []
    t_grid = np.arange(1, t_max + 1)

    for scale_mode in SCALE_MODES:
        for s in s_values:
            for sigma_raw in sigmas:
                sigma, d_R_scaled = _scaled_params(r, s, sigma_raw, d_R, scale_mode)

                for t in t_grid:
                    vals = bound_decomposition_limit(
                        t, r, s, sigma, delta, d_R_scaled, "uniform"
                    )
                    for component, value in vals.items():
                        rows.append(dict(
                            time=t,
                            past_discount=r,
                            future_discount=s,
                            sigma=sigma_raw,
                            delta=delta,
                            scale_mode=scale_mode,
                            soundness="uniform",
                            component=component,
                            value=value,
                        ))
    return pd.DataFrame(rows)


def plot_uniform_time_experiment(df, scale_mode="average"):
    """
    Plot uniform limiting width over monitored time.

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by ``run_uniform_time_experiment``.
    scale_mode : str, optional
        Scale mode to plot.

    Logic
    -----
    The total limiting half-width is plotted on log-log axes, with style indicating
    sigma and hue indicating the future discount.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Figure 12 plotting grid.
    """
    sub = df[(df["scale_mode"] == scale_mode) & (df["component"] == "total")].copy()
    # sub["future_discount"] = sub["future_discount"].map(lambda s: f"$s={s}$")

    g = sns.relplot(
        kind="line",
        data=sub,
        x="time",
        y="value",
        style="sigma",
        hue="future_discount",
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.25,
    )

    g.set_axis_labels("Time step $t$", r"Limit half-width $(n\to\infty)$")
    # g.set_titles(col_template=r"$s$ = {col_name}")
    g.fig.suptitle(
        f"Uniform limit uncertainty width vs. time  [{scale_mode}]",
        y=1.02,
        fontsize=8,
    )
    _fix_suptitle_layout(g)
    g.set(yscale="log", xscale="log")
    return g


def plot_specific_uniform_time_experiment(df, scale_mode="average", sigma=0.1, future_discount=0.95):
    """
    Plot one selected uniform time curve.

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by ``run_uniform_time_experiment``.
    scale_mode : str, optional
        Scale mode to filter.
    sigma : float, optional
        Raw sigma value to filter.
    future_discount : float, optional
        Future discount value to filter.

    Logic
    -----
    The helper extracts one total-width curve and plots it without the multi-curve
    legend used by the full Figure 12 plot.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Single-configuration uniform-time grid.
    """
    sub = df[(df["scale_mode"] == scale_mode) & (df["component"] == "total") & (df["sigma"] == sigma) & (
                df["future_discount"] == future_discount)].copy()
    # sub["future_discount"] = sub["future_discount"].map(lambda s: f"$s={s}$")

    g = sns.relplot(
        kind="line",
        data=sub,
        x="time",
        y="value",
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.25,
    )

    g.set_axis_labels("Time step $t$", r"Limit half-width $(n\to\infty)$")
    # g.set_titles(col_template=r"$s$ = {col_name}")
    g.fig.suptitle(
        rf"Uniform limit uncertainty width vs. time  [{scale_mode}, $\sigma$={sigma}, $s$={future_discount}]",
        y=1.02,
        fontsize=8,
    )
    _fix_suptitle_layout(g)
    return g


# =====================================================================
# EXPERIMENT C — FINITE-n CONVERGENCE
# =====================================================================

def run_convergence_experiment(
        t=1,
        r=R_PAST,
        s_values=(0.95, 0.99, 0.999),
        sigmas=(0.1, 1.0),
        delta=DELTA,
        d_R=D_R,
        n_values=None,
):
    """
    Generate finite-horizon convergence data for the uncertainty width.

    Parameters
    ----------
    t : int, optional
        Monitored time.
    r : float, optional
        Past discount.
    s_values : sequence[float], optional
        Future discounts to compare.
    sigmas : sequence[float], optional
        Raw sigma values.
    delta : float, optional
        Error level.
    d_R : float, optional
        Range diameter.
    n_values : array-like, optional
        Observation horizons; a logarithmic grid is used by default.

    Logic
    -----
    For each configuration, finite-horizon width components and the corresponding
    limit width are recorded for all observation horizons.

    Returns
    -------
    pandas.DataFrame
        Long-form data underlying Figure 10.
    """
    if n_values is None:
        n_values = np.unique(
            np.round(np.logspace(np.log10(t), np.log10(t + 10_000), 400)).astype(int)
        )

    rows = []
    for scale_mode in SCALE_MODES:
        for s in s_values:
            for sigma_raw in sigmas:
                sigma, d_R_scaled = _scaled_params(r, s, sigma_raw, d_R, scale_mode)

                for n in n_values:
                    for soundness in SOUNDNESSES:
                        vals = bound_decomposition(
                            t, int(n), r, s, sigma, delta, d_R_scaled, soundness
                        )
                        lim_vals = bound_decomposition_limit(
                            t, r, s, sigma, delta, d_R_scaled, soundness
                        )

                        for component, value in vals.items():
                            rows.append(dict(
                                time=t,
                                observations=int(n),
                                past_discount=r,
                                future_discount=s,
                                sigma=sigma_raw,
                                delta=delta,
                                scale_mode=scale_mode,
                                soundness=soundness,
                                component=component,
                                value=value,
                                curve="finite",
                            ))

                        rows.append(dict(
                            time=t,
                            observations=int(n),
                            past_discount=r,
                            future_discount=s,
                            sigma=sigma_raw,
                            delta=delta,
                            scale_mode=scale_mode,
                            soundness=soundness,
                            component="limit",
                            value=lim_vals["total"],
                            curve="limit",
                        ))
    return pd.DataFrame(rows)


def plot_convergence_experiment(df, scale_mode="average", cap=None, log_y=False):
    """
    Plot convergence of finite-horizon widths to their limit.

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by ``run_convergence_experiment``.
    scale_mode : str, optional
        Scale mode to plot.
    cap : float, optional
        Optional upper clipping value for visual readability.
    log_y : bool, optional
        Whether to use a logarithmic y-axis.

    Logic
    -----
    A FacetGrid separates soundness levels and sigma values. The x-axis is always
    log-scaled to show early and late horizons in one plot.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Figure 10 plotting grid.
    """
    sub = df[df["scale_mode"] == scale_mode].copy()
    if cap is not None:
        sub["value"] = sub["value"].clip(upper=cap)

    sub["future_discount"] = sub["future_discount"].map(lambda s: f"$s={s}$")

    g = sns.relplot(
        kind="line",
        data=sub,
        x="observations",
        y="value",
        hue="future_discount",
        style="component",
        col="soundness",
        row="sigma",
        col_order=SOUNDNESSES,
        row_order=sorted(df["sigma"].unique()),
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.25,
    )

    g.set(xscale="log")
    if log_y:
        g.set(yscale="log")

    g.set_axis_labels("Observation step $n$", "Half-width")
    g.set_titles(col_template="{col_name}", row_template=r"$\sigma$ = {row_name}")

    cap_text = "" if cap is None else f", capped at {cap}"
    scale_text = "log-log" if log_y else "log-linear"
    g.fig.suptitle(
        f"Convergence of uncertainty width  [{scale_mode}, {scale_text}{cap_text}]",
        y=1.02,
        fontsize=11,
    )
    _fix_suptitle_layout(g)
    return g


# =====================================================================
# EXPERIMENT D — LIMIT RATE VS 1-s
# =====================================================================

def run_limit_rate_experiment(
        t=1,
        r=R_PAST,
        sigmas=(0.1, 0.25, 0.5, 1.0),
        deltas=(0.2, 0.1, 0.05),
        d_R=D_R,
        resolution=3000,
):
    """
    Generate auxiliary data for the limit-rate-vs-1-s experiment.

    Parameters
    ----------
    t : int, optional
        Monitored time.
    r : float, optional
        Past discount.
    sigmas, deltas : sequence[float], optional
        Statistical parameters to compare.
    d_R : float, optional
        Range diameter.
    resolution : int, optional
        Number of grid points for ``1-s``.

    Logic
    -----
    The future discount is swept through ``s = 1 - k/resolution`` and limiting width
    components are recorded for each configuration.

    Returns
    -------
    pandas.DataFrame
        Long-form limit-rate data.
    """
    rows = []
    for scale_mode in SCALE_MODES:
        for sigma_raw in sigmas:
            for delta in deltas:
                for k in range(1, resolution):
                    one_minus_s = k / resolution
                    s = 1.0 - one_minus_s

                    sigma, d_R_scaled = _scaled_params(r, s, sigma_raw, d_R, scale_mode)

                    for soundness in SOUNDNESSES:
                        vals = bound_decomposition_limit(
                            t, r, s, sigma, delta, d_R_scaled, soundness
                        )
                        for component, value in vals.items():
                            rows.append(dict(
                                time=t,
                                past_discount=r,
                                future_discount=s,
                                one_minus_discount=one_minus_s,
                                sigma=sigma_raw,
                                delta=delta,
                                scale_mode=scale_mode,
                                soundness=soundness,
                                component=component,
                                value=value,
                            ))
    return pd.DataFrame(rows)


def plot_limit_rate_experiment(df, soundness="pointwise", scale_mode="average"):
    """
    Plot limiting width as a function of one minus the future discount.

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by ``run_limit_rate_experiment``.
    soundness : str, optional
        Soundness level to plot.
    scale_mode : str, optional
        Scale mode to plot.

    Logic
    -----
    The same data are repeated under linear, log-x, and log-log axis scalings to
    make the asymptotic rate easier to inspect.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Limit-rate comparison grid.
    """
    sub = df[
        (df["soundness"] == soundness)
        & (df["scale_mode"] == scale_mode)
        & (df["component"] == "total")
        ].copy()

    scale_types = ["linear", "log-x", "log-log"]
    plot_df = pd.concat(
        [sub.assign(axis_scale=s) for s in scale_types],
        ignore_index=True,
    )

    g = sns.relplot(
        kind="line",
        data=plot_df,
        x="one_minus_discount",
        y="value",
        hue="delta",
        style="sigma",
        col="axis_scale",
        col_order=scale_types,
        style_order=sorted(sub["sigma"].unique()),
        facet_kws={"sharex": False, "sharey": False},
        height=3,
        aspect=1.25,
    )

    for row in g.axes:
        row[0].set_xscale("linear")
        row[0].set_yscale("linear")

        row[1].set_xscale("log")
        row[1].set_yscale("linear")

        row[2].set_xscale("log")
        row[2].set_yscale("log")

    g.set_axis_labels(r"$1-s$", r"Limit total half-width $(n\to\infty)$")
    g.set_titles(row_template=r"$\sigma$ = {row_name}", col_template="{col_name}")
    g.fig.suptitle(
        f"Limit uncertainty rate vs. discount factor  [{soundness}, {scale_mode}]",
        y=1.02,
        fontsize=11,
    )
    _fix_suptitle_layout(g)
    return g


# =====================================================================
# 6.  EXPERIMENT SCRIPT WRAPPER
# =====================================================================

FIGURE_NAMES = {
    "process": "figure_05_beta_process.pdf",
    "heatmap": "figure_06_verdict_heatmap.pdf",
    "ci_evolution": "figure_07_uncertainty_evolution.pdf",
    "width_decomposition": "figure_08_width_decomposition.pdf",
    "ci_first_verdict": "figure_09a_ci_at_first_verdict.pdf",
    "ci_end": "figure_09b_ci_at_end.pdf",
    "convergence": "figure_10_convergence.pdf",
    "limit_discount": "figure_11_limit_discount.pdf",
    "uniform_time": "figure_12_uniform_time.pdf",
    "uniform_time_specific": "figure_12_uniform_time_specific.pdf",
    "mc_violation_rates": "figure_table_02_mc_violation_rates.pdf",
}


def _ensure_dir(path):
    """
    Create an output directory if needed.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory requested by the caller.

    Logic
    -----
    The path is converted to ``Path`` and created with ``parents=True``.

    Returns
    -------
    pathlib.Path
        Normalised directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig_or_grid, path, dpi=500):
    """
    Save and close a matplotlib or seaborn figure object.

    Parameters
    ----------
    fig_or_grid : matplotlib.figure.Figure or seaborn.axisgrid.FacetGrid
        Object to save. If a FacetGrid is provided, its ``fig`` attribute is used.
    path : str or pathlib.Path
        Destination file path.
    dpi : int, optional
        Rasterisation resolution used by ``savefig``.

    Logic
    -----
    The figure is written with a tight bounding box and closed immediately to avoid
    memory growth during the full workflow.

    Returns
    -------
    pathlib.Path
        Saved path.
    """
    path = Path(path)
    fig = getattr(fig_or_grid, "fig", fig_or_grid)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def _flatten_monitor_results(results):
    """
    Flatten monitor-result dictionaries into one CSV-ready table.

    Parameters
    ----------
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor outputs returned by ``run_monitor``.

    Logic
    -----
    Each per-probe DataFrame is copied and concatenated in row order. An empty input
    returns an empty DataFrame.

    Returns
    -------
    pandas.DataFrame
        Long-form monitor grid.
    """
    frames = []
    for (_, _), df in results.items():
        frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarise_first_verdicts(results):
    """
    Summarise the first decisive verdict for each monitored time and soundness.

    Parameters
    ----------
    results : dict[tuple[int, str], pandas.DataFrame]
        Monitor outputs returned by ``run_monitor``.

    Logic
    -----
    For each trace, the first row with verdict not equal to zero is extracted; if no
    such row exists, the time is marked as unreleased.

    Returns
    -------
    pandas.DataFrame
        One summary row per ``(t, soundness)`` pair.
    """
    rows = []
    for (t, snd), df in results.items():
        decisive = df[df["verdict"] != 0]
        first = decisive.iloc[0] if len(decisive) else None
        rows.append(dict(
            t=t,
            soundness=snd,
            violation_rate=float(df["violation"].mean()),
            first_decisive_n=(int(first["n"]) if first is not None else np.nan),
            first_verdict=(int(first["verdict"]) if first is not None else 0),
            s_true=float(df["s_true"].iloc[0]) if len(df) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values(["t", "soundness"])


def _mean_pm_std_plain(x, digits=3):
    """
    Format mean plus standard deviation for the variant-aware table.

    Parameters
    ----------
    x : array-like
        Values to aggregate.
    digits : int, optional
        Number of decimal places.

    Logic
    -----
    The implementation mirrors ``_mean_pm_std`` and is kept with the table variant
    helper to preserve the notebook-derived block structure.

    Returns
    -------
    str
        LaTeX-ready ``mean plus/minus sd`` string or ``--`` if no values are available.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return "--"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{digits}f} $\\pm$ {sd:.{digits}f}"


def export_mc_metrics_latex_by_variant(metrics_df, path, digits=3):
    """
    Write the variant-aware Monte Carlo summary table used by the script.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-run Monte Carlo metrics. If present, ``variant`` separates parameter
        blocks such as ``(a,b)`` and ``(10a,10b)``.
    path : str or pathlib.Path
        Destination ``.tex`` file.
    digits : int, optional
        Decimal places in formatted means and standard deviations.

    Logic
    -----
    Rows are grouped first by variant and then by soundness. Coverage, release, and
    verdict-quality metrics are formatted for a booktabs LaTeX table.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        Rendered table data and the LaTeX source written to ``path``.
    """
    cols = [
        ("coverage_violation_rate", "Interval viol."),
        ("any_coverage_violation", "Any interval"),
        ("release_fraction", "Released"),
        ("mean_delay", "Delay"),
        ("incorrect_first_rate", "Wrong rate"),
        ("any_incorrect", "Any wrong"),
    ]
    variant_order = list(dict.fromkeys(metrics_df["variant"].tolist())) if "variant" in metrics_df else ["MC"]
    soundness_order = [s for s in SOUNDNESSES if s in set(metrics_df["soundness"])]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Aggregate statistics of the Monte Carlo experiments. Coverage: Interval viol. is the fraction of statistical uncertainty interval violations; Any interval is the fraction of runs with at least one statistical uncertainty interval violation. Release: Released is the fraction of monitored time indices that receive a decisive verdict; Delay is the average release delay. Verdicts: Wrong rate is the fraction of incorrect first verdicts among released verdicts; Any wrong is the fraction of runs with at least one incorrect verdict.}")
    lines.append(r"\label{tab:mc_metrics}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$(a,b)$ & Sound. & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Release} & \multicolumn{2}{c}{Verdicts} \\")
    lines.append(r" & & Interval viol. & Any interval & Released & Delay & Wrong rate & Any wrong \\")
    lines.append(r"\midrule")
    rows = []
    for variant in variant_order:
        vsub = metrics_df[metrics_df["variant"] == variant] if "variant" in metrics_df else metrics_df
        first = True
        for snd in soundness_order:
            sub = vsub[vsub["soundness"] == snd]
            vals = [_mean_pm_std_plain(sub[col], digits) for col, _ in cols]
            setting = variant if first else ""
            lines.append(" & ".join([setting, snd.capitalize()] + vals) + r" \\")
            rows.append([variant, snd.capitalize()] + vals)
            first = False
        if variant != variant_order[-1]:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines) + "\n"
    Path(path).write_text(latex)
    return pd.DataFrame(rows, columns=["Setting", "Soundness"] + [name for _, name in cols]), latex


def _phase_means(phases):
    """
    Extract Beta means from a phase specification.

    Parameters
    ----------
    phases : sequence[tuple[int, float, float]]
        Tuples ``(length, alpha, beta)``.

    Logic
    -----
    Each phase mean is computed as ``alpha/(alpha+beta)``.

    Returns
    -------
    list[float]
        Per-phase latent means.
    """
    return [a / (a + b) for _, a, b in phases]


def _phase_bounds(phases):
    """
    Compute inclusive index bounds for each phase.

    Parameters
    ----------
    phases : sequence[tuple[int, float, float]]
        Tuples ``(length, alpha, beta)``.

    Logic
    -----
    The cumulative phase length determines the first and last time index of each
    piecewise-stationary block.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive ``(start, end)`` bounds.
    """
    bounds = []
    pos = 0
    for length, _, _ in phases:
        bounds.append((pos, pos + length - 1))
        pos += length
    return bounds


def run_single_simulation(output_dir, seed, sim_probe_stride=1, save_csv=True):
    """
    Run the single-realisation appendix workflow with PHASES_SIM.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory for generated figures and optional CSV files.
    seed : int
        Random seed for the displayed Beta trajectory.
    sim_probe_stride : int, optional
        Stride between monitored probe times in the single simulation.
    save_csv : bool, optional
        Whether to write reusable CSV files.

    Logic
    -----
    The function installs the high-variance SIM parameters, recomputes sigma and the
    average normalisation, runs the monitor, writes CSV data, and saves Figures 5--9.

    Returns
    -------
    dict[tuple[int, str], pandas.DataFrame]
        Monitor results for the displayed simulation.
    """
    """Run the one-realisation appendix simulation using PHASES_SIM."""
    global PHASES, N_STEPS, SIGMA, SIGMA_AVR, D_R_AVR, PROBE_TIMES, R_PAST, S_FUT, LAMBDA
    PHASES = PHASES_SIM
    N_STEPS = sum(p[0] for p in PHASES)
    PROBE_TIMES = list(range(1, N_STEPS + 1, sim_probe_stride))
    R_PAST = 0.95
    S_FUT = 0.95
    LAMBDA = normalization_factor(R_PAST, S_FUT)
    SIGMA = max(beta_subgaussian_proxy(p[1], p[2]) for p in PHASES)
    SIGMA_AVR = SIGMA / LAMBDA
    D_R_AVR = D_R / LAMBDA

    scale_denom = LAMBDA
    r_inf_mon = R_INF / scale_denom
    r_sup_mon = R_SUP / scale_denom
    d_r_mon = D_R / scale_denom

    rng = np.random.default_rng(seed)
    X, P, phase_bounds, phase_means = generate_process(PHASES, rng)

    print("Running one-realisation monitor with SIM parameters")
    print(f"  phases={[(a, b) for _, a, b in PHASES]}, sigma={SIGMA:.6f}, lambda={LAMBDA:.6f}")
    results = run_monitor(
        X, P,
        r=R_PAST, s=S_FUT,
        sigma=SIGMA, delta=DELTA, d_R=d_r_mon,
        L=TARGET_L, U=TARGET_U, eps=EPSILON,
        probe_times=PROBE_TIMES,
        phases=PHASES,
        soundnesses=SOUNDNESSES,
        scale_denom=scale_denom,
        R_inf=r_inf_mon,
        R_sup=r_sup_mon,
    )

    if save_csv:
        _flatten_monitor_results(results).to_csv(output_dir / "stochastic_monitor_grid.csv", index=False)
        _summarise_first_verdicts(results).to_csv(output_dir / "stochastic_monitor_first_verdicts.csv", index=False)
        pd.DataFrame({"time": np.arange(len(X)), "observation": X, "latent_mean": P}).to_csv(
            output_dir / "stochastic_beta_process.csv", index=False
        )

    probe_plot = [t for t in (200, 400) if t in PROBE_TIMES]
    if not probe_plot:
        probe_plot = PROBE_TIMES[: min(2, len(PROBE_TIMES))]

    _save_figure(plot_process(
        X, P, phase_bounds, phase_means,
        TARGET_L, TARGET_U, EPSILON,
        r=R_PAST, s=S_FUT,
        phases=PHASES,
        scale_denom=scale_denom,
    ), output_dir / FIGURE_NAMES["process"], dpi=500)

    _save_figure(plot_verdict_heatmap(results, PROBE_TIMES, phase_bounds),
                 output_dir / FIGURE_NAMES["heatmap"], dpi=500)

    _save_figure(plot_ci_evolution(results, probe_plot, TARGET_L, TARGET_U, EPSILON, phase_bounds),
                 output_dir / FIGURE_NAMES["ci_evolution"], dpi=500)

    _save_figure(plot_bound_decomposition(results, probe_plot, phase_bounds),
                 output_dir / FIGURE_NAMES["width_decomposition"], dpi=500)

    _save_figure(plot_CI_process(
        X, P, phase_bounds, phase_means,
        TARGET_L, TARGET_U, EPSILON,
        r=R_PAST, s=S_FUT,
        phases=PHASES,
        results=results,
        scale_denom=scale_denom,
        min_interval=False,
    ), output_dir / FIGURE_NAMES["ci_first_verdict"], dpi=500)

    _save_figure(plot_CI_process(
        X, P, phase_bounds, phase_means,
        TARGET_L, TARGET_U, EPSILON,
        r=R_PAST, s=S_FUT,
        phases=PHASES,
        results=results,
        scale_denom=scale_denom,
        min_interval=True,
    ), output_dir / FIGURE_NAMES["ci_end"], dpi=500)

    return results


def run_mc_workflow(output_dir, n_mc, seed, quick=False, include_sim_block=True, save_csv=True):
    """
    Run the Monte Carlo workflow for both Table 2 parameter blocks by default.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory for generated tables, CSV files, and MC plots.
    n_mc : int
        Number of independent Monte Carlo repetitions.
    seed : int
        Random seed for the Monte Carlo generator.
    quick : bool, optional
        Whether to use a coarser probe grid and fewer default repetitions.
    include_sim_block : bool, optional
        Whether to include the high-variance ``(a,b)`` block before the
        lower-variance ``(10a,10b)`` block. The default reproduces Table 2.
    save_csv : bool, optional
        Whether to write reusable CSV files.

    Logic
    -----
    For each requested phase variant, the function recomputes sigma, runs the Monte
    Carlo study, writes raw rate/metric CSVs, exports Table 2, and saves the
    violation-rate diagnostic plot. By default both the ``(a,b)`` and
    ``(10a,10b)`` variants are included, matching the paper table.

    Returns
    -------
    pandas.DataFrame
        Raw per-run Monte Carlo metrics across all included variants.
    """
    """Run the Monte Carlo appendix experiment for the requested variants."""
    global PHASES, N_STEPS, SIGMA, SIGMA_AVR, D_R_AVR, PROBE_TIMES, R_PAST, S_FUT, LAMBDA, N_MC
    R_PAST = 0.95
    S_FUT = 0.95
    LAMBDA = normalization_factor(R_PAST, S_FUT)
    scale_denom = LAMBDA
    r_inf_mon = R_INF / scale_denom
    r_sup_mon = R_SUP / scale_denom
    d_r_mon = D_R / scale_denom
    N_MC = n_mc

    variants = []
    if include_sim_block:
        variants.append(("$(a,b)$", PHASES_SIM))
    variants.append(("$(10a,10b)$", PHASES_MC))

    all_metrics = []
    last_pw = last_loc = last_unif = None
    last_probe_plot = None

    for label, phases in variants:
        PHASES = phases
        N_STEPS = sum(p[0] for p in PHASES)
        if quick:
            probe_times = list(range(1, N_STEPS + 1, 25))
        else:
            probe_times = list(range(1, N_STEPS + 1, 1))
        PROBE_TIMES = probe_times
        SIGMA = max(beta_subgaussian_proxy(p[1], p[2]) for p in PHASES)
        SIGMA_AVR = SIGMA / LAMBDA
        D_R_AVR = D_R / LAMBDA

        print(f"Running Monte Carlo with {label} parameters")
        print(f"  phases={[(a, b) for _, a, b in PHASES]}, sigma={SIGMA:.6f}, runs={n_mc}, probes={len(probe_times)}")

        pw_df, loc_df, unif_df, mc_metrics = monte_carlo(
            r=R_PAST, s=S_FUT,
            sigma=SIGMA, delta=DELTA, d_R=d_r_mon,
            L=TARGET_L, U=TARGET_U, eps=EPSILON,
            phases=PHASES,
            probe_times=probe_times,
            soundnesses=SOUNDNESSES,
            n_runs=n_mc,
            seed=seed,
            scale_denom=scale_denom,
            R_inf=r_inf_mon,
            R_sup=r_sup_mon,
        )
        for df in (pw_df, loc_df, unif_df, mc_metrics):
            df["variant"] = label
        all_metrics.append(mc_metrics)
        last_pw, last_loc, last_unif = pw_df, loc_df, unif_df
        last_probe_plot = [t for t in (200, 400) if t in probe_times] or probe_times[: min(2, len(probe_times))]

        safe_label = "mc" if "10" in label else "sim"
        if save_csv:
            pw_df.to_csv(output_dir / f"mc_{safe_label}_pointwise_violation_rates.csv", index=False)
            loc_df.to_csv(output_dir / f"mc_{safe_label}_local_violation_rates.csv", index=False)
            unif_df.to_csv(output_dir / f"mc_{safe_label}_uniform_violation_rates.csv", index=False)
            mc_metrics.to_csv(output_dir / f"mc_{safe_label}_metrics_raw.csv", index=False)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    if save_csv and len(metrics):
        metrics.to_csv(output_dir / "table_02_mc_metrics_raw.csv", index=False)
    mc_table, mc_latex = export_mc_metrics_latex_by_variant(
        metrics,
        output_dir / "table_02_mc_metrics.tex",
        digits=3,
    )
    mc_table.to_csv(output_dir / "table_02_mc_metrics_summary.csv", index=False)

    if last_pw is not None:
        _save_figure(plot_violation_rates(last_pw, last_loc, last_unif, DELTA, last_probe_plot),
                     output_dir / FIGURE_NAMES["mc_violation_rates"], dpi=500)
    return metrics


def run_rq4_workflow(output_dir, quick=False, save_csv=True):
    """
    Run the Appendix F.4 uncertainty-width experiments.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory for generated plots and optional CSV files.
    quick : bool, optional
        Whether to shorten the grids for a smoke test.
    save_csv : bool, optional
        Whether to write the long-form experiment data.

    Logic
    -----
    The wrapper runs the convergence, limit-discount, and uniform-time experiments
    and saves Figures 10--12 using the notebook plotting functions.

    Returns
    -------
    None
        All artefacts are written to ``output_dir``.
    """
    """Run Appendix F.4 width experiments using the notebook plotting logic."""
    global R_PAST
    R_PAST = 0.0

    print("Running Appendix F.4 uncertainty-width experiments")
    convergence_n = None
    uniform_t_max = 10_000
    if quick:
        convergence_n = np.unique(np.round(np.logspace(0, np.log10(2_000), 150)).astype(int))
        uniform_t_max = 2_000

    convergence_df = run_convergence_experiment(n_values=convergence_n)
    if save_csv:
        convergence_df.to_csv(output_dir / "figure_10_convergence_data.csv", index=False)
    g = plot_convergence_experiment(convergence_df)
    _save_figure(g, output_dir / FIGURE_NAMES["convergence"], dpi=500)

    limit_df = run_limit_discount_experiment()
    if save_csv:
        limit_df.to_csv(output_dir / "figure_11_limit_discount_data.csv", index=False)
    g = plot_limit_discount_experiment(limit_df)
    _save_figure(g, output_dir / FIGURE_NAMES["limit_discount"], dpi=500)

    uniform_df = run_uniform_time_experiment(t_max=uniform_t_max)
    if save_csv:
        uniform_df.to_csv(output_dir / "figure_12_uniform_time_data.csv", index=False)
    g = plot_uniform_time_experiment(uniform_df)
    _save_figure(g, output_dir / FIGURE_NAMES["uniform_time"], dpi=500)
    g = plot_specific_uniform_time_experiment(uniform_df)
    _save_figure(g, output_dir / FIGURE_NAMES["uniform_time_specific"], dpi=500)


def parse_args(argv=None):
    """
    Parse command-line arguments for the reproduction script.

    Parameters
    ----------
    argv : sequence[str], optional
        Argument list to parse. If omitted, ``argparse`` reads from ``sys.argv``.

    Logic
    -----
    The parser exposes output location, Monte Carlo size, random seed, smoke-test
    mode, and switches for skipping optional workflow blocks.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the stochastic discounted-sum monitoring appendix plots and Monte Carlo table."
    )
    parser.add_argument(
        "--output-dir",
        default="experimental-results/stochastic-monitoring",
        help="Directory for generated PDFs, CSVs, and tables.",
    )
    parser.add_argument("--n-mc", type=int, default=N_MC, help="Number of Monte Carlo repetitions.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test with fewer MC runs/probes and shorter RQ4 grids.")
    parser.add_argument("--skip-mc", action="store_true", help="Skip the Monte Carlo table/violation-rate plot.")
    parser.add_argument("--skip-rq4", action="store_true", help="Skip Appendix F.4 Figures 10-12.")
    parser.add_argument("--mc-only-low-variance", action="store_true",
                        help="Only generate the lower-variance (10a,10b) Monte Carlo block.")
    parser.add_argument("--sim-probe-stride", type=int, default=1, help="Stride for monitored SIM probe times.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write reusable CSV data.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Execute the full stochastic monitoring reproduction workflow.

    Parameters
    ----------
    argv : sequence[str], optional
        Command-line argument list for testing or programmatic invocation.

    Logic
    -----
    The output directory is prepared, then the script runs the single simulation,
    the Monte Carlo workflow unless skipped, and the Appendix F.4 width experiments
    unless skipped.

    Returns
    -------
    None
        Results are written to disk and progress is printed to stdout.
    """
    args = parse_args(argv)
    output_dir = _ensure_dir(args.output_dir)
    n_mc = min(args.n_mc, 10) if args.quick and args.n_mc == N_MC else args.n_mc
    save_csv = not args.no_csv

    print(f"Writing outputs to {output_dir}")
    run_single_simulation(output_dir, seed=args.seed, sim_probe_stride=args.sim_probe_stride, save_csv=save_csv)

    if not args.skip_mc:
        run_mc_workflow(
            output_dir,
            n_mc=n_mc,
            seed=args.seed,
            quick=args.quick,
            include_sim_block=not args.mc_only_low_variance,
            save_csv=save_csv,
        )
    if not args.skip_rq4:
        run_rq4_workflow(output_dir, quick=args.quick, save_csv=save_csv)

    print("Done.")


if __name__ == "__main__":
    main()
