#!/usr/bin/env python3
"""
Streaming monitor for discounted sums with past/future discount factors.

Implements the algorithm described in the paper section "Monitor Construction":
- Maintain a global past-sum accumulator Psum with update: Psum <- x_t + r * Psum
- Maintain up to tau active "slots" (Past_i, Future_i, pos_i)
- For each new time t >= T, allocate a free slot and start monitoring time pos=t
- Each step updates each active slot's Future component and computes an uncertainty interval
- If the uncertainty interval is fully inside I -> verdict 1
  If the uncertainty interval is disjoint from I -> verdict 0
  Otherwise -> inconclusive (no verdict yet)

Notes:
- This code uses bounds m = inf R and M = sup R, i.e., assumes R is bounded.
- tau is computed from the formula in the draft; you should ensure the expression inside log_s is in (0,1)
  if you expect tau to be positive when 0 < s < 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

class Verdict(Enum):
    BOT = 0
    TOP = 1


@dataclass
class Slot:
    in_use: bool = False
    past: float = 0.0
    future: float = 0.0
    pos: int = 0  # target time point for this slot


def log_base(x: float, base: float) -> float:
    if x <= 0:
        raise ValueError(f"log_base: x must be > 0, got {x}")
    if base <= 0 or base == 1.0:
        raise ValueError(f"log_base: base must be > 0 and != 1, got {base}")
    return math.log(x) / math.log(base)

def aux_omega(r,s,t,nl,nu):
    num1 = (r**2)*(1-np.pow(r,2*(t-nl)))
    num2 = (1-np.pow(s,2*(nu-t+1)))
    den1 = 1-r*r
    den2 = 1-s*s
    return num1/den1 + num2/den2


def tau_star(r: float, s: float, eps: float, T: int, infR: float, supR: float, use_averages: bool) -> int:
    """
    tau*(r,s,eps,T) = ceil( log_s( 2*(eps + r^{T+1}/(1-r))*(1-s) ) ) - 1
    as in Eq. (approximate-monitors-tau) in the provided draft.

    Caution: for 0 < s < 1, log_s(.) is decreasing and math.log(s) < 0.
    """
    if not (0.0 <= r < 1.0 and 0.0 <= s < 1.0):
        raise ValueError("r and s must be in [0,1).")
    if s == 0.0:
        # With s=0, future discount kills all but present; the formula is not meaningful.
        # You can set tau=0 or 1 depending on how you treat future uncertainty.
        return 0
    avg_normalization_value = 1+(r/(1-r))+(s/(1-s)) if use_averages else 1
    inner = (2*eps/((supR - infR)*avg_normalization_value) - (r ** (T + 1)) / (1.0 - r)) * (1.0 - s)
    if inner <= 0:
        raise ValueError(f"T has to be large enough compared with epsilon, epsilon > r^(T+1)/(1-r), now {eps=} < {r**(T+1)/(1-r)}")

    # The formula expects inner > 0. If inner >= 1 and 0<s<1, log_s(inner) <= 0,
    # which can make tau negative. Clamp at 0 for a usable monitor size.
    val = log_base(inner, s)
    tau = math.ceil(val) - 1
    # print(f"{tau=}")
    return max(0, int(tau))


class WindowAverageMonitor:
    """
    Streaming monitor implementing the algorithm in the "Monitor Construction" subsection.

    Parameters:
      tau: window length (has to be odd)
      R_bounds: (m, M) = (inf R, sup R)
      I: target interval [L,U]
      eps: monitor tolerance
      T: start time (do not allocate slots before t>=T)
      tau: optional override for number of slots; if None, compute tau_star(r,s,eps,T)
    """

    def __init__(
        self,
        r: float,
        s: float,
        R_bounds: Tuple[float, float],
        I: Tuple[float, float],
        eps: float,
        T: int,
        use_averages: bool = False,
        tau: Optional[int] = None,
    ):
        if not (0.0 <= r < 1.0 and 0.0 <= s < 1.0):
            raise ValueError("r and s must be in [0,1).")
        m, M = R_bounds
        if m > M:
            raise ValueError("R_bounds must be (m,M) with m <= M.")
        L, U = I
        if L > U:
            raise ValueError("I must be (L,U) with L <= U.")
        if T < 0:
            raise ValueError("T must be >= 0.")

        self.r = r
        self.s = s
        self.m = m
        self.M = M
        self.L = L
        self.U = U
        self.eps = eps
        self.Lin = L - eps
        self.Lout = L + eps
        self.Uin = U+eps
        self.Uout = U-eps

        self.T = T
        self.use_averages = use_averages
        self.avg_normalization_value = 1+(r/(1-r))+(s/(1-s))
        self.n_active_monitors = 0

        self.tau = tau if tau is not None else tau_star(r, s, eps, T, m, M,use_averages)
        self.slots: List[Slot] = [Slot() for _ in range(self.tau)]
        self.psum: float = 0.0
        self.t: int = -1  # will become 0 on first step

    def _allocate_slot(self, pos: int) -> None:
        for slot in self.slots:
            if not slot.in_use:
                slot.in_use = True
                slot.past = self.psum
                slot.future = 0.0
                slot.pos = pos
                return
        raise RuntimeError(
            "No free slot available. Either increase tau or check that verdicts are being produced as expected."
        )

    def step(self, x_t: float) -> List[Tuple[int, Verdict]]:
        """
        Process one observed value x_t.

        Returns a list of (pos, verdict) for any slots that became decidable at this step.
        """
        self.t += 1
        t = self.t

        if self.use_averages:
            x_t = x_t/self.avg_normalization_value      

        # Update global past sum
        self.psum = x_t + self.r * self.psum
        # print(f"{self.psum=}")


        verdicts: List[Tuple[int, int, float, float]] = []

        # Update active slots, compute uncertainty intervals, and decide
        for slot in self.slots:
            if not slot.in_use:
                continue

            # Future update
            dt = t - slot.pos
            slot.future += x_t * (self.s ** dt)

            # Uncertainty interval (as written in your construction)
            gamma = (self.r ** slot.pos) / (1.0 - self.r) + (self.s ** dt) / (1.0 - self.s)
            base = slot.past + slot.future
            lo = base + gamma * self.m
            hi = base + gamma * self.M

            # Decide
            if lo >= self.Lin and hi <= self.Uin:
                verdicts.append((slot.pos, 1, lo, hi))
                slot.in_use = False
                self.n_active_monitors -= 1
            elif hi < self.Lout or lo > self.Uout:
                verdicts.append((slot.pos, 0, lo, hi))
                slot.in_use = False
                self.n_active_monitors -= 1

        # Activate a new slot at time t if t >= T
        if t >= self.T and self.tau > 0:
            self._allocate_slot(pos=t)
            self.n_active_monitors += 1

        return self.n_active_monitors, verdicts

    def run(self, xs: Iterable[float]) -> List[Tuple[int, int, float, float]]:
        """Convenience method: process a finite iterable and collect all produced verdicts."""
        out: List[Tuple[int, int, float, float]] = []
        for x in xs:
            out.extend(self.step(x))
        return out




@dataclass
class SlotVec:
    in_use: bool = False
    pos: int = -1
    posA: int = -1
    posB: int = -1
    past: Optional[np.ndarray] = None   # shape (4,)
    future: Optional[np.ndarray] = None # shape (4,)

def interval_div(num_lo, num_hi, den_lo, den_hi):
    """
    Safe interval for num/den assuming den_lo > 0 (strict).
    If den_lo <= 0, returns (None, None) to indicate undefined/unbounded.
    """
    if den_lo <= 0:
        return None, None
    # den is positive, so monotone: min at num_lo/den_hi, max at num_hi/den_lo
    return num_lo / den_hi, num_hi / den_lo

def interval_phi(
    accA_lo, accA_hi, seenA_lo, seenA_hi,
    accB_lo, accB_hi, seenB_lo, seenB_hi
):
    rA_lo, rA_hi = interval_div(accA_lo, accA_hi, seenA_lo, seenA_hi)
    rB_lo, rB_hi = interval_div(accB_lo, accB_hi, seenB_lo, seenB_hi)
    if rA_lo is None or rB_lo is None:
        return None, None
    # (A - B) interval: [A_lo - B_hi, A_hi - B_lo]
    return rA_lo - rB_hi, rA_hi - rB_lo


class DiscountedPhiMonitor:
    """
    Monitors phi = accA/seenA - accB/seenB with the same slot-based discounted-sum construction,
    but for a 4-vector stream x_t = [seenA, accA, seenB, accB].
    Used for demographic parity.


    R_bounds is now per-component bounds: (m_vec, M_vec), each shape (4,).
    Typical for indicator increments: m_vec = [0,0,0,0], M_vec = [1,1,1,1].
    """

    def __init__(
        self,
        r: float,
        s: float,
        R_bounds: Tuple[np.ndarray, np.ndarray],
        I: Tuple[float, float],
        eps: float,
        T: int,
        tau: Optional[int] = None,
        synchronous : Optional[bool] = True
    ):
        if not (0.0 <= r < 1.0 and 0.0 <= s < 1.0):
            raise ValueError("r and s must be in [0,1).")
        L, U = I
        if L > U:
            raise ValueError("I must be (L,U) with L <= U.")
        if T < 0:
            raise ValueError("T must be >= 0.")

        m_vec, M_vec = R_bounds
        m_vec = np.asarray(m_vec, dtype=float)
        M_vec = np.asarray(M_vec, dtype=float)
        if m_vec.shape != (4,) or M_vec.shape != (4,):
            raise ValueError("R_bounds must be (m_vec, M_vec) with shape (4,).")
        if np.any(m_vec > M_vec):
            raise ValueError("Need m_vec <= M_vec componentwise.")

        self.r, self.s = r, s
        self.m_vec, self.M_vec = m_vec, M_vec

        self.L, self.U, self.eps = L, U, eps
        self.Lin = L - eps
        self.Lout = L + eps
        self.Uin = U + eps
        self.Uout = U - eps

        self.T = T
        self.tau = tau if tau is not None else tau_star(r, s, eps, T, float(m_vec.min()), float(M_vec.max()), False)
        self.slots: List[SlotVec] = [SlotVec() for _ in range(self.tau)]
        self.psum = np.zeros(4, dtype=float)  # global past discounted sum for each component
        self.t = -1
        self.tA = -1
        self.tB = -1
        self.synchronous = synchronous
        self.n_active_monitors = 0

    def _allocate_slot(self, pos: int) -> None:
        for slot in self.slots:
            if not slot.in_use:
                slot.in_use = True
                slot.pos = pos
                slot.posA = max(self.tA, 0)
                slot.posB = max(self.tB, 0)
                slot.past = self.psum.copy()
                slot.future = np.zeros(4, dtype=float)
                return
        raise RuntimeError("No free slot available. Increase tau or check verdict production.")

    def step(self, x_t: np.ndarray):
        """
        x_t: shape (4,) vector [seenA, accA, seenB, accB].
        Returns: (n_active_monitors, verdicts)
        verdicts: list of (pos, v, phi_lo, phi_hi)
        """
        x_t = np.asarray(x_t, dtype=float)
        if x_t.shape != (4,):
            raise ValueError("x_t must have shape (4,)")

        self.t += 1
        t = self.t

        # Update global past sum (vector)
        if self.synchronous:
            self.psum = x_t + self.r * self.psum
        else:
            group = "A" if x_t[0] == 1 else "B"

            # advance only the relevant clock and update only relevant psum components
            if group == "A":
                self.tA += 1
                self.psum[0:2] = x_t[0:2] + self.r * self.psum[0:2]
            else:
                self.tB += 1
                self.psum[2:4] = x_t[2:4] + self.r * self.psum[2:4]
            


        verdicts = []

        for slot in self.slots:
            if not slot.in_use:
                continue
            if self.synchronous:
                dt = t - slot.pos
                slot.future += x_t * (self.s ** dt)

                gamma = (self.r ** slot.pos) / (1.0 - self.r) + (self.s ** dt) / (1.0 - self.s)
                base = slot.past + slot.future
            else:
                if group == "A":
                    dtA = self.tA - slot.posA
                    slot.future[0:2] += x_t[0:2] * (self.s ** dtA)
                else:
                    dtB = self.tB - slot.posB
                    slot.future[2:4] += x_t[2:4] * (self.s ** dtB)

                gammaA = (self.r ** slot.posA) / (1 - self.r) + (self.s ** (self.tA - slot.posA)) / (1 - self.s) if self.tA >= 0 else 0.0
                gammaB = (self.r ** slot.posB) / (1 - self.r) + (self.s ** (self.tB - slot.posB)) / (1 - self.s) if self.tB >= 0 else 0.0
                gamma = np.array([gammaA, gammaA, gammaB, gammaB], float)
                base = slot.past + slot.future
            



            lo_vec = base + gamma * self.m_vec
            hi_vec = base + gamma * self.M_vec

            # unpack: [seenA, accA, seenB, accB]
            seenA_lo, accA_lo, seenB_lo, accB_lo = lo_vec
            seenA_hi, accA_hi, seenB_hi, accB_hi = hi_vec

            phi_lo, phi_hi = interval_phi(
                accA_lo, accA_hi, seenA_lo, seenA_hi,
                accB_lo, accB_hi, seenB_lo, seenB_hi
            )

            # If phi interval undefined (denominators may be 0), we cannot decide yet.
            if phi_lo is None:
                continue

            if phi_lo >= self.Lin and phi_hi <= self.Uin:
                verdicts.append((slot.pos, 1, phi_lo, phi_hi))
                slot.in_use = False
                self.n_active_monitors -= 1
            elif phi_hi < self.Lout or phi_lo > self.Uout:
                verdicts.append((slot.pos, 0, phi_lo, phi_hi))
                slot.in_use = False
                self.n_active_monitors -= 1

        if t >= self.T and self.tau > 0:
            self._allocate_slot(pos=t)
            self.n_active_monitors += 1

        return self.n_active_monitors, verdicts


def df_events_to_stream4(df: pd.DataFrame, col: str = "input") -> np.ndarray:
    """
    Map each event row into x_t = [seenA, accA, seenB, accB].

    Assumes df[col] values encode one of:
      "A_acc", "A_rej", "B_acc", "B_rej"
    Adjust mapping here if you use different encoding (ints, tuples, etc).
    """
    mapping = {
        "A_acc": np.array([1, 1, 0, 0], dtype=float),
        "A_rej": np.array([1, 0, 0, 0], dtype=float),
        "B_acc": np.array([0, 0, 1, 1], dtype=float),
        "B_rej": np.array([0, 0, 1, 0], dtype=float),
    }
    return np.vstack([mapping[v] for v in df[col].values])


def phi_monitor_df2df(df: pd.DataFrame, mon: DiscountedPhiMonitor, col: str = "input") -> pd.DataFrame:
    stream4 = df_events_to_stream4(df, col=col)

    out = df.copy()
    out["verdict"] = np.nan
    out["time_verdict"] = np.nan
    out["phiL"] = np.nan
    out["phiU"] = np.nan
    out["active_monitors"] = np.nan

    for t in range(len(stream4)):
        n_active, verdicts = mon.step(stream4[t])
        out.loc[t, "active_monitors"] = n_active
        for pos, v, phi_lo, phi_hi in verdicts:
            out.loc[pos, "verdict"] = v
            out.loc[pos, "time_verdict"] = t
            out.loc[pos, "phiL"] = phi_lo
            out.loc[pos, "phiU"] = phi_hi

    out["wait_time"] = out["time_verdict"] - out.index
    out["active_monitors_normalized"] = out["active_monitors"] / mon.tau
    return out




class DiscountedSumMonitor:
    """
    Streaming monitor implementing the algorithm in the "Monitor Construction" subsection.

    Parameters:
      r: past discount factor in [0,1)
      s: future discount factor in [0,1)
      R_bounds: (m, M) = (inf R, sup R)
      I: target interval [L,U]
      eps: monitor tolerance 
      T: start time (do not allocate slots before t>=T)
      tau: optional override for number of slots; if None, compute tau_star(r,s,eps,T)
    """

    def __init__(
        self,
        r: float,
        s: float,
        R_bounds: Tuple[float, float],
        I: Tuple[float, float],
        eps: float,
        T: int,
        use_averages: bool = False,
        stochastic: bool = False,
        delta: float = 0.1,
        stoch_uniform: bool = False,
        tau: Optional[int] = None
    ):
        if not (0.0 <= r < 1.0 and 0.0 <= s < 1.0):
            raise ValueError("r and s must be in [0,1).")
        m, M = R_bounds
        if m > M:
            raise ValueError("R_bounds must be (m,M) with m <= M.")
        L, U = I
        if L > U:
            raise ValueError("I must be (L,U) with L <= U.")
        if T < 0:
            raise ValueError("T must be >= 0.")

        self.r = r
        self.s = s
        self.m = m
        self.M = M
        self.L = L
        self.U = U
        self.eps = eps
        self.Lin = L - eps
        self.Lout = L + eps
        self.Uin = U+eps
        self.Uout = U-eps

        self.stochastic = stochastic
        self.delta = delta
        self.stoch_uniform = stoch_uniform

        self.T = T
        self.use_averages = use_averages
        self.avg_normalization_value = 1+(r/(1-r))+(s/(1-s))
        self.n_active_monitors = 0

        self.tau = tau if tau is not None else tau_star(r, s, eps, T, m, M,use_averages)
        self.slots: List[Slot] = [Slot() for _ in range(self.tau)]
        self.psum: float = 0.0
        self.t: int = -1  # will become 0 on first step

    def _allocate_slot(self, pos: int) -> None:
        for slot in self.slots:
            if not slot.in_use:
                slot.in_use = True
                slot.past = self.psum
                slot.future = 0.0
                slot.pos = pos
                return
        raise RuntimeError(
            "No free slot available. Either increase tau or check that verdicts are being produced as expected."
        )

    def step(self, x_t: float) -> List[Tuple[int, Verdict]]:
        """
        Process one observed value x_t.

        Returns a list of (pos, verdict) for any slots that became decidable at this step.
        """
        self.t += 1
        t = self.t

        if self.use_averages:
            x_t = x_t/self.avg_normalization_value      

        # Update global past sum
        self.psum = x_t + self.r * self.psum
        # print(f"{self.psum=}")


        verdicts: List[Tuple[int, int, float, float]] = []

        # Update active slots, compute uncertainty intervals, and decide
        for slot in self.slots:
            if not slot.in_use:
                continue

            # Future update
            dt = t - slot.pos
            slot.future += x_t * (self.s ** dt)

            # Uncertainty interval (as written in your construction)
            gamma = (self.r ** slot.pos) / (1.0 - self.r) + (self.s ** dt) / (1.0 - self.s)
            beta = 0
            if self.stochastic:
                omega = aux_omega(self.r,self.s,slot.pos,0,t)
                sigma2 = ((self.M-self.m)/self.avg_normalization_value)**2 if self.use_averages else (self.M-self.m)**2
                Vn = sigma2*omega
                if self.stoch_uniform:
                    Vn = max(1,Vn)
                    rad = 2*np.log(1 + np.log2(Vn)) + np.log(2*np.pi*np.pi/(6*self.delta))
                    rad = Vn*rad
                    fact = np.pow(2,1/4) + np.pow(2,-1/4)/np.sqrt(2)
                    beta = fact*np.sqrt(rad)
                    
                else:
                    rad = 0.5*Vn*np.log(2/self.delta)
                    beta = np.sqrt(rad)
                    

        
            base = slot.past + slot.future
            lo = base + gamma * self.m - beta
            hi = base + gamma * self.M + beta

            # Decide
            if lo >= self.Lin and hi <= self.Uin:
                verdicts.append((slot.pos, 1, lo, hi))
                slot.in_use = False
                self.n_active_monitors -= 1
            elif hi < self.Lout or lo > self.Uout:
                verdicts.append((slot.pos, 0, lo, hi))
                slot.in_use = False
                self.n_active_monitors -= 1

        # Activate a new slot at time t if t >= T
        if t >= self.T and self.tau > 0:
            self._allocate_slot(pos=t)
            self.n_active_monitors += 1

        return self.n_active_monitors, verdicts

    def run(self, xs: Iterable[float]) -> List[Tuple[int, int, float, float]]:
        """Convenience method: process a finite iterable and collect all produced verdicts."""
        out: List[Tuple[int, int, float, float]] = []
        for x in xs:
            out.extend(self.step(x))
        return out



def discount_monitor_df2df(df, mon):
    """
    assumes it has a column named input, which is the one to monitor
    """

    stream = df["input"].values

    df["verdict"] = np.nan
    df["time_verdict"] = np.nan
    df["valL"] = np.nan
    df["valU"] = np.nan
    df["active_monitors"] = np.nan


    for t in range(len(stream)):
        x = stream[t]
        n_active_monitors , verdicts = mon.step(x)
        df.loc[t,"active_monitors"] = n_active_monitors
        for pos, v, vallo, valhi in verdicts:
            # print(f"t={mon.t:>2} produced verdict for pos={pos:>2}: {v}, with values {vallo:.3f}, {valhi:.3f}")
            df.loc[pos, "verdict"] = v
            df.loc[pos,"time_verdict"] = t
            df.loc[pos, "valL"] = vallo
            df.loc[pos, "valU"] = valhi
    df["wait_time"] = df.time_verdict - df.index
    df["active_monitors_normalized"] = df["active_monitors"]/mon.tau
    return df


def monitor_one_stream(params):
    r = params["r"]
    s = params["s"]
    infR = params["infR"]
    supR = params["supR"]
    R_bounds = (infR, supR)
    I = (params["L"], params["U"])
    eps = params["eps"]
    T = params["T"]
    avg = params["use_averages"]
    if "uniform" in params:
        # stochastic: bool = False,
        # delta: float = 0.1,
        # stoch_uniform: bool = False,
        # tau: Optional[int] = None
        mon = DiscountedSumMonitor(r=r, s=s, R_bounds=R_bounds, I=I, eps=eps, T=T, use_averages = avg,
                                    stochastic = True, stoch_uniform = params["uniform"], tau = 30000)
    else:
        mon = DiscountedSumMonitor(r=r, s=s, R_bounds=R_bounds, I=I, eps=eps, T=T, use_averages = avg)

    orig_df = pd.read_csv(params["data_source"])
    data_column_name = params["data_column_name"]
    df = orig_df.sort_values("time").reset_index()[[data_column_name]].rename(columns={data_column_name:"input"})
    
    df = discount_monitor_df2df(df, mon)

    save_dir = params["data_results"]

    df.to_csv(save_dir)

  


def powerdata_decrease_eps():
    params_file = "experimental-setups/RQ1-google-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)

def powerdata_decrease_interval_length():
    params_file = "experimental-setups/RQ1-google-decrease-target-interval.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)    


def mnist_increase_noise():
    params_file = "experimental-setups/RQ1-mnist-traces-increase-noise.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)


def ffb_adult_race_decrease_eps():

    params_file = "experimental-setups/ffb-adult-race-decrease-eps.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        df_events = pd.read_csv(params["data_source"])
        R_bounds = (np.zeros(4), np.ones(4))
        s = params["s"]
        r = params["r"]
        L = params["L"]
        U = params["U"]
        mon = DiscountedPhiMonitor(r=params["r"], s=params["s"], R_bounds=R_bounds, I=(params["L"], params["U"]), 
        eps=params["eps"], T=params["T"], tau = 2000, synchronous = params["synchronous"])
        df_out = phi_monitor_df2df(df_events, mon, col="input")
        df_out.to_csv(params["data_results"])



    with open("experimental-setups/params_ffb_example.json", "r") as fp:
        params = json.load(fp)[0]

def ffb_adult_race_decrease_interval_length():
    params_file = "experimental-setups/ffb-adult-race-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        df_events = pd.read_csv(params["data_source"])
        R_bounds = (np.zeros(4), np.ones(4))
        s = params["s"]
        r = params["r"]
        L = params["L"]
        U = params["U"]
        mon = DiscountedPhiMonitor(r=params["r"], s=params["s"], R_bounds=R_bounds, I=(params["L"], params["U"]), 
        eps=params["eps"], T=params["T"], tau = 2000, synchronous = params["synchronous"])
        df_out = phi_monitor_df2df(df_events, mon, col="input")
        df_out.to_csv(params["data_results"])


def stochastic_decrease_interval_length():
    params_file = "experimental-setups/stochastic-decrease-interval-length.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)

    
def stochastic_decrease_interval_length_rand():
    params_file = "experimental-setups/stochastic-decrease-interval-length_rand.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)



def stochastic_decrease_interval_length30k():
    params_file = "experimental-setups/stochastic-decrease-interval-length30k.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)

    
def stochastic_decrease_interval_length30k_rand():
    params_file = "experimental-setups/stochastic-decrease-interval-length30k_rand.json"
    with open(params_file, "r") as fp:
        params_vec = json.load(fp)

    for i in tqdm(range(len(params_vec))):
        params = params_vec[i]
        monitor_one_stream(params)





def main():
    # powerdata_decrease_eps()
    # powerdata_decrease_interval_length()
    # mnist_increase_noise()
    # ffb_adult_race_decrease_eps()
    # ffb_adult_race_decrease_interval_length()
    stochastic_decrease_interval_length()
    stochastic_decrease_interval_length_rand()

    # stochastic_decrease_interval_length30k()
    # stochastic_decrease_interval_length30k_rand()
    




if __name__ == "__main__":
    main()
