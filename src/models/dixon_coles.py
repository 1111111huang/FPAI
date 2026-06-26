"""Dixon-Coles football match outcome model (US#73).

Fits team-level attack and defence parameters via MLE, with the low-score
tau correction from Dixon & Coles (1997). Used as a principled statistical
baseline against which ML models are compared.

Targets covered:
  result_3way  — probabilities from joint goal distribution
  btts         — P(H>=1) * P(A>=1) from joint distribution
  home_goals   — lambda_h directly
  away_goals   — lambda_a directly
  total_goals  — lambda_h + lambda_a
  home_corners / away_corners / total_corners — per-team historical means
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_MAX_GOALS = 8


def _tau_matrix(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Return (_MAX_GOALS+1) x (_MAX_GOALS+1) tau correction matrix."""
    mg = _MAX_GOALS
    T = np.ones((mg + 1, mg + 1))
    T[0, 0] = 1.0 - lam_h * lam_a * rho
    T[1, 0] = 1.0 + lam_a * rho
    T[0, 1] = 1.0 + lam_h * rho
    T[1, 1] = 1.0 - rho
    return T


class DixonColesModel:
    """Dixon-Coles (1997) Poisson model with low-score correction.

    Parameter layout (internal):
      [mu, attack_1..N-1, defence_0..N-1, home_adv, rho]
      attack_0 is fixed to 0 for identifiability.
    """

    def __init__(self) -> None:
        self.teams_: list[str] = []
        self._attack: dict[str, float] = {}
        self._defence: dict[str, float] = {}
        self._home_adv: float = 0.0
        self._rho: float = 0.0
        self._mu: float = 0.0
        self._mean_attack: float = 0.0
        self._mean_defence: float = 0.0
        # Per-team corner means keyed by (team, venue) where venue in {"home","away"}
        self._corner_home: dict[str, float] = {}
        self._corner_away: dict[str, float] = {}
        self._global_corner_home: float = float("nan")
        self._global_corner_away: float = float("nan")
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches_df: pd.DataFrame) -> "DixonColesModel":
        """Fit parameters on historical match data.

        Parameters
        ----------
        matches_df : DataFrame with columns [home_team, away_team, fthg, ftag].
                     Optionally [hc, ac] for corner baseline means.
        """
        df = matches_df.dropna(subset=["home_team", "away_team", "fthg", "ftag"]).copy()
        df["fthg"] = df["fthg"].astype(int)
        df["ftag"] = df["ftag"].astype(int)

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        N = len(all_teams)
        team_idx = {t: i for i, t in enumerate(all_teams)}
        self.teams_ = all_teams

        h_idx = df["home_team"].map(team_idx).to_numpy(dtype=int)
        a_idx = df["away_team"].map(team_idx).to_numpy(dtype=int)
        hg = df["fthg"].to_numpy(dtype=int)
        ag = df["ftag"].to_numpy(dtype=int)

        # param layout: [mu, atk_1..N-1, dfc_0..N-1, home_adv, rho]
        n_params = 1 + (N - 1) + N + 1 + 1

        def _unpack(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float, float]:
            mu = p[0]
            atk = np.concatenate([[0.0], p[1:N]])
            dfc = p[N: 2 * N]
            home_adv = p[2 * N]
            rho = p[2 * N + 1]
            return mu, atk, dfc, home_adv, rho

        def neg_ll(p: np.ndarray) -> float:
            mu, atk, dfc, home_adv, rho = _unpack(p)
            lam_h = np.exp(mu + home_adv + atk[h_idx] - dfc[a_idx])
            lam_a = np.exp(mu + atk[a_idx] - dfc[h_idx])
            ll = (
                hg * np.log(lam_h + 1e-12) - lam_h
                + ag * np.log(lam_a + 1e-12) - lam_a
            )
            # tau correction only for 0-0, 1-0, 0-1, 1-1
            tau = np.ones(len(hg))
            m00 = (hg == 0) & (ag == 0)
            m10 = (hg == 1) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
            tau[m10] = 1.0 + lam_a[m10] * rho
            tau[m01] = 1.0 + lam_h[m01] * rho
            tau[m11] = 1.0 - rho
            ll += np.log(np.clip(tau, 1e-12, None))
            return float(-ll.sum())

        p0 = np.zeros(n_params)
        p0[0] = float(np.log(df["fthg"].mean() + 1e-6))

        bounds = [(None, None)] * n_params
        bounds[-1] = (-0.99, 0.99)  # rho

        LOGGER.info("Fitting Dixon-Coles: %d teams, %d matches", N, len(df))
        result = minimize(
            neg_ll, p0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        if not result.success:
            LOGGER.warning("Dixon-Coles MLE did not fully converge: %s", result.message)

        mu, atk, dfc, home_adv, rho = _unpack(result.x)
        self._mu = float(mu)
        self._home_adv = float(home_adv)
        self._rho = float(rho)
        self._attack = {t: float(atk[i]) for i, t in enumerate(all_teams)}
        self._defence = {t: float(dfc[i]) for i, t in enumerate(all_teams)}
        self._mean_attack = float(np.mean(list(self._attack.values())))
        self._mean_defence = float(np.mean(list(self._defence.values())))

        # Corner per-team means
        if "hc" in df.columns and "ac" in df.columns:
            hc_clean = df.dropna(subset=["hc"])
            ac_clean = df.dropna(subset=["ac"])
            self._corner_home = hc_clean.groupby("home_team")["hc"].mean().to_dict()
            self._corner_away = ac_clean.groupby("away_team")["ac"].mean().to_dict()
            self._global_corner_home = float(df["hc"].dropna().mean())
            self._global_corner_away = float(df["ac"].dropna().mean())

        self._fitted = True
        LOGGER.info(
            "Dixon-Coles fit: mu=%.4f home_adv=%.4f rho=%.4f",
            self._mu, self._home_adv, self._rho,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        atk_h = self._attack.get(home_team, self._mean_attack)
        dfc_h = self._defence.get(home_team, self._mean_defence)
        atk_a = self._attack.get(away_team, self._mean_attack)
        dfc_a = self._defence.get(away_team, self._mean_defence)
        lam_h = float(np.exp(self._mu + self._home_adv + atk_h - dfc_a))
        lam_a = float(np.exp(self._mu + atk_a - dfc_h))
        return lam_h, lam_a

    def _joint_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        """Return (mg+1)x(mg+1) joint P(H=h, A=a) with tau correction."""
        mg = _MAX_GOALS
        ph = poisson.pmf(np.arange(mg + 1), lam_h)
        pa = poisson.pmf(np.arange(mg + 1), lam_a)
        joint = np.outer(ph, pa)
        joint *= _tau_matrix(lam_h, lam_a, self._rho)
        s = joint.sum()
        if s > 0:
            joint /= s
        return joint

    def predict_match(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Return predictions for all FPAI targets for one fixture."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        lam_h, lam_a = self._lambdas(home_team, away_team)
        joint = self._joint_matrix(lam_h, lam_a)

        # result_3way — rows=home goals, cols=away goals
        p_home = float(np.sum(np.tril(joint, k=-1)))   # h > a
        p_away = float(np.sum(np.triu(joint, k=1)))    # a > h
        p_draw = float(np.trace(joint))                 # h == a
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        # btts: P(H>=1 AND A>=1) via inclusion-exclusion on joint
        p_btts = float(1.0 - joint[0, :].sum() - joint[:, 0].sum() + joint[0, 0])
        p_btts = float(np.clip(p_btts, 0.0, 1.0))

        # corners: per-team historical mean, fall back to global mean
        hc = self._corner_home.get(home_team, self._global_corner_home)
        ac = self._corner_away.get(away_team, self._global_corner_away)

        return {
            "result_3way_p_home": p_home,
            "result_3way_p_draw": p_draw,
            "result_3way_p_away": p_away,
            "result_3way_pred": ["home", "draw", "away"][int(np.argmax([p_home, p_draw, p_away]))],
            "btts_prob": p_btts,
            "btts_pred": int(p_btts >= 0.5),
            "home_goals": lam_h,
            "away_goals": lam_a,
            "total_goals": lam_h + lam_a,
            "home_corners": hc,
            "away_corners": ac,
            "total_corners": float(hc) + float(ac) if not (np.isnan(hc) or np.isnan(ac)) else float("nan"),
        }

    def predict_batch(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of predictions for each row in matches_df.

        Expects columns: [home_team, away_team].
        """
        rows = []
        for _, row in matches_df.iterrows():
            pred = self.predict_match(str(row["home_team"]), str(row["away_team"]))
            rows.append(pred)
        return pd.DataFrame(rows, index=matches_df.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, str(target_path))
        LOGGER.info("DixonColesModel saved to %s", target_path)

    @classmethod
    def load(cls, path: str) -> "DixonColesModel":
        obj = cls()
        obj.__dict__.update(joblib.load(str(path)))
        return obj
