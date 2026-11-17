"""Block 3c: frame scores and touching-team probabilities.

  1. Combine ball-location pitch control p1 and dead-ball probability D.
  2. Build E[T,3] log-emissions:
     E[:,0] = log(p1 * (1-D))       # H: home live ownership
     E[:,1] = log((1-p1) * (1-D))   # A: away live ownership
     E[:,2] = log(D)               # D: dead ball
  3. Touching-team probabilities C[T,2]: C[:,0] = p1 (group 1), C[:,1] = 1-p1 (group 2).
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

from .deadball import dead_prob
from .params import Params, ensure_params
from .pitch_control import pitch_control


@dataclass
class ScoreDetails:
    """Container for detailed block-3 scoring results."""
    E: np.ndarray      # [T, 3] float64 (log-probabilities cho H, A, D)
    C: np.ndarray      # [T,2] float64: group-1 / group-2 touching probabilities.
    p1: np.ndarray     # [T] float64: group-1 pitch-control probability.
    D: np.ndarray      # [T] float64: dead-ball probability.


def score_frames(V: Any, Bhat: np.ndarray, K: np.ndarray,
                 params: Optional[Union[Params, dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Standard pipeline / nap2 API: ALWAYS return exactly two arrays (E, C).

    Returns:
        E: np.ndarray [T,3] float64, log H/A/D probabilities
        C: np.ndarray [T,2] float64, group-1 / group-2 touching probabilities
    """
    details = score_frames_detailed(V, Bhat, K, params)
    return details.E, details.C


def score_frames_detailed(V: Any, Bhat: np.ndarray, K: np.ndarray,
                          params: Optional[Union[Params, dict]] = None) -> ScoreDetails:
    """Detailed E[T,3], C[T,2], p1[T] and D[T] for diagnostics and ablations.

    Returns:
        ScoreDetails dataclass with all fields (E, C, p1, D).
    """
    p = ensure_params(params)
    T = V.n_frames if hasattr(V, "n_frames") else V.P.shape[0]

    # 1. Pitch Control p1
    p1 = pitch_control(V, Bhat, p)

    # 2. Dead-ball probability D
    D = dead_prob(V, Bhat, p)

    # 3. Three-class probabilities: H, A, D
    # Conditional on a live ball, split ownership by p1 and 1-p1.
    eps = 1e-12
    p_H = np.clip(p1 * (1.0 - D), eps, 1.0 - eps)
    p_A = np.clip((1.0 - p1) * (1.0 - D), eps, 1.0 - eps)
    p_D = np.clip(D, eps, 1.0 - eps)

    # Normalize probabilities to sum to 1.0 per frame.
    p_sum = p_H + p_A + p_D
    p_H /= p_sum
    p_A /= p_sum
    p_D /= p_sum

    E = np.empty((T, 3), dtype=np.float64)
    E[:, 0] = np.log(p_H)
    E[:, 1] = np.log(p_A)
    E[:, 2] = np.log(p_D)

    # 4. Touching-team probability C[T,2]
    C = np.empty((T, 2), dtype=np.float64)
    C[:, 0] = p1
    C[:, 1] = 1.0 - p1

    return ScoreDetails(E=E, C=C, p1=p1, D=D)
