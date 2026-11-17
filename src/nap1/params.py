"""Stage-1 parameters in one place, with JSON reading/writing.

No neural network: each parameter is interpretable and fitted using counts (MLE), logistic
regression or grid search on DFL.
"""

import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class Params:
    # --- Block 1: View (preprocessing and velocity) ---
    max_step_m: float = 2.0          # Maximum adjacent-frame displacement before splitting a track (m).
    savgol_window: int = 7           # Savitzky-Golay velocity window, in frames (odd).
    savgol_poly: int = 2             # Savitzky-Golay polynomial order.

    # --- Block 2: Ball estimation / touch detection (official configuration fitted on DFL v3) ---
    qa: float = 15.0                 # Kalman process noise (acceleration, m/s², fitted on ball_xy).
    sigma_m: float = 0.10            # Ball-position measurement noise (m, fitted by Touch F1).
    tau: float = 8.0                 # Mahalanobis d² threshold for anomaly / touch detection.
    touch_radius: float = 3.2        # Player-contact radius around the ball (m).
    min_touch_gap: int = 12          # Minimum interval between touches (frames; 12 frames = 0.48 s).
    v_sigma: float = 12.0            # Uncertainty growth for 1d_forward/2d_bounded (Gaussian NLL).
    sigma_cam: float = 19.05         # Nominal camera-center uncertainty during gap fusion (Gaussian NLL).
    touch_smooth_win: int = 2        # Touch-probability smoothing radius (+-frames).
    ball_gap_mode: str = "rts_cov"   # Gap uncertainty mode: 'rts_cov' (2x2 RTS), '2d_bounded', '1d_forward'.

    # --- Block 3a: Pitch control ---
    react_time: float = 0.0          # Player reaction time (s, fitted on DFL live frames).
    v_max: float = 5.0               # Maximum player running speed (m/s).
    sigma_pc: float = 0.25           # Arrival-time sigmoid slope, calibrated on DFL select.
    use_velocity: bool = False       # False: align timing with the ball (LiveAcc +0.08 vs t+0.7 s).

    # --- Block 3b: Dead-ball detection (six-feature logistic regression) ---
    w_dead_b: float = 4.058          # Intercept
    w_dead_vb: float = -0.344        # Ball-speed weight (m/s).
    w_dead_n5: float = -0.025        # Weight for the player count within 5 m of the ball.
    w_dead_dm: float = 0.083         # Nearest-player distance weight (m).
    w_dead_sl: float = -2.579        # Mean player-speed weight (m/s).
    w_dead_out: float = 2.855        # Soft P(out) weight, jointly fitted with L-BFGS-B.

    # --- Block 4: Four-state IOHMM (H, A, D-after-H, D-after-A) ---
    L_alive: float = 7.29            # Mean live H/A possession duration (s).
    L_dead: float = 24.5             # Mean dead-ball duration (s).
    share_dead: float = 0.234        # Share of possessions ending in dead ball, P(alive -> dead).
    p_switch_after_dead: float = 0.693# Team-switch probability after dead ball.
    touch_window: int = 6            # Per-team touch-evidence pooling radius (|u - t| <= W).
    eps_gate: float = 0.01           # Minimum background gate probability without a detected touch.
    lambda_emit: float = 0.2         # Balance between log-emissions and log-transition smoothness.
    deadball_switch_gate: bool = True # True: gate D exits (selected on select); False: ungated 70/30.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Params":
        """Construct from a dictionary, ignoring unknown fields."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Save parameters to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Params":
        """Read parameters from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def hand_prior(cls) -> "Params":
        """Original hand-set configuration before DFL-v3 fitting, retained for ablations and initialization."""
        return cls(
            qa=6.0,
            sigma_m=0.12,
            tau=30.0,
            touch_radius=2.0,
            min_touch_gap=8,
            v_sigma=1.5,
            sigma_cam=12.0,
            touch_smooth_win=2,
            ball_gap_mode="2d_bounded",
            react_time=0.7,
            v_max=5.0,
            sigma_pc=0.45,
            use_velocity=True,
            w_dead_b=1.0,
            w_dead_vb=-0.35,
            w_dead_n5=-0.80,
            w_dead_dm=0.55,
            w_dead_sl=-0.90,
            w_dead_out=0.0,
            L_alive=20.0,
            L_dead=25.0,
            share_dead=0.22,
            p_switch_after_dead=0.70,
            touch_window=6,
            eps_gate=0.02,
            lambda_emit=0.15,
            deadball_switch_gate=False,
        )


def ensure_params(p: Optional[Union[Params, Dict[str, Any]]] = None) -> Params:
    """Adapter ensuring a standard Params object.

    When p is None, load the fitted official src/nap1/configs/params_official.json through
    load_fold_params(None), so quick decode / score_frames / estimate_ball calls do not fall
    back to unfitted hand-set values. For CV folds 0..5, explicitly pass load_fold_params(fold)
    to prevent cross-fold leakage.
    """
    if p is None:
        return load_fold_params(None)
    if isinstance(p, Params):
        return p
    if isinstance(p, dict):
        return Params.from_dict(p)
    return load_fold_params(None)


def load_fold_params(fold: Optional[int] = None) -> Params:
    """Load fold-specific Params for leakage-free G3 comparisons.

    fold=None: params_official.json or default Params.
    Integer fold: params_fold_{fold}.json, falling back to official/default values.
    """
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    if fold is None:
        p_path = os.path.join(configs_dir, "params_official.json")
    else:
        p_path = os.path.join(configs_dir, f"params_fold_{fold}.json")

    if os.path.isfile(p_path):
        return Params.load(p_path)

    off_path = os.path.join(configs_dir, "params_official.json")
    if os.path.isfile(off_path):
        return Params.load(off_path)

    return Params()


def default_release_path() -> Optional[str]:
    """Prefer CURRENT-v3.json from release-v3.zip when available; otherwise let pipeline.feed choose its default."""
    root = REPO_ROOT
    v3 = os.path.join(root, "cache", "degradation", "CURRENT-v3.json")
    if os.path.isfile(v3):
        return v3
    return None


def default_devset_dir() -> str:
    """Prefer devset-v2 from release-v3.zip when available; otherwise use devset-v1."""
    root = REPO_ROOT
    v2 = os.path.join(root, "cache", "degradation", "devset-v2")
    if os.path.exists(v2):
        return v2
    return os.path.join(root, "cache", "degradation", "devset-v1")
