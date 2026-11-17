"""Stage 1: touch detection, touching-team estimation and two-level IOHMM decoding.

Five-block architecture aligned with pipeline.feed / pipeline.split / train_data:
  1. View: standard pipeline.view via from_feed / from_npz, or create_view; add boundary
     finite-difference velocities for tracks of at least two frames.
  2. Ball: bidirectional Kalman/RTS ball smoothing, touch K detection and Mahalanobis d²
     outlier rejection, fusing camera center during gaps.
  3. Score: pitch control p1 at the ball, logistic dead-ball D, log-scores E[T,3] and
     touching-team C[T,2], through the fixed score_frames API.
  4. Decode: four-state IOHMM (H, A, D-after-H, D-after-A) with a touch gate and Viterbi.
  5. Evaluation / submission: pipeline.evaluate_dev / score_export and submission CSV export.
"""

from .ball import estimate_ball
from .deadball import dead_prob
from .decode import codes_to_labels, decode, to_eval_codes, to_submission_df
from .params import Params, ensure_params, load_fold_params
from .pitch_control import pitch_control
from .score import ScoreDetails, score_frames, score_frames_detailed
from .view import View, create_view, from_feed, from_npz

__all__ = [
    "Params",
    "ensure_params",
    "load_fold_params",
    "View",
    "create_view",
    "from_feed",
    "from_npz",
    "estimate_ball",
    "pitch_control",
    "dead_prob",
    "ScoreDetails",
    "score_frames",
    "score_frames_detailed",
    "decode",
    "codes_to_labels",
    "to_eval_codes",
    "to_submission_df",
]
