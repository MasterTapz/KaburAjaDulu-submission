"""Possession labels from tracking.

predict() is the submitted method. It corrects the goalkeeper tracks (cleaning.py), turns every
frame into costs for H, A and D (evidence.py, setpieces.py, restarts.py), and segments each half
into H, A and D with a price per change of state (segment.py). Where there is no evidence, as with a lost or contested
ball, possession stays with the last team, as the rules define it.

baseline() is the organisers' reference baseline (nearest tracked player to the ball, then a
centred majority vote), plus the match structure the documentation states. It is the floor that
every version must beat.
"""

import numpy as np

from . import evidence
from . import params as P
from . import restarts, setpieces
from .cleaning import clean_goalkeepers
from .config import FPS, HALVES, KICKOFFS, N_FRAMES
from .segment import marginals, viterbi

NOBODY = -1
# The organisers state that the frames from the start of the graded second half to its kickoff
# are all dead ball.
DEAD_SPANS = ((HALVES[1][0], KICKOFFS[1]),)


# --- Baseline ---

def nearest_team(tracking):
    """Per frame: team of the nearest tracked player to the ball, NOBODY without a ball."""
    dx = tracking.x - tracking.ball_x[:, None]
    dy = tracking.y - tracking.ball_y[:, None]
    distance = np.where(np.isnan(dx), np.inf, dx * dx + dy * dy)
    home = distance[:, tracking.slot_team == P.H].min(axis=1)
    away = distance[:, tracking.slot_team == P.A].min(axis=1)
    codes = np.where(home <= away, P.H, P.A).astype(np.int8)
    codes[~np.isfinite(np.minimum(home, away))] = NOBODY
    return codes


def smooth(codes, window):
    """Majority vote over +/- window frames; D where nobody had the ball. Ties go home."""
    if window <= 0:
        return np.where(codes == P.H, "H", np.where(codes == P.A, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == P.H)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == P.A)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def apply_match_structure(labels):
    """X off the graded halves, where nothing is scored; D over the documented dead span."""
    out = np.full(N_FRAMES, "X", dtype="<U1")
    for first, last in HALVES:
        out[first:last + 1] = labels[first:last + 1]
    for first, stop in DEAD_SPANS:
        out[first:stop] = "D"
    return out


def baseline(tracking, window=None):
    return apply_match_structure(smooth(nearest_team(tracking),
                                        P.BASELINE_WINDOW if window is None else window))


# --- The submitted method ---

def change_factors():
    """(3, 3) price factors for H, A, D: 1 for a change of team, DEAD_SWITCH_FACTOR for a change
    into or out of a dead ball; None when every change costs the same."""
    if P.DEAD_SWITCH_FACTOR == 1.0:
        return None
    factors = np.full((3, 3), P.DEAD_SWITCH_FACTOR)
    factors[P.H, P.A] = factors[P.A, P.H] = 1.0
    return factors


def costs(tracking, tags, halves=HALVES, dead_spans=DEAD_SPANS):
    """(cost, diagnostics): per-frame costs for H, A, D after the goalkeeper fix, with the
    set-piece shapes and the wait for restart kicks added and H and A ruled out over the dead
    spans."""
    tracking, fixed = clean_goalkeepers(tracking, halves)
    cost, diagnostics = evidence.costs(tracking, tags, halves)
    diagnostics["goalkeeper frames fixed"] = float(fixed)
    if P.W_SET_PIECE:
        shape = np.zeros(len(cost), dtype=bool)
        for fires in setpieces.detect(tracking, halves).values():
            shape |= fires
        cost[shape, P.H] += P.W_SET_PIECE
        cost[shape, P.A] += P.W_SET_PIECE
    if P.W_RESTART_WAIT:
        wait = restarts.restart_wait(tracking, halves)
        cost[wait, P.H] += P.W_RESTART_WAIT
        cost[wait, P.A] += P.W_RESTART_WAIT
    for first, stop in dead_spans:
        cost[first:stop, [P.H, P.A]] = P.FORBIDDEN
    return cost, diagnostics


def shift_stoppages(path, start, end):
    """Move each dead run of one half's states `start` frames earlier at its start and `end`
    frames earlier at its end (STOP_SHIFT_START, STOP_SHIFT_END; a negative number moves the edge
    later). Frames the run gains come from its neighbour, frames it gives up go to it. Each run
    and each neighbour keeps at least one frame; a run at the edge of the half keeps that edge."""
    out = path.copy()
    n = len(path)
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], change)).tolist()
    stops = np.concatenate((change, [n])).tolist()
    # Where each run starts now: a dead run that ends later pushes the next run's start, and the
    # dead run after that must not take that run's last frame.
    now = list(starts)
    for i, (a, b) in enumerate(zip(starts, stops)):
        if path[a] != P.D:
            continue
        new_a = a
        if a > 0:
            new_a = min(max(a - start, now[i - 1] + 1), b - 1)
            if new_a < a:
                out[new_a:a] = P.D
            else:
                out[a:new_a] = path[a - 1]
        if b < n:
            new_b = min(max(b - end, new_a + 1), stops[i + 1] - 1)
            if new_b < b:
                out[new_b:b] = path[b]
            else:
                out[b:new_b] = P.D
            now[i + 1] = new_b
    return out


def segment(cost, halves=HALVES, dead_spans=DEAD_SPANS):
    """Labels from costs: each half segmented on its own (nothing crosses half-time), X off the
    halves, D over the dead spans. The predicted stoppages then move by STOP_SHIFT_START and
    STOP_SHIFT_END. Split from predict() so calibration can reuse one match's costs across
    settings of the prices and shifts."""
    factors = change_factors()
    start = int(round(P.STOP_SHIFT_START * FPS))
    end = int(round(P.STOP_SHIFT_END * FPS))
    labels = np.full(len(cost), "X", dtype="<U1")
    for first, last in halves:
        part = slice(first, last + 1)
        path = viterbi(cost[part], P.SWITCH_COST, factors)
        if start or end:
            path = shift_stoppages(path, start, end)
        labels[part] = np.array(["H", "A", "D"])[path]
    for first, stop in dead_spans:
        labels[first:stop] = "D"
    return labels


def predict(tracking, tags, halves=HALVES, dead_spans=DEAD_SPANS):
    """(labels, diagnostics). halves and dead_spans default to our match; other matches pass
    their own (dead_spans=() where none is documented)."""
    cost, diagnostics = costs(tracking, tags, halves, dead_spans)
    return segment(cost, halves, dead_spans), diagnostics


def posteriors(tracking, tags, halves=HALVES, dead_spans=DEAD_SPANS):
    """(n, 3) probability of H, A, D per frame under the same costs and prices; 0 off the
    halves. How sure the model is: a diagnostic. The labels come from predict()."""
    cost, _ = costs(tracking, tags, halves, dead_spans)
    factors = change_factors()
    out = np.zeros((len(cost), 3))
    for first, last in halves:
        part = slice(first, last + 1)
        out[part] = marginals(cost[part], P.SWITCH_COST, factors)
    return out
