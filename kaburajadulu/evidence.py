"""Per-frame evidence for the three states H, A and D, written as costs.

A cost of 0 means the frame does not object to that state; a higher cost makes it less likely.
segment.py then finds the cheapest labelling, charging for every change of state. Frames with no
evidence cost nothing anywhere, so the current state simply continues. This is the rule for a
loose or contested ball: possession stays with the last team that had it (CHALLENGE.md).

The signals, each with its weight in params.py:

- control: the nearest player is close to the ball and clearly closer than any opponent;
- touch: a player is near while the ball turns or changes speed;
- ball out: the tracked ball is beyond a line;
- pace: the players walk (a dead ball) or run (live play);
- Smart Tagging: an attacking phase tagged for one team;
- team shape: how the two teams stand and move (shape.py).
"""

import warnings

import numpy as np
import pandas as pd

from . import params as P
from . import shape
from .config import FPS, HALVES, N_FRAMES, PITCH


def nearest(tracking):
    """(team, near, far) per frame: the nearest player's team, its distance to the ball, and the
    nearest opponent's distance. inf where there is no ball or nobody to measure."""
    dx = tracking.x - tracking.ball_x[:, None]
    dy = tracking.y - tracking.ball_y[:, None]
    distance = np.sqrt(np.where(np.isnan(dx), np.inf, dx * dx + dy * dy))
    home = distance[:, tracking.slot_team == P.H].min(axis=1)
    away = distance[:, tracking.slot_team == P.A].min(axis=1)
    return np.where(home <= away, P.H, P.A), np.minimum(home, away), np.maximum(home, away)


def control(tracking):
    """(team, clear) per frame: the nearest player's team, and whether it clearly controls."""
    team, near, far = nearest(tracking)
    with np.errstate(invalid="ignore"):
        clear = (near <= P.CONTROL_RADIUS) & (far - near >= P.CONTROL_MARGIN)
    return team, clear


def touches(tracking):
    """(team, touch) per frame: a player within TOUCH_RADIUS while the ball turns by more than
    TOUCH_ANGLE or changes speed by more than TOUCH_SPEED_CHANGE (Vidal-Codina et al., 2022).

    The ball's velocity in and out of the frame is measured over TOUCH_STEP frames each side, so
    one jittery position does not make a touch; below TOUCH_MIN_SPEED its direction is noise.
    """
    team, near, _ = nearest(tracking)
    k, n = P.TOUCH_STEP, len(tracking.ball_x)
    position = np.stack([tracking.ball_x, tracking.ball_y], axis=1)
    v_in = np.full((n, 2), np.nan)
    v_out = np.full((n, 2), np.nan)
    v_in[k:] = (position[k:] - position[:-k]) * FPS / k
    v_out[:-k] = (position[k:] - position[:-k]) * FPS / k
    speed_in = np.hypot(v_in[:, 0], v_in[:, 1])
    speed_out = np.hypot(v_out[:, 0], v_out[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (v_in * v_out).sum(axis=1) / (speed_in * speed_out)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        turned = (angle > P.TOUCH_ANGLE) & (np.minimum(speed_in, speed_out) >= P.TOUCH_MIN_SPEED)
        kicked = np.abs(speed_out - speed_in) > P.TOUCH_SPEED_CHANGE
        touch = (near <= P.TOUCH_RADIUS) & (turned | kicked)
    return team, touch


def ball_out(tracking):
    """True where the tracked ball is more than OUT_TOLERANCE beyond a line."""
    length, width = PITCH
    x, y = tracking.ball_x, tracking.ball_y
    with np.errstate(invalid="ignore"):   # NaN compares False: an unseen ball is not out
        return ((x < -P.OUT_TOLERANCE) | (x > length + P.OUT_TOLERANCE)
                | (y < -P.OUT_TOLERANCE) | (y > width + P.OUT_TOLERANCE))


def pace(tracking):
    """Mean speed of the tracked players per frame, as a centred 1 s rolling median (m/s)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(tracking.speed, axis=1)
    return pd.Series(mean).rolling(P.PACE_WINDOW, center=True, min_periods=1).median().values


def future_reach(tracking, seconds, halves=HALVES):
    """(team, clear) per frame: the team whose player now stands nearest to where the ball will
    be in `seconds`, and whether that player is within FUTURE_RADIUS of it and FUTURE_MARGIN
    nearer than any opponent. False where the ball is not seen then, is out of play then, or
    `seconds` later is past the end of the half (nothing is read across half-time)."""
    k = int(round(seconds * FPS))
    n = len(tracking.ball_x)
    inside = np.zeros(n, dtype=bool)
    for first, last in halves:
        inside[first:max(first, last + 1 - k)] = True
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bx[:n - k], by[:n - k] = tracking.ball_x[k:], tracking.ball_y[k:]
    out = np.zeros(n, dtype=bool)
    out[:n - k] = ball_out(tracking)[k:]
    dx = tracking.x - bx[:, None]
    dy = tracking.y - by[:, None]
    distance = np.sqrt(np.where(np.isnan(dx), np.inf, dx * dx + dy * dy))
    home = distance[:, tracking.slot_team == P.H].min(axis=1)
    away = distance[:, tracking.slot_team == P.A].min(axis=1)
    near, far = np.minimum(home, away), np.maximum(home, away)
    with np.errstate(invalid="ignore"):
        clear = (near <= P.FUTURE_RADIUS) & (far - near >= P.FUTURE_MARGIN) & ~out & inside
    return np.where(home <= away, P.H, P.A), clear


def tag_costs(tags, n):
    """(n, 3) costs: W_TAG against the team a tagged attacking phase says is not on the ball."""
    cost = np.zeros((n, 3))
    for code, start, end, side in tags:
        if code in P.ATTACK_CODES and side in (P.H, P.A):
            cost[max(0, start):min(n, end), P.A if side == P.H else P.H] += P.W_TAG
    return cost


def graded_mask(n=N_FRAMES, halves=HALVES):
    mask = np.zeros(n, dtype=bool)
    for first, last in halves:
        mask[first:last + 1] = True
    return mask


def costs(tracking, tags, halves=HALVES):
    """(n_frames, 3) costs for H, A, D, plus a dict of diagnostics for the console."""
    H, A, D = P.H, P.A, P.D
    n = len(tracking.ball_x)
    cost = np.zeros((n, 3))

    out = ball_out(tracking)
    team, clear = control(tracking)
    clear &= ~out                       # a ball out of play is not under anyone's control
    cost[clear & (team == H), A] += P.W_CONTROL
    cost[clear & (team == A), H] += P.W_CONTROL
    cost[clear, D] += P.W_CONTROL_DEAD
    cost[out, H] += P.W_OUT
    cost[out, A] += P.W_OUT
    touch_team, touch = touches(tracking)
    touch &= ~out
    if P.W_TOUCH:
        hold = np.ones(P.TOUCH_HOLD)
        for side, other in ((H, A), (A, H)):
            held = np.convolve((touch & (touch_team == side)).astype(float), hold)[:n] > 0
            cost[held, other] += P.W_TOUCH

    speed = pace(tracking)
    with np.errstate(invalid="ignore"):
        slow = np.nan_to_num(np.clip((P.PACE_SLOW - speed) / P.PACE_RAMP, 0.0, 1.0))
        live = np.nan_to_num(np.clip((speed - P.PACE_LIVE) / P.PACE_RAMP, 0.0, 1.0))
    cost[:, H] += P.W_PACE * slow
    cost[:, A] += P.W_PACE * slow
    cost[:, D] += P.W_PACE * live

    cost += tag_costs(tags, n)

    if P.W_FUTURE:
        for seconds in P.FUTURE_STEPS:
            reach, sure = future_reach(tracking, seconds, halves)
            cost[sure & (reach == H), A] += P.W_FUTURE / len(P.FUTURE_STEPS)
            cost[sure & (reach == A), H] += P.W_FUTURE / len(P.FUTURE_STEPS)

    if P.W_SHAPE:
        logit = shape.logit(tracking, halves)   # > 0: home
        cost[logit > 0, A] += P.W_SHAPE * logit[logit > 0]
        cost[logit < 0, H] -= P.W_SHAPE * logit[logit < 0]

    graded = graded_mask(n, halves)
    return cost, {
        "frames with clear control (%)": 100.0 * clear[graded].mean(),
        "frames with a touch (%)": 100.0 * touch[graded].mean(),
        "frames with the ball out (%)": 100.0 * out[graded].mean(),
        "frames of slow play (%)": 100.0 * (slow[graded] > 0).mean(),
        "frames of clearly live pace (%)": 100.0 * (live[graded] > 0).mean(),
    }
