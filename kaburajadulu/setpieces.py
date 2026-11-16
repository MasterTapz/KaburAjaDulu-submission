"""Set-piece shapes as dead-ball evidence (W_SET_PIECE).

Before a restart, the Laws of the Game put the players in a known shape. Each detector below
fires on a frame where the tracked ball and players stand in that shape:

- corner: the ball at a corner flag, many players in the penalty area at that end;
- throw-in: the ball on a touchline, a player beyond that line next to it;
- goal kick: the ball in a goal area, no attacker in that penalty area, a defender at it;
- corner (players) and throw-in (players): the same from the players only, for ball gaps.

The three ball detectors also need the ball to move less than 0.5 m over 1 s. A kickoff
detector never fired, on IDSSE or on our match; it is in commit 9650e7e.

The thresholds are set from meaning (the Laws' distances plus an allowance for tracking error).
scripts/validate_setpieces.py measures each detector's precision and recall on IDSSE and on
Metrica's Sample Game 3 before any use.
"""

import numpy as np

from . import params as P
from .cleaning import attack_direction
from .config import FPS, PITCH

PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16     # m (Laws of the Game)
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16   # m


def _still(tracking, seconds=1.0, radius=0.5):
    """True where the tracked ball moves less than radius over the surrounding `seconds`."""
    k = int(round(seconds * FPS / 2))
    x, y = tracking.ball_x, tracking.ball_y
    dx = np.full(len(x), np.nan)
    dy = np.full(len(y), np.nan)
    dx[k:-k] = x[2 * k:] - x[:-2 * k]
    dy[k:-k] = y[2 * k:] - y[:-2 * k]
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (np.hypot(dx, dy) < radius)


def detect(tracking, halves):
    """{name: bool per frame} for the five set-piece shapes."""
    length, width = PITCH
    x, y = tracking.x, tracking.y
    bx, by = tracking.ball_x, tracking.ball_y
    seen = np.isfinite(bx)
    home = tracking.slot_team == P.H
    away = tracking.slot_team == P.A
    direction = attack_direction(tracking, halves)       # +1: home attacks x = length
    tracked = np.isfinite(x)
    tol = 1.0
    # The ball waits for every restart; once it is kicked, the shape is no longer evidence.
    still = seen & _still(tracking)
    out = {}

    with np.errstate(invalid="ignore"):
        near_ball = np.hypot(x - bx[:, None], y - by[:, None]) < 3.0

        # Corner: ball within 3 m of a corner flag; 6 or more players in the penalty area at
        # that end.
        end_x = np.where(bx < length / 2, 0.0, length)
        corner = (np.abs(bx - end_x) < 3.0) & (np.minimum(by, width - by) < 3.0)
        in_box = ((np.abs(x - end_x[:, None]) < PENALTY_DEPTH)
                  & (np.abs(y - width / 2) < PENALTY_HALF_WIDTH))
        out["corner"] = still & corner & (in_box.sum(axis=1) >= 6)

        # Throw-in: ball within 1.5 m of a touchline (or beyond it), away from the corners; a
        # player more than 0.5 m beyond that touchline and within 3 m of the ball.
        side_y = np.where(by < width / 2, 0.0, width)
        on_line = (np.abs(by - side_y) < 1.5) | (np.minimum(by, width - by) < 0)
        away_from_corner = (bx > 3.0) & (bx < length - 3.0)
        beyond = np.where(side_y[:, None] == 0.0, y < -0.5, y > width + 0.5)
        out["throw-in"] = still & on_line & away_from_corner & (beyond & near_ball).any(axis=1)

        # Goal kick: ball still for 1 s inside a goal area (tol allowed); no attacker in that
        # penalty area; a defender within 5 m of the ball.
        goal_x = np.where(bx < length / 2, 0.0, length)
        in_goal_area = ((np.abs(bx - goal_x) < GOAL_AREA_DEPTH + tol)
                        & (np.abs(by - width / 2) < GOAL_AREA_HALF_WIDTH + tol))
        # The team that defends the goal at goal_x: home if it attacks the other end.
        home_defends = np.where(goal_x == 0.0, direction > 0, direction < 0)
        attacker = np.where(home_defends[:, None], away[None, :], home[None, :])
        in_area = ((np.abs(x - goal_x[:, None]) < PENALTY_DEPTH)
                   & (np.abs(y - width / 2) < PENALTY_HALF_WIDTH))
        near5 = np.hypot(x - bx[:, None], y - by[:, None]) < 5.0
        defender_near = (near5 & ~attacker & tracked).any(axis=1)
        out["goal kick"] = (still & in_goal_area & ~(in_area & attacker).any(axis=1)
                            & defender_near & (direction != 0))

        corner_takers, throw_takers = _takers(tracking)
        out["corner (players)"] = corner_takers.any(axis=1)
        out["throw-in (players)"] = throw_takers.any(axis=1)
    return out


def _takers(tracking):
    """(corner, throw-in): (n_frames, n_slots) masks of the players who stand to take a
    restart, from the players only (our ball is often lost while play stops).

    Corner: a player within 2 m of a corner flag, with 8 or more players in the penalty area at
    that end. Throw-in: a player more than 0.5 m beyond a touchline, away from the corners, with
    a player on the pitch within 15 m. A track off the pitch with no player on the pitch near it
    is a substitute, a coach or an official; two such tracks together are too.
    """
    length, width = PITCH
    x, y = tracking.x, tracking.y
    corner = np.zeros(x.shape, dtype=bool)
    throw = np.zeros(x.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for end in (0.0, length):
            at_flag = (np.abs(x - end) < 2.0) & (np.minimum(y, width - y) < 2.0)
            box = (np.abs(x - end) < PENALTY_DEPTH) & (np.abs(y - width / 2) < PENALTY_HALF_WIDTH)
            corner |= at_flag & (box.sum(axis=1) >= 8)[:, None]
        on_pitch = ((x >= -0.5) & (x <= length + 0.5) & (y >= -0.5) & (y <= width + 0.5))
        off = ((y < -0.5) | (y > width + 0.5)) & (x > 3.0) & (x < length - 3.0)
        rows = np.flatnonzero(off.any(axis=1))
        for block in np.array_split(rows, max(1, len(rows) // 2000)):
            xs, ys = x[block], y[block]
            gap = np.hypot(xs[:, :, None] - xs[:, None, :], ys[:, :, None] - ys[:, None, :])
            near = np.nan_to_num(gap, nan=np.inf) < 15.0
            company = (near & on_pitch[block][:, None, :]).any(axis=2)
            throw[block] = off[block] & company
    return corner, throw
