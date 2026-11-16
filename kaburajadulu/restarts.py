"""The wait for the restart kick after the ball crosses a goal line (W_RESTART_WAIT).

When the ball goes over a goal line, play restarts with a goal kick or a corner kick. Until the
kick the ball is dead, also while a player fetches the ball, carries it and stands over it (the
control evidence reads that as possession). The kick is the first time the ball, placed at a
restart spot at that end (the goal area or a corner flag), moves off fast.

restart_wait() marks the frames from the ball going out to that kick; model.costs adds
W_RESTART_WAIT to H and A over them. Only a wait that ends with such a kick counts: on IDSSE
these waits are 98% dead ball. If the ball is seen far from that end first (a goal and its
kickoff, or a quick restart that the tracking missed), or if no kick comes within
RESTART_MAX_WAIT s, nothing is added (waits that end with the ball seen far away are 79-86%
dead ball on IDSSE and lowered three of the seven matches).

The thresholds are set from meaning (the Laws' distances, a placed ball, a kicked ball).
"""

import numpy as np

from . import params as P
from .config import FPS, PITCH
from .setpieces import GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH, PENALTY_DEPTH

GOAL_KICK, CORNER = 1, 2


def _spot(bx, by, end_x):
    """GOAL_KICK, CORNER or 0 per frame: the restart spot the ball is at, at the end end_x."""
    length, width = PITCH
    tol = 1.0
    with np.errstate(invalid="ignore"):
        goal_area = ((np.abs(bx - end_x) < GOAL_AREA_DEPTH + tol)
                     & (np.abs(by - width / 2) < GOAL_AREA_HALF_WIDTH + tol))
        corner = (np.abs(bx - end_x) < 3.0) & (np.minimum(by, width - by) < 3.0)
    return np.where(goal_area, GOAL_KICK, np.where(corner, CORNER, 0))


def restart_wait(tracking, halves, events=None):
    """True per frame from a ball over a goal line to its restart kick. A list passed as
    `events` gets (first frame, kick frame, GOAL_KICK or CORNER) for each wait."""
    length, width = PITCH
    bx, by = tracking.ball_x, tracking.ball_y
    n = len(bx)
    seen = np.isfinite(bx)
    step = max(1, int(round(0.2 * FPS)))
    speed = np.full(n, np.nan)                          # m/s over the next 0.2 s
    speed[:-step] = np.hypot(bx[step:] - bx[:-step], by[step:] - by[:-step]) * FPS / step
    tol = P.OUT_TOLERANCE
    with np.errstate(invalid="ignore"):
        over = seen & ((bx < -tol) | (bx > length + tol)) & (by > -tol) & (by < width + tol)
    placed = int(round(P.RESTART_PLACED * FPS))
    longest = int(round(P.RESTART_MAX_WAIT * FPS))

    wait = np.zeros(n, dtype=bool)
    for first, last in halves:
        t = first
        while t <= last:
            if not over[t]:
                t += 1
                continue
            end_x = 0.0 if bx[t] < length / 2 else length
            stop = min(last + 1, t + longest)
            spot = _spot(bx[t:stop], by[t:stop], end_x)
            far = seen[t:stop] & (np.abs(bx[t:stop] - end_x) > PENALTY_DEPTH + 3.0)
            kick = None
            for k in range(placed, stop - t):
                if far[k]:
                    break
                if spot[k] and speed[t + k] > P.RESTART_KICK_SPEED:
                    before = slice(k - placed, k)
                    at_spot = spot[before] == spot[k]
                    if (at_spot.mean() >= 0.8
                            and np.nanmedian(speed[t:t + k][before][at_spot])
                            < P.RESTART_PLACED_SPEED):
                        kick = k
                        break
            if kick is None:
                while t <= last and over[t]:
                    t += 1
                continue
            wait[t:t + kick] = True
            if events is not None:
                events.append((t, t + kick, int(spot[kick])))
            t += kick + 1
    return wait
