"""Team shape: who has the ball, read from the players alone.

With no ball in sight, the players still show who has it. The team on the ball spreads out,
pushes its back line up, and the whole block moves towards the goal it attacks. features()
measures this per frame; logit() turns it into the log-odds that the home team has the ball,
with the coefficients fitted on the IDSSE matches (params.SHAPE_COEF, scripts/fit_team_shape.py).
"""

import warnings

import numpy as np
import pandas as pd

from . import params as P
from .cleaning import attack_direction
from .config import FPS, PITCH

NAMES = ("spread along", "spread across", "block speed", "territory", "last line",
         "front line", "opponents' half", "area", "running",
         "block speed 3 s", "spread across 5 s", "territory 5 s",
         "spread along 10 s", "spread across 10 s", "block speed 10 s", "territory 10 s",
         "last line 30 s")


def centred_mean(values, frames):
    """Centred rolling mean over `frames`, skipping NaN."""
    return pd.DataFrame(values).rolling(frames, center=True, min_periods=1).mean().values


def features(tracking, halves):
    """(n, 17) players-only features (NAMES), each home minus away or seen from the home side,
    so that swapping the teams flips their sign:

      spread along     standard deviation of x, home minus away (m)
      spread across    standard deviation of y, home minus away (m)
      block speed      mean velocity of all players towards the goal home attacks (m/s)
      territory        mean x of all players towards the goal home attacks, -1 to 1
      last line        distance of each team's deepest outfield player from its own goal, home
                       minus away (m): the team on the ball pushes its back line up
      front line       how much further forward home's most advanced player is than away's (m)
      opponents' half  players in the opponents' half, home minus away
      area             standard deviation of x times that of y, home minus away (100 m^2)
      running          mean player speed from positions, home minus away (m/s)
      then longer means: block speed over 3 s, spread across and territory over 5 s, the first
      four over 10 s, and the last line over 30 s, because possession lasts longer than a second
      and a team keeps its back line up for the whole possession.

    The first nine are centred means over params.SHAPE_WINDOW (1 s). Frames where a team has
    fewer than params.SHAPE_MIN_PLAYERS tracked count as missing; where the 1 s window has no
    frame left, every feature is 0 (no evidence).
    """
    x, y, team = tracking.x, tracking.y, tracking.slot_team
    home, away = team == P.H, team == P.A
    length = PITCH[0]
    direction = attack_direction(tracking, halves)
    home_goal = np.where(direction > 0, 0.0, length)
    away_goal = length - home_goal
    k = FPS // 2
    vx = np.full(x.shape, np.nan)
    vy = np.full(y.shape, np.nan)
    vx[k:-k] = (x[2 * k:] - x[:-2 * k]) * FPS / (2 * k)
    vy[k:-k] = (y[2 * k:] - y[:-2 * k]) * FPS / (2 * k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spread = [np.nanstd(v[:, team == side], axis=1) for v in (x, y) for side in (P.H, P.A)]
        to_own = (np.abs(x[:, home] - home_goal[:, None]), np.abs(x[:, away] - away_goal[:, None]))
        # The deepest player is the keeper; the next one is the last outfield line.
        last = [np.sort(np.where(np.isnan(d), np.inf, d), axis=1)[:, 1] for d in to_own]
        front_home = np.nanmin(np.abs(x[:, home] - away_goal[:, None]), axis=1)
        front_away = np.nanmin(np.abs(x[:, away] - home_goal[:, None]), axis=1)
        speed = np.hypot(vx, vy)
        matrix = np.stack([
            spread[0] - spread[1],
            spread[2] - spread[3],
            np.nanmean(vx, axis=1) * direction,
            (np.nanmean(x, axis=1) - length / 2) * direction / (length / 2),
            np.where(np.isfinite(last[0] + last[1]), last[0] - last[1], np.nan),
            front_away - front_home,
            ((to_own[0] > length / 2).sum(axis=1) - (to_own[1] > length / 2).sum(axis=1)) * 1.0,
            (spread[0] * spread[2] - spread[1] * spread[3]) / 100.0,
            np.nanmean(speed[:, home], axis=1) - np.nanmean(speed[:, away], axis=1)], axis=1)
    tracked = np.isfinite(x)
    enough = ((tracked[:, home].sum(axis=1) >= P.SHAPE_MIN_PLAYERS)
              & (tracked[:, away].sum(axis=1) >= P.SHAPE_MIN_PLAYERS))
    matrix[~enough] = np.nan
    short = centred_mean(matrix, P.SHAPE_WINDOW)
    out = np.concatenate([
        short,
        centred_mean(matrix[:, 2:3], 3 * FPS),
        centred_mean(matrix[:, [1, 3]], 5 * FPS),
        centred_mean(matrix[:, :4], 10 * FPS),
        centred_mean(matrix[:, 4:5], 30 * FPS)], axis=1)
    out[~np.isfinite(short[:, 0])] = np.nan
    return np.nan_to_num(out)


def logit(tracking, halves):
    """Per frame: log-odds that the home team has the ball (> 0: home), 0 without evidence."""
    return features(tracking, halves) @ np.asarray(P.SHAPE_COEF)
