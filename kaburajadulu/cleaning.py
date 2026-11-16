"""Automatic fixes to the provided tracking, the same on every match. Nothing is corrected by hand.
"""

import copy
import warnings

import numpy as np

from . import params as P
from .config import PITCH


def attack_direction(tracking, halves):
    """Per frame: +1 where the home team attacks towards x = PITCH[0], -1 towards x = 0, 0 off the
    halves. Defenders stand nearer their own goal than attackers, goalkeepers included, so over a
    half the home team's mean position lies on the side it defends."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = (np.nanmean(tracking.x[:, tracking.slot_team == P.H], axis=1)
               - np.nanmean(tracking.x[:, tracking.slot_team == P.A], axis=1))
    direction = np.zeros(len(gap))
    for first, last in halves:
        direction[first:last + 1] = -np.sign(np.nanmean(gap[first:last + 1]))
    return direction


def clean_goalkeepers(tracking, halves):
    """(copy of tracking, frames changed): goalkeeper tracks on the wrong half corrected.

    The tracker sometimes hands one goalkeeper's track to the other, so a team's keeper appears
    in the opponents' goal (our match: the away keeper's track is in the home penalty area on 8%
    of its first-half frames). A keeper never stands in the opponents' half in open play, so a
    keeper track there belongs to the other keeper: the two tracks are swapped when the other is
    also on the wrong side or missing, and the wrong one is dropped when the other keeper is
    already in place. Each team's keeper is found per half as the player most often inside its
    own penalty area.
    """
    out = copy.copy(tracking)
    out.x, out.y, out.speed = tracking.x.copy(), tracking.y.copy(), tracking.speed.copy()
    direction = attack_direction(tracking, halves)
    length = PITCH[0]
    changed = 0
    for first, last in halves:
        part = slice(first, last + 1)
        home_goal = 0.0 if direction[first] > 0 else length
        keeper, wrong = {}, {}
        for side, goal in ((P.H, home_goal), (P.A, length - home_goal)):
            slots = np.flatnonzero(tracking.slot_team == side)
            with np.errstate(invalid="ignore"):
                inside = np.abs(tracking.x[part][:, slots] - goal) < 16.5
                share = np.where(np.isfinite(tracking.x[part][:, slots]), inside, False).mean(axis=0)
                keeper[side] = slots[int(np.argmax(share))]
                wrong[side] = np.abs(tracking.x[part, keeper[side]] - goal) > length / 2
        present = {side: np.isfinite(tracking.x[part, keeper[side]]) for side in (P.H, P.A)}
        swap = (wrong[P.H] & (wrong[P.A] | ~present[P.A])) | (wrong[P.A] & ~present[P.H])
        drop = {side: wrong[side] & ~swap for side in (P.H, P.A)}
        rows = np.flatnonzero(swap) + first
        for values in (out.x, out.y, out.speed):
            values[rows, keeper[P.H]], values[rows, keeper[P.A]] = (values[rows, keeper[P.A]],
                                                                    values[rows, keeper[P.H]])
            for side in (P.H, P.A):
                values[np.flatnonzero(drop[side]) + first, keeper[side]] = np.nan
        changed += int(swap.sum() + drop[P.H].sum() + drop[P.A].sum())
    return out, changed
