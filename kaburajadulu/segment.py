"""Cheapest labelling of a sequence of per-frame costs, charging a price per change of state.

This is the Viterbi algorithm with a switching cost. It turns frame-level evidence into segments:
a state changes only when the evidence against it adds up to more than the price of a change, and
the change lands on the frame where that evidence begins. Short contrary blips (a touch, a
deflection) cannot pay for two changes and are absorbed.

The price may vary by kind of change: factors[i, j] scales the price of going from state i to
state j, so a change of team and a change into or out of a dead ball can cost different amounts.
"""

import numpy as np


def viterbi(cost, switch, factors=None):
    """State index per frame minimising sum(cost[t, state_t]) plus the price of every change.

    switch is one price for every frame, or an array: switch[t] is the price of changing state
    between frame t - 1 and frame t. factors, if given, is a (k, k) array scaling that price per
    kind of change (the diagonal is ignored). Ties keep the current state.
    """
    n, k = cost.shape
    price = np.broadcast_to(np.asarray(switch, dtype=np.float64), (n,))
    states = np.arange(k)
    back = np.empty((n, k), dtype=np.int8)
    back[0] = states
    total = cost[0].copy()
    if factors is None:
        for t in range(1, n):
            best = int(np.argmin(total))
            moved = total[best] + price[t]
            change = moved < total
            back[t] = np.where(change, best, states)
            total = np.where(change, moved, total) + cost[t]
    else:
        scale = np.array(factors, dtype=np.float64)
        scale[states, states] = 0.0
        for t in range(1, n):
            candidates = total[:, None] + price[t] * scale
            best = np.argmin(candidates, axis=0)
            keep = candidates[states, states] <= candidates[best, states]
            back[t] = np.where(keep, states, best)
            total = np.where(keep, total, candidates[best, states]) + cost[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmin(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def marginals(cost, switch, factors=None):
    """(n, k) probability of each state per frame under the same costs and prices.

    Reads each cost as minus a log-likelihood and each price as minus the log-chance of that
    change, and sums over every labelling instead of keeping the cheapest (forward-backward).
    How sure the model is, frame by frame: a diagnostic, not used for the labels.
    """
    n, k = cost.shape
    price = np.broadcast_to(np.asarray(switch, dtype=np.float64), (n,))
    scale = np.ones((k, k)) if factors is None else np.array(factors, dtype=np.float64)
    states = np.arange(k)
    scale[states, states] = 0.0
    forward = np.empty((n, k))
    forward[0] = -cost[0]
    for t in range(1, n):
        m = forward[t - 1][:, None] - price[t] * scale          # [from, to]
        top = m.max(axis=0)
        forward[t] = top + np.log(np.exp(m - top).sum(axis=0)) - cost[t]
    backward = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        m = (backward[t + 1] - cost[t + 1])[None, :] - price[t + 1] * scale
        top = m.max(axis=1)
        backward[t] = top + np.log(np.exp(m - top[:, None]).sum(axis=1))
    log_p = forward + backward
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)
