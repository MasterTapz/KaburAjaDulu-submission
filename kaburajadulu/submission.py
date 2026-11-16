"""Write and check the submission file: header frame,label, one row per tactical video frame."""

import os

import numpy as np

from .config import FPS, HALVES, N_FRAMES

LABELS = ("H", "A", "D", "X")


def check(labels):
    """None if the labels make a valid submission, else a one-line reason."""
    if len(labels) != N_FRAMES:
        return "row count {} != {}".format(len(labels), N_FRAMES)
    bad = sorted(set(np.unique(labels)) - set(LABELS))
    if bad:
        return "labels outside {}: {}".format(list(LABELS), bad[:5])
    return None


def write(path, labels):
    """Always '\\n' line endings, so the same labels give the same bytes on every OS."""
    reason = check(labels)
    if reason:
        raise ValueError("not a valid submission: " + reason)
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summary(labels):
    """(label counts on graded frames, number of possessions, mean possession length in s).

    A possession is a maximal run of H or of A inside one half.
    """
    graded = np.concatenate([labels[first:last + 1] for first, last in HALVES])
    counts = {label: int(np.sum(graded == label)) for label in LABELS}
    lengths = []
    for first, last in HALVES:
        half = labels[first:last + 1]
        starts = np.flatnonzero(np.concatenate(([True], half[1:] != half[:-1])))
        stops = np.concatenate((starts[1:], [len(half)]))
        keep = np.isin(half[starts], ("H", "A"))
        lengths.extend((stops - starts)[keep].tolist())
    mean_s = float(np.mean(lengths)) / FPS if lengths else 0.0
    return counts, len(lengths), mean_s
