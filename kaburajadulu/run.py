"""Build the submission from the provided data. This is the command that reproduces our file.

    python -m kaburajadulu.run                          # data in ./data, writes output/KaburAjaDulu.csv
    python -m kaburajadulu.run --data D:/mtah2027       # data somewhere else
    python -m kaburajadulu.run --baseline               # the organisers' baseline, for comparison
"""

import argparse
import os
import time

from . import model, submission
from .config import OUTPUT_DIR, TEAM
from .data import load_tags, load_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="bundle folder holding tactical/ (default: ./data or "
                                       "$MTAH_DATA_DIR)")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, TEAM + ".csv"))
    parser.add_argument("--baseline", action="store_true",
                        help="build the organisers' baseline (baseline_smooth50) instead")
    parser.add_argument("--window", type=int, default=None,
                        help="--baseline only: half-width of the smoothing vote, in frames")
    args = parser.parse_args()

    started = time.time()
    tracking = load_tracking(args.data)
    diagnostics = {}
    if args.baseline:
        labels = model.baseline(tracking, window=args.window)
    else:
        labels, diagnostics = model.predict(tracking, load_tags(args.data))
    for name, value in diagnostics.items():
        print("{:<32}{:.3f}".format(name, value))
    path = submission.write(args.out, labels)

    counts, possessions, mean_s = submission.summary(labels)
    graded = sum(counts.values())
    print("graded frames  H {H:,}  A {A:,}  D {D:,}  X {X:,}".format(**counts))
    print("dead ball      {:.1f}% of graded frames".format(100.0 * counts["D"] / graded))
    print("possessions    {:,}, mean {:.1f} s".format(possessions, mean_s))
    print("wrote {} in {:.0f}s".format(path, time.time() - started))


if __name__ == "__main__":
    main()
