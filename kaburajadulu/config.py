"""Match constants and data paths. Every other module reads them from here.

All frame numbers are tactical-video frames, which is also the submission row index. The values
come from the organisers' documentation (DATA.md, FAQ.md, games.py).
"""

import os

TEAM = "KaburAjaDulu"
GAME = "BAR-ATH"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Same variable the organisers' repository uses, so one setting serves both.
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

N_FRAMES = 199319           # rows in a submission
FPS = 25
PITCH = (105.3, 68.0)       # metres; tracking coordinates are normalised 0-1

# Graded video frames per half, inclusive. Everything outside is never scored.
HALVES = ((11185, 90651), (110808, 190197))
# The second half is graded from 110808 but kicks off at 113742; the frames between are all D.
KICKOFFS = (11185, 113742)


def tactical_paths(data_dir=None):
    """Paths to the tactical recording's files, laid out as the bundle unpacks."""
    folder = os.path.join(data_dir or DATA_DIR, "tactical")
    stem = os.path.join(folder, "{}_tactical".format(GAME))
    return {
        "atd_meta": stem + "_atd.xml",
        "atd_raw": stem + "_atd.txt",
        "homography": stem + "_homography.csv",
        "smart_tagging": stem + "_smart_tagging.xml",
        "video": stem + ".mp4",
    }
