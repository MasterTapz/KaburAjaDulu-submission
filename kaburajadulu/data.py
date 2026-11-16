"""Load the tactical tracking (ATD) into arrays indexed by video frame.

Raw line format: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed" with NaN for an
unseen object. frame_id counts from 1, video frames from 0, so record frame_id lands on row
frame_id - 1. Positions are normalised 0-1 and converted to metres here.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .config import FPS, N_FRAMES, PITCH, tactical_paths

# Params
PLAYER_FIELDS = 3   # x, y, speed


class Tracking(object):
    """Arrays over video frames. x, y: (n_frames, n_slots) metres; speed: same shape, m/s.
    ball_x, ball_y: (n_frames,). slot_team: 0 home, 1 away, per track slot. NaN wherever the
    tracker saw nothing."""


def slot_teams(meta_path):
    """0/1 per track slot, in the order of <Player> in the metadata. The first <Team> is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    return np.array([0 if player.get("teamId") == team_ids[0] else 1
                     for player in root.iter("Player")], dtype=np.int8)


def load_tracking(data_dir=None):
    paths = tactical_paths(data_dir)
    missing = [path for path in (paths["atd_meta"], paths["atd_raw"]) if not os.path.isfile(path)]
    if missing:
        raise IOError("match data not found:\n  {}\nPut the bundle in ./data or set "
                      "MTAH_DATA_DIR (or pass --data).".format("\n  ".join(missing)))

    teams = slot_teams(paths["atd_meta"])
    n_slots = len(teams)
    with open(paths["atd_raw"], "rb") as handle:
        raw = handle.read()
    # One separator for every field turns the file into a plain CSV the C parser reads in seconds.
    # round_trip parsing is correctly rounded, so the numbers match Python's float() everywhere.
    table = pd.read_csv(io.BytesIO(raw.translate(bytes.maketrans(b":;", b",,"))),
                        header=None, dtype=np.float64, engine="c",
                        float_precision="round_trip").values
    if table.shape[1] < 1 + PLAYER_FIELDS * n_slots + 2:
        raise ValueError("ATD has {} columns, expected at least {}".format(
            table.shape[1], 1 + PLAYER_FIELDS * n_slots + 2))

    rows = table[:, 0].astype(np.int64) - 1
    keep = (rows >= 0) & (rows < N_FRAMES)
    players = table[:, 1:1 + PLAYER_FIELDS * n_slots].reshape(-1, n_slots, PLAYER_FIELDS)
    ball = table[:, 1 + PLAYER_FIELDS * n_slots:]

    def placed(values, scale):
        out = np.full((N_FRAMES,) + values.shape[1:], np.nan)
        out[rows[keep]] = values[keep] * scale
        return out

    tracking = Tracking()
    tracking.slot_team = teams
    tracking.x = placed(players[:, :, 0], PITCH[0])
    tracking.y = placed(players[:, :, 1], PITCH[1])
    # The metadata labels this channel kph, but the values are m/s, as DATA.md says.
    tracking.speed = placed(players[:, :, 2], 1.0)
    tracking.ball_x = placed(ball[:, 0], PITCH[0])
    tracking.ball_y = placed(ball[:, 1], PITCH[1])
    return tracking


def load_tags(data_dir=None):
    """Smart Tagging as [(code, start_frame, end_frame, side)], end exclusive.

    <start>/<end> are seconds from the first tactical video frame, so seconds x FPS is a frame.
    side is 0 for Barcelona, 1 for Athletic and None for N/A (every throw-in and kickoff).
    """
    path = tactical_paths(data_dir)["smart_tagging"]
    tags = []
    for instance in ET.parse(path).getroot().iter("instance"):
        team = ""
        for label in instance.findall("label"):
            if (label.findtext("group") or "").strip().lower() == "team":
                team = (label.findtext("text") or "").strip().lower()
        side = 0 if "barcelona" in team else 1 if ("atletic" in team or "athletic" in team) \
            else None
        tags.append(((instance.findtext("code") or "").strip().upper(),
                     int(round(float(instance.findtext("start")) * FPS)),
                     int(round(float(instance.findtext("end")) * FPS)), side))
    return tags
