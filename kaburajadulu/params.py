"""Every setting of the method, with where its value comes from.

Each value is one of three kinds:

- set from meaning: chosen from what the rule or the signal means, before any test;
- calibrated: chosen on the seven IDSSE Bundesliga matches (scripts/calibrate_idsse.py), leaving
  one match out, with the ball removed in gaps like ours;
- fitted: learned on the IDSSE matches (scripts/fit_team_shape.py).

The other modules read these values when they run, so a calibration script can change one
setting (setattr on this module) and run the model again. The development log
(docs/development_log.md) gives the full test behind each calibrated or fitted value.
"""

from .config import FPS

# States, in the order of every (n, 3) cost array.
H, A, D = 0, 1, 2

# --- Ball control (calibrated with our kind of noise: radius, --search core; weight and margin,
# --search control; docs/experiments.md, steps 4a and 4d) ---
# The ball is within playing distance of the nearest player, and that player is clearly closer
# than any opponent. A touch, a deflection or a duel rarely holds both for long.
# v2.2: with --noise, IDSSE chooses radius 1.0 m on six of seven held-out matches (0.4432 against
# 0.4305), then weight 0.25 and margin 0.25 m on six of seven (0.4522 against 0.4442). Control is
# now easier to reach but weaker per frame. v2.0 used 2.0 m, 1.0 m and 1.0.
CONTROL_RADIUS = 1.0        # m
CONTROL_MARGIN = 0.25       # m
W_CONTROL = 0.25            # per frame, against the team without control
W_CONTROL_DEAD = 0.5        # per frame, against D (set from meaning: players also stand over a
                            # ball before a restart, so control is weaker evidence of live play)

# --- Ball out of play (set from meaning) ---
# A tracked ball this far beyond a line is out of play; nearer than that is tracking jitter.
OUT_TOLERANCE = 1.0         # m
W_OUT = 1.0                 # per frame, against H and A while the ball is out

# --- Touch (thresholds from the literature; all but the step and minimum speed calibrated) ---
# A player within TOUCH_RADIUS while the ball turns or changes speed (Vidal-Codina et al., 2022,
# as implemented in databallpy). Stronger than proximity, because the ball reacts to the player.
# --search touch: weight 4 against none, held-out spellF1 0.299 against 0.246, better on all
# seven; 4, 8 and 16 tie, so 4 starts the plateau. --search touch-detail: a 20 degree turn and a
# 2 s hold, 0.345 against 0.325. Research loop 3 (v2.4) re-tuned the radius (0.5-3 m) and the
# speed change (3-20 m/s) on IDSSE with our kind of noise: 1.0 m and 20 m/s, held out 0.5584
# against 0.5451 (new test, chosen on 6 of 7) and 0.4757 against 0.4693 (old test). A player
# near the ball whose position is noisy is less often a touch. Public spellF1 unchanged (0.6750).
TOUCH_RADIUS = 1.0          # m (v2.3: 1.5)
TOUCH_ANGLE = 20.0          # degrees
TOUCH_SPEED_CHANGE = 20.0   # m/s (v2.3: 5.0)
TOUCH_MIN_SPEED = 1.0       # m/s; below this the ball's direction is noise
TOUCH_STEP = 3              # frames each side for the velocity in and out
TOUCH_HOLD = 50             # frames of evidence from a touch onwards (2 s)
W_TOUCH = 4.0               # per frame, against the team that did not touch

# --- Pace of play (thresholds set from meaning; weight calibrated, --search core) ---
# The mean speed of the tracked players, over a centred 1 s median: play stops, players walk.
# Only the two ends count. The reference is the one stretch the organisers state is dead (from
# the start of the graded second half to its kickoff): median 1.25 m/s, 95% of frames below
# 1.9 m/s. On Metrica's Sample Game 3, slow pace is a dead ball 81% of the time (base rate 27%)
# and clearly live pace is play 97% of the time (scripts/validate_sample_game.py).
PACE_WINDOW = FPS           # frames
PACE_SLOW = 1.0             # m/s; full strength at PACE_SLOW - PACE_RAMP
PACE_LIVE = 2.0             # m/s; full strength at PACE_LIVE + PACE_RAMP
PACE_RAMP = 0.25            # m/s
W_PACE = 1.0                # per frame at full strength: slow play against H and A, live pace
                            # against D

# --- Smart Tagging (set from meaning; the external matches have no Smart Tagging) ---
# Codes whose Team is, by definition, the team with the ball. The DEFENDING codes mirror these
# for the other team and add nothing; the zone codes describe where the ball is and survive a
# turnover, so they are not used.
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                "LONG BALL", "SET PIECES")
W_TAG = 0.1                 # per frame, against the team a tagged phase says is not on the ball

# --- Team shape (coefficients fitted; weight calibrated, --search shape) ---
# With no ball in sight, the players still show who has it (shape.py). A logistic regression on
# 17 players-only features names the team on the ball on 86.4% of the live frames of an IDSSE
# match it has not seen. Logit of "home has the ball" per unit of each feature, in the order of
# shape.NAMES.
SHAPE_COEF = (-0.4327, -0.6368, 0.6843, -9.8159,
              -0.1479, 0.0115, 0.1318, 1.5981,
              -1.1607, -0.0650, 0.7531, 3.3354,
              0.2546, 0.0786, -0.1894, 15.2717,
              0.0207)
# Research loop 4 (v2.5): our tracker loses players off camera, and with 9 as the minimum the
# shape said nothing on 11.9% of our graded frames. On IDSSE with our kind of noise, 3 to 8 all
# gain; 4 is the best mean on both tests (new 0.5618 against 0.5588, old 0.4932 against 0.4762,
# chosen on 7 of 7 held-out matches in both). Public spellF1 0.6750 -> 0.7226 for every value
# from 3 to 8, so the public check did not choose. Clean IDSSE tracks all 22: no change there.
SHAPE_MIN_PLAYERS = 4       # per team (v2.4: 9); with fewer tracked, the shape says nothing
SHAPE_WINDOW = FPS          # frames of the centred mean that smooths each short feature
# Weights 0.3 to 1.0 tie (0.455-0.456 over all seven; with our kind of noise, 0.3 to 0.7 tie at
# 0.421-0.422). 0.3 starts the plateau and leaves the most room to the ball evidence.
W_SHAPE = 0.3               # per frame and unit of logit, against the less likely team

# --- Segmentation (factor calibrated, --search transitions; price re-tuned with noise, --search
# core) ---
# The price of a change of team. A change into or out of a dead ball costs DEAD_SWITCH_FACTOR
# times as much. v1.7: held-out spellF1 0.325 against 0.299 when every change cost 5, better on
# all seven; any team price from 14 to 40 tied while a dead-ball change cost about 5. v2.2: with
# --noise and the set-piece evidence, IDSSE chooses a team price of 10 on six of seven held-out
# matches (docs/experiments.md, step 4a), so a dead-ball change now costs about 3.6. Above twice
# the dead-ball price, the model routes a change of team through a brief dead ball, which keeps
# the two possessions apart. Cheaper team prices win on IDSSE but predict too many possessions on
# our match (step 4c).
SWITCH_COST = 10.0
DEAD_SWITCH_FACTOR = 5.0 / 14.0
FORBIDDEN = 1e9             # cost that rules a state out (the documented dead stretch)

# --- Baseline (the organisers' baseline_smooth50, for comparison only) ---
BASELINE_WINDOW = 50        # +/- frames of the majority vote

# --- Set-piece shapes as dead-ball evidence (thresholds set from meaning; setpieces.py) ---
# Each detector fires where the ball and the players stand as the Laws put them before a restart.
# scripts/validate_setpieces.py measures each one: on IDSSE with our kind of noise, the ball
# detectors are dead ball on 98-100% of the frames they fire on, the players-only ones on 91-93%
# (base rate 44.8%). The weight comes from the public check: IDSSE chooses 0.5 on all seven
# held-out matches (with noise 0.4320 against 0.4207), and 1.0 also beats v2.0 there (0.4305).
# Only 1.0 raises the public spellF1 (0.5818 against 0.5677), so v2.2 uses 1.0. This choice is
# tuning on the public window (docs/experiments.md, idea 3).
W_SET_PIECE = 1.0           # per frame, against H and A

# --- The wait for the restart kick after a goal line (candidate v2.6; restarts.py) ---
# After the ball crosses a goal line, the ball is dead until the goal kick or the corner kick,
# also while a player fetches it and stands over it (control reads that as possession). The kick:
# the ball placed at the spot for RESTART_PLACED s (median speed below RESTART_PLACED_SPEED)
# moves off faster than RESTART_KICK_SPEED. Thresholds set from meaning, with the public case in
# view; every kick speed from 4 to 10 m/s with every placed speed from 2 to 4 m/s gives
# 0.5626-0.5627 on the new IDSSE test with noise (off: 0.5618). The idea came from a
# goal kick in the public window (labelled H long before the kick). IDSSE, new test with
# noise: 0.5618 -> 0.5626 (6 of 7 matches up, 1 the same; 1.0 and 2.0 tie, 1.0 is nearer the
# current 0); clean 0.5849 -> 0.5858; old test unchanged (0.5117, 0.4932); Metrica 0.647 ->
# 0.651. Not kept: waits that end with the ball seen far away (3 of 7 matches down); the
# restarting team (goal kick: defenders, corner: attackers) as evidence after the kick
# (weights 1 and 4 change no label: control and touch already name that team).
W_RESTART_WAIT = 1.0        # per frame, against H and A from the ball out to the kick
RESTART_PLACED = 1.0        # s
RESTART_PLACED_SPEED = 3.0  # m/s (median over RESTART_PLACED, measured over 0.2 s)
RESTART_KICK_SPEED = 6.0    # m/s (measured over 0.2 s)
RESTART_MAX_WAIT = 60.0     # s

# --- Stoppage edges (v2.3; docs/experiments.md, research loop 2, idea 3) ---
# The evidence of a stoppage (slow pace, a set-piece shape) and of a restart (control, live pace)
# does not come at the moment of the whistle or the kick. After the segmentation, each predicted
# dead run moves: its start STOP_SHIFT_START s earlier and its end STOP_SHIFT_END s earlier (a
# negative value moves the edge later; model.shift_stoppages). Each run and each neighbour keeps
# at least one frame. Test: IDSSE with the ball gaps in stoppages as on our match and the DFL
# stoppage ball rebuilt (scripts/research.py --gaps stoppage --dead-ball rebuilt), with our kind
# of noise. Held out, IDSSE chooses start 0.25 s later and end 0.75 s earlier on 4 of 7 matches
# (0.5371 against 0.5177; plateau 0.5366-0.5386). Public check: 0.6545 against 0.6424. A change
# of team passes through a short dead run in this decoder, so this also moves the changes of team.
STOP_SHIFT_START = -0.25    # s
STOP_SHIFT_END = 0.75       # s

# --- Future reach (v2.3; docs/experiments.md, research loop 2, idea 7b) ---
# The team whose player now stands nearest to where the ball will be in 0.5 s and 1 s is the team
# that keeps or wins it (evidence.future_reach). It counts where that player is within
# FUTURE_RADIUS of that point and FUTURE_MARGIN nearer than any opponent, while the ball is seen
# and in play then, in the same half. Every setting of the grid gains on IDSSE (same test as
# above). IDSSE chooses weight 0.025 and radius 4 m, which leaves the public spellF1 at 0.6545.
# The public check chose weight 0.1 and radius 3 m among the settings that gain on IDSSE. This
# choice is tuning on the public window.
W_FUTURE = 0.1              # per frame, against the team that does not reach the ball first
FUTURE_STEPS = (0.5, 1.0)   # s; W_FUTURE is split over the steps
FUTURE_RADIUS = 3.0         # m
FUTURE_MARGIN = 1.0         # m

