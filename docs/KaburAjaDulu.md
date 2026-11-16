# Team KaburAjaDulu: technical document

More than a Hack 2027, Phase 1: frame-level ball possession. FC Barcelona (home, `H`) vs Athletic
Club (away, `A`), LaLiga 2024-25, matchday 2. Submitted version: v2.6.
Code: https://github.com/MasterTapz/KaburAjaDulu-submission

## 1. Approach and rationale

`spellF1` scores whole possessions. If one frame in a possession has the wrong team, the possession
breaks into two parts and does not count as correct. Also, the ball position is missing on 41% of
the graded frames of the competition match, mostly in gaps longer than 10 s, and play often
continues in these gaps. In a gap, a frame-by-frame classifier usually flickers. For this reason,
the method labels the match in segments.

Each signal gives a cost per frame for `H`, `A` and `D` (see the table below). A cost of 0 means
that the frame has no evidence against that label. The Viterbi algorithm (Viterbi, 1967) finds the
labels with the lowest total cost for each half. Each change of label has a price: 10 for a change
of team, and 3.6 for a change into or out of a dead ball. Thus, the label changes only when the
evidence against it is more than that price (section 5 gives the possession rules). In the table,
"noisy IDSSE" is IDSSE with the noise of the competition tracking (section 3).

The evidence of a stoppage and of a restart does not come at the moment of the whistle or the kick.
Thus, a fixed step moves the start of each predicted stoppage 0.25 s later and its end 0.75 s
earlier (the best fit on IDSSE). Frames outside the graded halves are `X`, and the second half is
`D` from the start of grading to its kickoff, as the FAQ says.

| Evidence | When it applies | Cost per frame | Source of the value |
|---|---|---|---|
| Clear control | ball within 1.0 m of the nearest player, each opponent at least 0.25 m further away | +0.25 other team, +0.5 `D` | tuned on noisy IDSSE |
| Touch | a player within 1.0 m when the ball turns more than 20° or changes speed by more than 20 m/s | +4.0 other team, for 2 s | detector from Vidal-Codina et al. (2022); radius and speed change tuned on noisy IDSSE |
| Team shape | the players alone: spread, line heights, block movement (section 3) | +0.3 × \|logit\| to the less likely team | fitted and tuned on IDSSE |
| Future reach | a player within 3 m of the ball position 0.5 s or 1 s later, each opponent at least 1 m further away | +0.05 per step, other team | distances fixed in advance; weight from the public check |
| Ball out | tracked ball more than 1.0 m over a line | +1.0 `H` and `A` | fixed in advance |
| Set-piece shape | ball and players placed as before a corner, throw-in or goal kick (5 detectors, 93% precision on IDSSE) | +1.0 `H` and `A` | shapes from the Laws of the Game; weight from the public check |
| Restart wait | ball over a goal line, until a kick from the goal area or a corner flag (98% precision on IDSSE) | +1.0 `H` and `A` | thresholds fixed in advance; weight tuned on noisy IDSSE |
| Slow pace | mean player speed less than 1.0 m/s (1 s median) | up to +1.0 `H` and `A` | threshold fixed in advance; weight tuned on IDSSE |
| Live pace | mean player speed more than 2.0 m/s | up to +1.0 `D` | threshold fixed in advance; weight tuned on IDSSE |
| Smart Tagging | an attacking phase tagged for one team | +0.1 other team | fixed in advance |

The organisers' baseline uses the nearest player, then a majority vote to smooth the labels. It
labels each frame without a ball as `D`. Its single window size must balance flicker against late
changes. A learned version of the same evidence (v2, a logistic regression on all the signals) was
also tested. On held-out IDSSE matches, it was not better than the hand-built model. Thus, the
submission uses the simpler model, in which each weight has a stated meaning.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players and ball) | the input to all tracking evidence and to the goalkeeper fix |
| Smart Tagging | a weak input: an attacking phase names the team with the ball (+0.1) |
| Tactical homography and video | review only: the labels were drawn on the video to examine errors. Not a model input |
| Tactical and broadcast audio | tested, not used. The whistle detector needs the whistle in both recordings. 98% of its whistles have a dead ball within 10 s, but the audio does not show when the stoppage ends, so it split possessions |
| Broadcast video, ATD and homography | not used by the model. Aligned to the tactical recording by the two ball tracks (one offset per half, correct to the frame) to test whistles, close-ups and replays as dead-ball evidence. None increased public spellF1 |

Three external datasets were used. None of their labels comes from the competition match.

| Dataset | Source and licence | Labels | Size | Used for |
|---|---|---|---|---|
| IDSSE, 7 Bundesliga matches | DFL open data (Bassek et al., 2025); CC-BY 4.0; Hugging Face mirror `pysport/idsse-data` | DFL's per-frame ball status (alive or dead) and team in possession | 1,002,644 frames (about 11 h), 3,061 possessions | team-shape fit (17 coefficients); weights and prices; the set-piece check; the dropped learned model |
| Metrica Sample Games 1, 2 and 3 | Metrica Sports, GitHub `metrica-sports/sample-data` | event data with all stoppages | 3 matches | the dead-ball signal check (Game 3); the first price calibration (v1.4, replaced later) |
| SkillCorner open data, 20 A-League matches | GitHub `SkillCorner/opendata`, MIT licence | team and player in possession per frame | 1.6 million frames | tested, not used. With it, the team-shape fit had a lower held-out IDSSE accuracy (0.856 against 0.863) |

Some signals failed the held-out tests. On Metrica Sample Game 3, two dead-ball signals failed: a
static ball, and a ball lost as it moves out of play. A fixed timing shift matched the DFL labelling
convention, not the competition convention. A restart wait after each ball out had a lower IDSSE
score in three versions. Thus, v2.6 keeps only the goal-line case that ends with a detected kick.
Evidence from the restarting team after that kick changed no label.

## 3. Missing and noisy tracking

The competition tracking is automatic, and nobody corrected it. Each fix below is the same for all
matches.

The ball is missing on 41% of the graded frames: 72.5% of the dead frames and 11.9% of the live
frames. In a gap, there is no control or touch evidence. Thus, only other evidence can change the
label. The pace evidence does not need the ball. The team shape finds possession from the players
alone. The team with the ball spreads out and moves its back line up, and the full block moves
toward the goal that it attacks. The team shape is a logistic regression (Pedregosa et al., 2011) on
17 such features: the spread of each team, the height of each back line and front line, the players
in the opponents' half, and the velocity and position of the block, over 1, 5, 10 and 30 s. On a
held-out IDSSE match, it finds the team with the ball on 86% of the live frames. Before the team
shape, the team was wrong on 34% of the gap frames, against 6% of the frames with a ball.

Players also drop out in the gaps. In the ball gaps of the competition match, the median frame has
21 tracked players, and one frame in ten has 13 or fewer. The team shape works with a minimum of 4
tracked players per team (v2.5). With fewer, it gives no evidence (on 7.9% of the gap frames).

Tracks also get mixed. On about 12% of the frames, one team has more than 11 tracks. Sometimes the
tracker gives the track of one goalkeeper to the other, and the wrong team gets control near the
goal. In open play, a goalkeeper is never in the opponents' half. Thus, the pipeline gives such a
track to the other goalkeeper, or removes it if the other goalkeeper already has a track. This rule
has no parameter and changes 18,002 frames of the competition match.

The IDSSE tracking is complete. Thus, each IDSSE match gets a copy with the noise of the competition
tracking (noisy IDSSE). The ball is removed in random gaps (lognormal, median 10.7 s) until 41% of
the frames have no ball. Players are removed in the gaps to match the competition player counts, and
on 11.4% of the frames, players are copied into the tracks of the other team. A stricter test puts
most gaps in stoppages, as in the competition match, and lets the stoppage ball roll and move to the
restart position (DFL freezes it).

## 4. How the settings were chosen

No label of any type comes from the competition match, and no tracking was corrected by hand. The
IDSSE labels use the same method as the competition labels: dead where the tracking marks the ball
as dead, and otherwise the team that the tracking names. Each test holds out one match at a time:
the settings are chosen on six matches, and the seventh is scored. A change is kept only if its mean
held-out spellF1 is better than that of the current version. The same test with the team-shape fit
held out gives 0.453.

| Version | Change | Chosen by | IDSSE held out | Public spellF1 / macroF1 |
|---|---|---|---|---|
| v0 | organisers' baseline, ported | n/a | n/a | 0.3256 / 0.6207 |
| v1 | Viterbi segmentation | n/a | n/a | 0.4125 / n/a |
| v1.5 | pace weight calibrated | IDSSE | 0.246 | 0.4000 / 0.7208 |
| v1.6 | touch evidence | IDSSE | 0.299 | 0.4513 / 0.7198 |
| v1.7 | separate prices for team and dead-ball changes | IDSSE | 0.325 | 0.4800 / 0.7208 |
| v1.8 | a touch counts for 2 s | IDSSE | 0.345 | 0.4353 / 0.7210 |
| v1.9 | team shape | IDSSE | 0.427 | 0.5375 / 0.8155 |
| v1.10 | goalkeeper track fix | IDSSE | 0.427 | 0.5333 / 0.8248 |
| v1.11 | team shape from 16 features | IDSSE | 0.453 | 0.5677 / 0.8256 |
| v1.12 (= v2.0) | back-line height over 30 s added | public (IDSSE tie) | 0.453 | 0.5677 / 0.8480 |
| v2.2 | set-piece shapes; control and prices tuned again | IDSSE and public | 0.487 | 0.6424 / 0.8264 |
| v2.3 | future reach; stoppage edges moved | IDSSE and public | 0.504 | 0.6750 / 0.8555 |
| v2.4 | touch radius and speed change tuned again | IDSSE | 0.512 | 0.6750 / 0.8579 |
| v2.5 | team shape from 4 tracked players, not 9 | IDSSE | 0.512 | 0.7226 / 0.8636 |
| **v2.6** | restart wait after a goal line | IDSSE | **0.512** | **0.7467 / 0.9128** |

The public window has 17 possessions, so one possession changes spellF1 by about 0.06. Up to v1.11,
the window was scored once per version, after the settings were fixed. Three choices used the public
window. v1.12 had the same IDSSE score as v1.11 and a better public macroF1. In v2.2 and v2.3, a
change had to increase public spellF1 without a decrease in the noisy IDSSE score (0.421, then
0.452, then 0.469). The public check chose the set-piece weight and the future-reach weight. (IDSSE
preferred a set-piece weight of 0.5, which scores 0.50 to 0.51 there.) This is tuning on the public
window. v2.3 also increased the stricter test (0.518 to 0.545) and Metrica.

From v2.4, noisy IDSSE chose all values, and the public check could only reject a change. From v2.3
to v2.6, noisy IDSSE increased from 0.469 to 0.493, the stricter test from 0.545 to 0.563 and the
Metrica mean from 0.633 to 0.651. v2.6 alone gives a small gain on IDSSE (0.5618 to 0.5626; 6 of 7
matches better).

Twice, the labels were drawn on the tactical video of the public 5 minutes to examine errors. This
gave three ideas, each judged on IDSSE: the goalkeeper fix, a restart wait after each ball out
(lower score) and the restart wait after a goal line. The last idea came from one goal kick that had
the label `H` long before the kick. Thus, its public gain (0.7226 to 0.7467) is not independent
evidence. Referee whistles, also scored on the window, never increased spellF1.

## 5. Possession rules

- A change of possession needs a touch or clear control by the other team. Three frames of touch
  evidence are more than the price of 10, and so is about 1.6 s of clear control. Future reach and
  the team shape add to both. A change of team costs more than two dead-ball changes. Thus, the
  label goes through a short `D`, which keeps the two possessions separate.
- A loose or contested ball gives no ball evidence and no cost. Thus, possession stays with the last
  team, as the challenge rules say, unless the team shape points to the other team for long enough.
- The evidence from a touch, a deflection or an aerial duel without control is less than the price
  of a change and the change back. Thus, it does not split a possession.
- The ball is dead when it is out of play, when play is at walking pace for some seconds, or when
  the players stand as before a corner, throw-in or goal kick. After the ball crosses a goal line,
  it stays dead until the goal kick or the corner.
- Play starts again when the control or live-pace evidence is more than the dead-ball evidence. Then
  the predicted restart moves 0.75 s earlier.
- A goalkeeper track in the opponents' half belongs to the other goalkeeper (section 3).

## 6. Limitations

- Part of the method was tuned on the public window (v1.12, and the weights in v2.2 and v2.3). If
  IDSSE alone chooses the v2.2 settings, they score below v2.0 on the public window. Also, the
  stricter IDSSE test does not agree with the public check on the set-piece weight and the control
  radius.
- The model misses 5 of the 17 public possessions. In each, the evidence (mostly the team shape)
  points to the other team, and no touch by the true team is detected.
- Some dead balls get a live label: short stoppages without a slowdown, and slow restarts. For
  example, before a goal kick, the keeper seems to control the ball while players jog. The restart
  wait covers goal kicks and corners only if the ball is seen placed and then kicked. Its IDSSE gain
  is small (+0.0008).
- IDSSE is cleaner than the competition data. The noise model copies the player dropouts and the
  extra tracks, but it closes only part of the difference. Also, the IDSSE ball gaps are simulated,
  not real.
- The DFL possession flag changes about 10 frames after the evidence of this model. The labels are
  not shifted to match it.

Future work: a model that stays accurate on tracking as noisy as the competition data, a whistle
detector that separates the referee from the crowd, and a restart detector tested on labelled data.
Ball positions from the broadcast or from video ball detection did not help.

## 7. Reproduction

Python 3.13; `requirements.txt` pins all dependencies (`numpy`, `pandas` and their dependencies).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # macOS / Linux: .venv/bin/python
.venv/Scripts/python -m kaburajadulu.run --data <folder holding tactical/>
```

The last command writes `output/KaburAjaDulu.csv`, the submitted file, byte for byte, in about 10 s,
with no manual step. This was tested from a new clone of the public repository, in a new Python 3.13
environment (SHA-256 `e5c492ee…`). The calibration and fitting scripts for IDSSE and Metrica are
available to the organisers on request. Their results are the commented values in
`kaburajadulu/params.py`.

## 8. References

1. Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal
   and event data in elite soccer. *Scientific Data*, 12, 195.
   [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y)
2. Metrica Sports. More than a Hack 2027: competition data, rules and evaluation code (`spellF1`,
   `macroF1`).
   [github.com/metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
3. Metrica Sports. Sample data (Sample Games 1, 2 and 3).
   [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
4. Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine
   Learning Research*, 12, 2825-2830.
5. PySport. kloppy, version 3.18.0. [github.com/PySport/kloppy](https://github.com/PySport/kloppy)
6. SkillCorner. Open data.
   [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata)
7. Vidal-Codina, F., Evans, N., El Fakir, B. and Billingham, J. (2022). Automatic event detection
   in football using tracking data. *Sports Engineering*, 25, 18.
   [doi:10.1007/s12283-022-00381-6](https://doi.org/10.1007/s12283-022-00381-6)
8. Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum
   decoding algorithm. *IEEE Transactions on Information Theory*, 13(2), 260-269.
   [doi:10.1109/TIT.1967.1054010](https://doi.org/10.1109/TIT.1967.1054010)
