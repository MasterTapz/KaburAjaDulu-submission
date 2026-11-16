# Team KaburAjaDulu: technical document

More than a Hack 2027, Phase 1: frame-level ball possession. FC Barcelona (home, `H`) vs Athletic
Club (away, `A`), LaLiga 2024-25, matchday 2. Submitted version: v2.6.
Code: https://github.com/MasterTapz/KaburAjaDulu-submission

## 1. Approach and why

`spellF1` scores whole possessions. If one frame in the middle of a possession gets the wrong
team, the possession splits in two and we lose it. Our ball is also missing on 41% of the graded
frames, mostly in gaps longer than 10 s, and play often goes on during those gaps. A
frame-by-frame classifier tends to flicker there, so we label the match in segments.

Each frame gets a cost for `H`, `A` and `D` from the tracking (table below). A cost of 0 means
the frame has nothing against that label. The Viterbi algorithm (Viterbi, 1967) then finds the
cheapest labelling of each half, where a change of team costs 10 and a change into or out of a
dead ball costs 3.6. The label changes only when the evidence against it adds up to more than
that price. A touch or a deflection is too short to pay for a change and the change back, so it
does not split a possession. A frame with no evidence costs nothing and the label holds, which
is the challenge's own rule for a loose or contested ball.

Two fixed steps follow. On IDSSE, our evidence of a stoppage and of a restart both arrive late,
so each predicted stoppage starts 0.25 s later and ends 0.75 s earlier. Frames outside the graded
halves are `X`, and the second half is `D` from the start of grading to its kickoff, as the FAQ
says.

| Evidence | Fires when | Cost per frame | How the value was set |
|---|---|---|---|
| Clear control | ball within 1.0 m of the nearest player, every opponent at least 0.25 m further away | +0.25 other team, +0.5 `D` | tuned on IDSSE with our noise |
| Touch | a player within 1.0 m while the ball turns more than 20° or changes speed by more than 20 m/s | +4.0 other team, for 2 s | detector from Vidal-Codina et al. (2022); tuned on IDSSE, radius and speed change with our noise |
| Team shape | the players alone: spread, line heights, block movement (section 3) | +0.3 × \|logit\| to the less likely team | fitted and tuned on IDSSE |
| Future reach | a player within 3 m of where the ball will be 0.5 s or 1 s later, every opponent at least 1 m further away | +0.05 per step, other team | distances fixed in advance; weight from the public check |
| Ball out | tracked ball more than 1.0 m over a line | +1.0 `H` and `A` | fixed in advance |
| Set-piece shape | ball and players placed as before a corner, throw-in or goal kick (5 detectors, 93% precision on IDSSE) | +1.0 `H` and `A` | shapes from the Laws of the Game; weight from the public check |
| Restart wait | ball over a goal line, until it is kicked from the goal area or a corner flag (98% precision on IDSSE) | +1.0 `H` and `A` | thresholds fixed in advance; weight tuned on IDSSE with our noise |
| Slow pace | players' mean speed under 1.0 m/s (1 s median) | up to +1.0 `H` and `A` | threshold fixed in advance; weight tuned on IDSSE |
| Live pace | players' mean speed over 2.0 m/s | up to +1.0 `D` | threshold fixed in advance; weight tuned on IDSSE |
| Smart Tagging | an attacking phase tagged for one team | +0.1 other team | fixed in advance |

The organisers' baseline takes the nearest player and smooths with a majority vote. It labels
every frame without a ball as `D`, and its one window size has to trade flicker against late
changes. We also tried a learned version of our own evidence (v2, a logistic regression on all
the signals). It did not beat the hand-built model on held-out IDSSE matches, so we submit the
simpler model, where every weight has a stated meaning.

## 2. Assets used and dropped

| Asset | How we used it |
|---|---|
| Tactical ATD (players and ball) | the input to every tracking clue and to the goalkeeper fix |
| Smart Tagging | a weak input: attacking phases name the team on the ball (+0.1) |
| Tactical homography and video | review only. We drew our labels over the video to look at our mistakes. Not a model input |
| Tactical and broadcast audio | tested, not used. Our referee-whistle detector needs the whistle in both recordings at once. 98% of its whistles are followed by a dead ball within 10 s, but nothing in the audio shows when the stoppage ends, so as evidence it split possessions |
| Broadcast video, ATD and homography | not used by the model. We synchronised it to the tactical recording by matching the two ball tracks (one offset per half, exact to the frame), to test the whistle detector and close-ups and replays as dead-ball evidence. Neither raised public spellF1 |

We used three external datasets. None of their labels come from the competition match.

| Dataset | Source and licence | Labels | Size | Used for |
|---|---|---|---|---|
| IDSSE, 7 Bundesliga matches | DFL open data (Bassek et al., 2025); CC-BY 4.0; Hugging Face mirror `pysport/idsse-data` | DFL's per-frame ball status (alive or dead) and team in possession | 1,002,644 frames (about 11 h), 3,061 possessions | team-shape fit (17 coefficients); weights and prices; the set-piece check; the learned model we dropped |
| Metrica Sample Games 1, 2 and 3 | Metrica Sports, GitHub `metrica-sports/sample-data` | event data with every stoppage | 3 matches | the dead-ball signal check (Game 3); our first price calibration (v1.4, later replaced) |
| SkillCorner open data, 20 A-League matches | GitHub `SkillCorner/opendata`, MIT licence | team and player in possession per frame | 1.6 million frames | tested, not used: adding it to the team-shape fit lowered held-out IDSSE accuracy (0.856 against 0.863) |

Several signals did not survive held-out testing. A static ball and a ball lost while heading out
of play both failed as dead-ball signals on Metrica Sample Game 3. A fixed timing shift matched
DFL's labelling convention rather than the competition's. A restart wait after every ball out
lost on IDSSE in three versions, so v2.6 keeps only the goal-line case that ends with a detected
kick. Using the restarting team as evidence after that kick changed no label.

## 3. Missing and noisy tracking

Our tracking is automatic and nobody corrected it. Each fix below is the same on every match.

The ball is missing on 41% of graded frames: 72.5% of the dead frames and 11.9% of the live
ones. In a gap there is no control or touch evidence, so the label holds unless other evidence
pays for a change. Pace needs no ball. The team shape reads possession from the players: the
team on the ball spreads out, pushes its back line up, and the whole block moves toward the goal
it attacks. A logistic regression (Pedregosa et al., 2011) on 17 such features (each team's
spread, the height of each back and front line, players in the opponents' half, block velocity
and position, over 1, 5, 10 and 30 s) names the team on the ball on 86% of the live frames of a
held-out IDSSE match. Before we added it, the team was wrong on 34% of the gap frames, against
6% of the frames with a ball.

Players drop out in the gaps. In our ball gaps the median frame has 21 tracked players, and
one frame in ten has 13 or fewer. The team shape works with as few as 4 tracked players per team
(v2.5). Below that it stays silent, on 7.9% of our gap frames.

Tracks also get mixed up. One team has more than 11 tracks on about 12% of our frames. The tracker
sometimes hands one goalkeeper's track to the other, so the wrong team gets control near
goal. A goalkeeper never stands in the opponents' half in open play, so we give such a track to
the other goalkeeper, or drop it if that goalkeeper is already in place. This rule has no
parameter and changes 18,002 frames of our match.

The IDSSE tracking is complete, so we add our kind of noise to a copy of each match. We cut the
ball in random gaps (lognormal, median 10.7 s) until 41% of frames have none, remove players in
the gaps to match our counts, and copy players into the other team's tracks on 11.4% of frames.
A stricter test puts the gaps mostly in stoppages, as on our match, and lets the stoppage ball
roll and be carried to the restart spot (DFL freezes it). Control, the prices, the touch radius
and speed change, the team-shape minimum and the restart-wait weight were tuned on these noisy
copies.

## 4. How we chose the settings

No label of any kind comes from the competition match, and we corrected nothing by hand. IDSSE
labels are made the same way as the competition's: dead where the tracking marks the ball dead,
otherwise the team the tracking names. We leave one match out (choose on six, score the
seventh, repeat) and adopt a change only if the mean held-out spellF1 beats the current version.
The team-shape fit was also checked with the fit itself held out (0.453).

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
| v2.2 | set-piece shapes; control and prices re-tuned | IDSSE and public | 0.487 | 0.6424 / 0.8264 |
| v2.3 | future reach; stoppage edges moved | IDSSE and public | 0.504 | 0.6750 / 0.8555 |
| v2.4 | touch radius and speed change re-tuned | IDSSE | 0.512 | 0.6750 / 0.8579 |
| v2.5 | team shape from 4 tracked players, not 9 | IDSSE | 0.512 | 0.7226 / 0.8636 |
| **v2.6** | restart wait after a goal line | IDSSE | **0.512** | **0.7467 / 0.9128** |

The public window holds 17 possessions, so one possession moves spellF1 by about 0.06. Up to
v1.11 we scored it once per version, after the settings were fixed. Three choices did lean on it.
v1.12 tied v1.11 on IDSSE and won on public macroF1. In v2.2 and v2.3, a change had to keep
noisy IDSSE from dropping (it went from 0.421 to 0.452 to 0.469) and had to raise public spellF1.
The public check picked the set-piece weight (IDSSE preferred 0.5, which scores 0.50 to 0.51 in
public) and the future-reach weight. That is tuning on the public window. v2.3 also gained on the
stricter test (0.518 to 0.545) and on Metrica.

From v2.4 on, IDSSE with our noise chose every value, and the public check could only say no.
From v2.3 to v2.6, noisy IDSSE rose from 0.469 to 0.493, the stricter test from 0.545 to 0.563
and the Metrica mean from 0.633 to 0.651. v2.6 alone adds little on IDSSE (0.5618 to 0.5626, 6 of
7 matches up).

Twice we drew our labels over the public 5 minutes on the tactical video to look at our mistakes.
That gave three ideas, each judged on IDSSE: the goalkeeper fix, a restart wait after every ball
out (it lost) and the restart wait after a goal line. The last came from one goal kick that we
labelled `H` long before the kick, so its public gain (0.7226 to 0.7467) is not independent
evidence. Referee whistles, also scored on the window, never raised spellF1.

## 5. Possession rules

- A change of possession needs a touch or clear control by the other team. Three frames of touch
  evidence pay the price of 10, and so does about 1.6 s of clear control; future reach and the
  team shape add to both. A change of team costs more than two dead-ball changes, so it passes
  through a brief `D`, which keeps the two possessions apart.
- A loose or contested ball gives no ball evidence, so possession stays with the last team,
  unless the team shape points the other way long enough to pay for a change.
- A touch, deflection or aerial duel without control is too short to pay for a change and the
  change back.
- The ball is dead when it is out of play, when play has slowed to walking pace for several
  seconds, or when the players stand as before a corner, throw-in or goal kick. After the ball
  crosses a goal line, it stays dead until the goal kick or corner is taken.
- Play restarts when control or live pace outweighs the dead-ball evidence. The predicted restart
  then moves 0.75 s earlier.
- A goalkeeper track in the opponents' half belongs to the other goalkeeper (section 3).

## 6. Limitations

- Part of the method is tuned on the public window (v1.12, and the weights in v2.2 and v2.3).
  Chosen by IDSSE alone, the v2.2 settings score below v2.0 on the public window, and the
  stricter IDSSE test still disagrees with it on the set-piece weight and the control radius.
- We miss 5 of the 17 public possessions. In each, the evidence points to the other team, mostly
  through the team shape, and we detect no touch by the true team.
- Some dead balls come out live: short stoppages without a slowdown, and slow restarts, where a
  goal kick being set up looks like a keeper in control with players jogging. The restart wait
  covers goal kicks and corners only when our ball is seen placed and then kicked, and its IDSSE
  gain is small (+0.0008).
- IDSSE is cleaner than our data. Our noise copies the player dropouts and extra tracks, but it
  closes only part of the gap, and the IDSSE ball gaps are simulated, not real.
- DFL's possession flag switches about 10 frames after our evidence; we did not shift to match.

With more time we would build a model that holds up on tracking as noisy as ours, a whistle
detector that separates the referee from the crowd, and a restart detector tested on labelled
data. Filling ball gaps from the broadcast or from ball detection on the video did not help.

## 7. Reproduction

Python 3.13, with every dependency pinned in `requirements.txt` (`numpy`, `pandas` and pandas'
own dependencies).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # macOS / Linux: .venv/bin/python
.venv/Scripts/python -m kaburajadulu.run --data <folder holding tactical/>
```

This writes `output/KaburAjaDulu.csv`, the submitted file, byte for byte, in about 10 s, with no
manual step. We checked this from a fresh clone of the public repository in a fresh Python 3.13
environment (SHA-256 `e5c492ee…`). Our calibration and fitting scripts for IDSSE and Metrica are
in our development repository, which we share with the organisers on request; their results are
the commented values in `kaburajadulu/params.py`.

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
