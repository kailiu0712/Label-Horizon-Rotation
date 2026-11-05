# When does a shorter training label help?

The code behind the Superday deck: an extension of Song, Liu & Chen, *The Label Horizon Paradox* (ICML 2026).

**Question.** Fix the return you are evaluated on, and change only the label you train on. When does a shorter label beat the aligned one?

**Answer.** Three drivers, in a chain: features set the clocks, clocks set the rotation, and rotation together with sample size sets the best label.

$$\mathbb{E}\,\mathrm{IC}(\delta;\Delta)\approx\rho^*_\Delta\times\mathrm{corr}(s_\delta,s_\Delta)\times\sqrt{\frac{R^2_\delta}{R^2_\delta+d/n}}$$

## Layout

```
label_horizon/
├── config.py               every path, setting and the two scenarios
├── __main__.py             runs the stages in order
│
├── data/                   raw bars → one panel
│   ├── features.py           the 20 descriptors
│   ├── extract.py            minute files → features and prices per stock-day
│   └── panel.py              labels, per-date sums, and the Panel class (ridge + IC)
│
├── market/                 real-data results; each takes a Panel
│   ├── walk_forward.py       IC of every label on the fixed target; label rules
│   ├── feature_groups.py     clock share of each feature group; FAST and SLOW
│   ├── rotation.py           direction correlation, cross-fitted
│   ├── sample_size.py        the edge against training dates; crossover n*
│   ├── pieces.py             feature correlations with each piece of the target
│   └── inference.py          date-block bootstrap
│
├── synthetic/              known-truth simulation
│   ├── world.py              simulated markets and their exact moments
│   ├── theory.py             the formula
│   ├── evaluate.py           one simulated market through the pipeline
│   └── experiments.py        the three studies
│
├── figures.py              every figure in the deck
├── report.py               every number on the slides
└── tests/test_slides.py    those numbers, pinned
```

## From slide to code

| Slide | Shows | Code | Output |
|---|---|---|---|
| 6–7 | Extended formula, crossover n\* | `synthetic/theory.py`, `ReliabilityModel` in `market/sample_size.py` | |
| 8 | Two signal clocks | `synthetic/world.py`: `ar1_profile`, `simulate` | `B11_clocks.png` |
| 9 | Clocks rotate the direction; sample size picks the winner | `synthetic/experiments.py`: `run_mixtures`, `run_sample_size` | `B13_synthetic_lab.png` |
| 10 | The paper's Figure 2 on real data | `market/walk_forward.py`: `walk_forward`, `paired_band` | `B1_paper_rerun.png` |
| 11 | Feature clocks; group-alone ICs | `market/feature_groups.py`: `clock_share`, `group_daily_ic` | `B12_feature_clocks.png` |
| 12 | Rotation measured | `market/rotation.py`: `direction_corr` | `B3_rotation.png` |
| 13 | The edge shrinks with data | `market/sample_size.py`: `subsample_gaps`, `reliability_model` | `B2_noise_dose.png` |
| 14 | Why the clocks differ (hypothesis) | `market/pieces.py` | `piece_correlations.csv` |
| A1 | Data and protocol | `data/extract.py`, `data/panel.py`, `config.py` | |
| A2 | Feature construction | `data/features.py`; FAST and SLOW in `config.py` | |
| A3 | Formula with ridge penalty | `synthetic/theory.py`: `expected_ic` | |
| A4 | Penalty and label choice | `market/walk_forward.py`: `label_rules` | `B7_lambda.png` |
| A5 | One clock, different shapes | `synthetic/experiments.py`: `run_atlas` | `B6_synthetic.png` |

Slides 2–5 present the paper and its limits; they use no code of ours.

## A walkthrough in five stops

1. **`data/features.py`**, `patch_descriptors`: what the model sees.
2. **`data/panel.py`**, top docstring and `Panel`: the design idea. Features and labels are standardised within each date, so any ridge fit needs only per-date sums. A walk-forward refit becomes one sum and one linear solve, and `Panel.fit` and `Panel.daily_ic` are all the rest of the code needs.
3. **`market/walk_forward.py`**, `walk_forward`: the main experiment, about ten lines.
4. **`market/rotation.py`**, `direction_corr`: the contribution. Fit on odd and even dates, cross the halves, correct for noise.
5. **`synthetic/world.py` and `theory.py`**: simulate a market with known truth and check the formula against it.

## Running it

From `superday/`, with `pip install -r label_horizon/requirements.txt`:

```
python -m label_horizon                     # everything
python -m label_horizon market figures      # or any stages
pytest label_horizon/tests                  # check the slide numbers
```

| Stage | Does | Time |
|---|---|---|
| `extract` | 600 raw files → features and prices | 30 min |
| `panels` | labels and per-date sums for both scenarios | 1 min |
| `market` | all real-data results | 6 min |
| `synthetic` | about 3,500 simulated markets | 40 min (estimate) |
| `figures`, `report` | figures and slide numbers | 1 min |

Inputs, read and never written: minute bars in `data/rq_csi1000_bar1m_2019-2026/`, and audited identities, index membership and end-of-minute prices in `data/minute_scenarios/`. Outputs go to `label_horizon/outputs/` and `data/label_horizon/`.

## Choices worth knowing

- **No look-ahead.** Training blocks are keyed by the quarter a label matures in, test dates by the quarter of the signal. `rotation(q)` uses only data before quarter q.
- **Holdout.** 2026 H1 was used for no choice and is scored once. `development()` excludes it from every curve; the rules table reports it separately.
- **S2 and S3 share a panel.** Same decision time, features and labels; only the target differs.
- **FAST and SLOW are frozen in `config.py`.** They were fixed on 2023 data by the rule in `response_classes`, in an earlier study with its own cohort and scaling. Rerunning the rule on this panel gives 10 FAST and 6 SLOW, recovering all 9 frozen FAST features and 5 of the 6 frozen SLOW ones. The frozen lists are what every result uses.
- **The validation block in `simulate`** is drawn and discarded so the published seeds reproduce.

## How this was checked against the original research code

| Step | Check | Result |
|---|---|---|
| Raw files → extract | 1.59M rows of features and prices, full rerun | identical |
| Extract → panels | every per-date sum, both scenarios | identical, largest difference 2e-13 |
| Market results | every daily IC, rotation, sample-size point, rule, group IC, clock share and interval, full rerun | largest difference 1e-13 |
| Simulation, formula | all 255 + 720 rows | largest difference 1e-16 |
| Simulation, sample size | full rerun, 30 seeds | largest difference 1e-11 |
| Simulation, atlas and mixtures | 10 single trials against the saved per-seed results, plus a 2-seed run of each study | largest difference 2e-9 |
| Figures | pixel comparison with the deck | 6 of 8 identical; the other 2 below |
| Slide numbers | `tests/test_slides.py` | pass |

`outputs/tables/atlas.csv`, `mixture_*.csv` and `one_clock_sample_size.csv` hold the archived full-run results under this package's column names; they were not regenerated here. `python -m label_horizon synthetic` overwrites them.

Two figures intentionally differ from the original code. With the old values put back, both match the deck pixel for pixel:

- **`B13_synthetic_lab.png`.** The dotted "unlimited data" line is now the Δ = 90 value, −0.0115. The original read −0.0124, the Δ = 30 value, from an unsorted row.
- **`B2_noise_dose.png`.** The n\* marker is now computed, at 256 dates. The original hard-coded 261, worked out by hand from rounded inputs. "n\* ≈ 260" still holds.

Not in this package: the bi-level replication (no results on the slides), the theory sheet's first-piece decomposition, and the "+35–50%" index rally on slide 5. The rally's original script was not kept; raw closes give about +34% to +37%, depending on the exact jump filter.
