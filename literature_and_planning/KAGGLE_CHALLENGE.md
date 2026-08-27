# UAV Remaining Useful Life — Challenge Brief

- **Competition:** [UAV Remaining Useful Life](https://www.kaggle.com/competitions/uav-remaining-useful-life)
- **Host:** MLMech Course
- **Type:** Private community prediction competition
- **Status checked:** 5 August 2026 (Europe/Berlin)
- **Start:** 1 July 2026, 15:15:24 CEST
- **Close:** 13 October 2026, 00:00:00 CEST
- **Award:** Kudos only; no Kaggle points or medals

## Challenge summary

A company operates autonomous UAVs for industrial inspection and monitoring. Each UAV records anonymized telemetry during successive flight cycles. Components degrade over time and eventually fail.

The task is to build a machine-learning regression model that estimates the **Remaining Useful Life (RUL)** of previously unseen UAVs from their telemetry histories. RUL is the number of flight cycles remaining before failure.

For the test set, the required prediction is the RUL at the **final available observation (flight cycle) of each UAV**. The public leaderboard evaluates one prediction per test UAV.

## Data

Each row represents one UAV during one flight cycle.

| Field | Meaning |
|---|---|
| `uav_id` | Unique UAV identifier; each UAV appears in multiple consecutive cycles |
| `flight_cycle` | Sequential flight number; larger values are later observations |
| `telemetry_01`–`telemetry_28` | Anonymized telemetry channels; channels may differ in usefulness and scale |
| `target` | Remaining Useful Life in flight cycles; present only in the training data |

The host states that no missing values were intentionally introduced, but recommends inspecting the data carefully.

### Described files

- `train.csv` — telemetry observations plus the RUL target.
- `test.csv` — telemetry for unseen UAVs with targets removed.
- `sample_submission.csv` — example of the required submission structure.
- `starter_notebook.ipynb` — described baseline workflow covering loading, exploratory analysis, visualization, regression models, leakage-safe validation, and submission generation.

Kaggle's data explorer currently lists **two CSV files totaling 22.36 MB** (`train.csv` and `test.csv`); `test.csv` is shown as 8.94 MB. The competition description mentions the sample submission and starter notebook in addition to those two CSVs.

## Evaluation

Submissions are scored using the **R² (coefficient of determination) score** between predicted RUL values and hidden ground truth. Higher is better; the ideal score is 1. Negative scores are possible when predictions are worse than a constant-mean baseline.

The public leaderboard uses approximately **30% of the test data**. Final standings use the other **70%**, so leaderboard positions may change substantially.

## Submission format

The CSV must contain exactly two columns and no index column:

```csv
id,RUL
UAV_0101,91.4
UAV_0102,37.2
UAV_0103,108.8
```

Requirements:

- Include exactly one row for every UAV in the test set.
- Use each test `uav_id` value unchanged in Kaggle's required `id` column.
- Put the predicted remaining flight cycles in `RUL`.
- Do not include a pandas/index column.

The live submission validator explicitly reported `ID column id not found in
submission` for a file headed `uav_id,RUL` on 27 August 2026. That validator
message supersedes the earlier competition-page wording recorded below.

## Competition-specific rules

- **Team size:** minimum 2, maximum 4 members.
- **Team mergers:** not allowed after registration.
- **Submission limit:** at most 5 submissions per day.
- **Final selections:** up to 2 submissions may be selected for final judging.
- **Competition data:** may be used only for participation in this competition and discussion in its Kaggle forum. It may not be redistributed, published, shared with third parties, or used outside the course without explicit host permission.
- **External data/models/tools:** allowed when publicly and equally accessible to participants, or otherwise reasonably accessible at minimal cost under the rules. Automated machine-learning tools are permitted, subject to the same accessibility and rules constraints.
- Participation or submission from multiple Kaggle accounts is prohibited.

The full official rules on Kaggle control if this summary differs from the live competition page.

## Modeling considerations highlighted by the host

- Determine which telemetry features are informative.
- Check whether features operate on comparable scales.
- Look for suspicious observations or entire UAV histories.
- Identify features correlated with degradation.
- Compare model families suited to regression on sequential telemetry.
- Prevent UAV leakage during validation by keeping observations from the same UAV in a single split.

Because the test prediction is made at the final observed cycle of each UAV, useful approaches may summarize each UAV's recent level, trend, volatility, and lifetime history. This is a modeling inference, not an official rule.

## Current competition snapshot

At the time checked, the overview showed:

- 26 entrants
- 11 participants
- 6 teams
- 190 submissions

The public leaderboard showed:

| Rank | Team | Public R² | Entries |
|---:|---|---:|---:|
| 1 | Gruppe Baldauf | 0.91155 | 99 |
| 2 | OpenIA-powered Team | 0.86739 | 38 |
| 3 | UAV Predict | 0.85801 | 26 |
| 4 | Aras Cevadi | 0.66359 | 19 |
| 5 | Paulo_power_by_copilot | -0.12661 | 7 |
| Benchmark | `sample_submission.csv` | -0.46701 | 1 |
| 6 | BennitK | -0.79913 | 1 |

These counts and scores are a point-in-time snapshot and will change as the competition continues.

## Available community resources

- The overview and data description say a starter notebook is provided.
- The Code tab showed **No notebooks found** when checked.
- The Models tab showed **No models found**.
- The Discussion tab showed **No discussions found**.

## Practical checklist

1. Inspect UAV counts, cycle-length distributions, target range, constants, duplicates, missing/non-finite values, and suspicious UAV histories.
2. Use group-aware validation by `uav_id`; never randomly split rows across the same UAV.
3. Establish simple baselines, including lifecycle/cycle-based and tree-based regression models.
4. Engineer per-UAV temporal features such as last value, rolling summaries, slopes, deltas, extrema, and cycle-normalized statistics.
5. Evaluate with out-of-fold R² using UAV-level splits.
6. Fit the final model, select the final cycle per test UAV, and emit exactly one `id,RUL` row per UAV, mapping each internal `uav_id` to `id`.
7. Verify column names, row count, UAV uniqueness, and absence of an index column before submitting.

---

Source: live Kaggle competition pages (Overview, Data, Code, Models, Discussion, Leaderboard, and Rules), accessed while signed in on 5 August 2026.
