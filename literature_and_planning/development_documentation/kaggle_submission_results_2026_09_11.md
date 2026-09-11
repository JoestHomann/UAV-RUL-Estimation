# Kaggle submission results reported on 11 September 2026

Scores below were supplied by the user in Kaggle completion screenshots in the
project conversation. File attribution follows the submission links supplied
immediately before each upload and the filenames shown in the screenshots.
Local CSV hashes were checked on 11 September. No Kaggle API verification or
independent access to the uploaded bytes was performed.

| Candidate | Kaggle R² | Change versus reported Run 7 |
|---|---:|---:|
| PE_31 unchanged simpler-script reproduction | **0.88177** | +0.00525 |
| Run 7, earlier user-reported reference | 0.87652 | 0 |
| PE_24 regime-aware TabPFN blend | 0.87545 | -0.00107 |
| PE_25 restricted TabPFN blend | 0.86926 | -0.00726 |

The earlier simpler-script score of 0.87877 is a separate historical result;
it is not the score of the regenerated CSV listed below. Run 7's score is
carried forward from conversation history, not newly shown in these screenshots.

## Submission identities

Paths are relative to the repository root.

- PE_31: `2_architecture_experiments/1_pipeline_experiments/experiments/PE_31/runs/run_1/reproduction/submission.csv`
  - SHA-256: `1fe5ff8fb9e1b44b6599aab74fe9b80e49ea334096911a2b4996c269b768bf7a`
- PE_24: `2_architecture_experiments/1_pipeline_experiments/experiments/PE_24/runs/run_1/exploratory_submission/submission_PE24_regime_tabpfn.csv`
  - SHA-256: `56e0c74ff3dddb42e5f4b78b673c83a7807d5927f339f2ae85812640802b5cd6`
- PE_25: `2_architecture_experiments/1_pipeline_experiments/experiments/PE_25/runs/run_1/exploratory_submission_v2/submission_PE25_restricted_tabpfn.csv`
  - SHA-256: `f71e8121de4bcf08ccec954c3ae2d30a23580c33f53ee4b7a198ae47a1d3efb7`

## Interpretation and next decision

PE_24's 1.896% and PE_25's 1.904% local mean-fold RMSE improvements did not
transfer to these leaderboard submissions. Both had already failed their local
promotion criteria, with bootstrap intervals including deterioration. These
uploads therefore do not contradict the original no-promotion decisions.

The scores do not establish a cause. Possible explanations to investigate
include endpoint-distribution mismatch, adaptation to repeatedly inspected
development UAVs, and changes in component/gate behavior between OOF fits and
full-training inference. The test labels are unknown; errors cannot be assigned
to individual test UAVs from the aggregate scores.

Retain the simpler reproduction as the strongest submitted candidate among
these reported results, and retain Run 7 as the established local control.
Pause further TabPFN weight/gate variants. Before another broad model search,
compare the two main pipelines under matched grouped splits, raw-label scoring,
and fully audited preprocessing, calibration, and final-refit procedures.
Audit validation endpoint coverage against observable test history lengths and
sensor summaries. Any revised protocol should be defined transparently and
checked alongside the historical suite, rather than tuned until it reproduces
the leaderboard ordering. New splits on these same UAVs are robustness checks,
not untouched evaluation data.

This external-result note does not modify immutable preparation registrations,
original validation verdicts, or saved production models. Preparation manifests'
`submitted_to_kaggle: false` fields describe their state when the files were
generated; the subsequent user uploads are recorded here.
