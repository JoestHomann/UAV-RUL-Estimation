# Sources for `UAV_RUL_Project_20min.pptx`

Everything in the deck is rebuilt from this directory.

## Figure policy

Charts come from the repository's own plotting code wherever that code exists.
`figure_manifest.csv` records, for every figure: its provenance, the repository
path it came from, the slide it is placed on, and — for a re-run — the result of
comparing the re-run's output table with the repository's own table.

| Provenance | Count | Meaning |
| --- | ---: | --- |
| `repository figure, unchanged` | 13 | Byte-identical copy |
| `repository script re-run at slide size` | 6 | Unmodified script, presentation canvas |
| `repository figure, cropped …` | 2 | Same image, cropped to one panel |
| `generated for the talk` | 4 | No repository figure or script covers it |

The Phase 0 figures are drawn on 11 × 9 in to 19 × 10 in canvases. Scaled onto
the template's 10 × 5.625 in canvas their axis labels reach the slide at three
to four points. `run_repo_figure.py` therefore imports each repository script
**unmodified** and overrides only four presentation quantities before calling
its `main()`:

1. the figure canvas size, so the figure sits on the slide at roughly 1:1;
2. the tick-label size used by the shared `style_axis` helper, which hard-codes
   `labelsize=8`;
3. the matplotlib base font size;
4. the figure-title size, which a few scripts hard-code at 14 pt.

No repository file is copied with edits, and no repository file is modified.
`build_figures.py` verifies this by re-reading the CSV each script wrote and
comparing it with the repository's own CSV; the largest numeric difference is
printed and stored in the manifest.

The four figures generated here use the repository's Phase 0 palette and axis
styling from `0_data_analysis/broad_data_review/plotting_common.py`.

## Files

| File | Purpose |
| --- | --- |
| `run_repo_figure.py` | Runs an unmodified repository plotting script at slide size |
| `build_figures.py` | Copies, crops, collects and generates everything in `figures/`, and writes `figure_manifest.csv` |
| `build_deck.py` | Builds the .pptx on a copy of the supplied template |
| `deck_content.py` | Slide titles, subtitles, panel text, source labels and the full narration |
| `build_ledger.py` | Validates `evidence_claims_source.csv` and writes `../evidence_ledger.csv` |
| `build_notes.py` | Writes `../speaker_notes.md` and prints the timing check |
| `evidence_claims_source.csv` | Editable source of truth for the 192 evidence claims |
| `chart_data/` | Measured values behind the generated figures |
| `figures/` | The figures placed in the deck |
| `figures_repo/` | Output of the re-run repository scripts, including their CSVs |

## Rebuild

With `python-pptx`, `matplotlib`, `pandas`, `pillow` and `scipy` installed, and
`<repo>` pointing at the checkout:

```bash
# 1. re-run the six repository Phase 0 scripts at slide size
R=<repo>/0_data_analysis
python run_repo_figure.py --script $R/broad_data_review/plot_constant_features.py \
    --figure-size 8.9 4.0 --label-size 6.0 --font-size 8 --suptitle-size 9.5 \
    -- --train-csv <repo>/data/train.csv --output-dir figures_repo/constant_features --dpi 200

python run_repo_figure.py --script $R/broad_data_review/plot_within_between_variance.py \
    --figure-size 5.45 4.10 --label-size 6.0 --font-size 8 \
    -- --train-csv <repo>/data/train.csv --output-dir figures_repo/within_between --dpi 200

python run_repo_figure.py --script $R/core_data_analysis/temporal_rul_analysis.py \
    --figure-size 8.45 3.55 --label-size 6.5 --font-size 8 \
    -- --train-csv <repo>/data/train.csv --output-dir figures_repo/temporal_rul_subset --dpi 200 \
       --channels telemetry_07 telemetry_13 telemetry_15 telemetry_16 telemetry_19 \
                  telemetry_21 telemetry_22 telemetry_23 telemetry_25 telemetry_28 \
                  telemetry_18 telemetry_26

python run_repo_figure.py --script $R/core_data_analysis/train_test_drift.py \
    --figure-size 9.0 3.95 --label-size 6.0 --font-size 8 --suptitle-size 9.5 \
    -- --train-csv <repo>/data/train.csv --test-csv <repo>/data/test.csv \
       --output-dir figures_repo/train_test_drift --dpi 200

python run_repo_figure.py --script $R/core_data_analysis/channel_classification.py \
    --figure-size 5.45 4.10 --label-size 6.0 --font-size 8 \
    -- --train-csv <repo>/data/train.csv --input-root $R/core_data_analysis/figures \
       --output-dir figures_repo/channel_classification --dpi 200

python run_repo_figure.py --script $R/broad_data_review/plot_history_length_distributions.py \
    --figure-size 9.0 3.05 --label-size 7.0 --font-size 8.5 \
    -- --train-csv <repo>/data/train.csv --test-csv <repo>/data/test.csv \
       --output-dir figures_repo/history_lengths --dpi 200

# 2. assemble the figure set and check each re-run against the repository table
python build_figures.py --repo-root <repo>

# 3. ledger and notes
python build_ledger.py
python build_notes.py

# 4. the deck
python build_deck.py --template <path to UAVRULEstimation_presentation.pptx> \
    --out ../UAV_RUL_Project_20min.pptx

# 5. optional PDF
soffice --headless --convert-to pdf ../UAV_RUL_Project_20min.pptx
```

`channel_classification.py` reads four upstream Phase 0 tables, so
`--input-root` must point at a `figures/` directory containing
`temporal_rul/`, `feature_redundancy/`, `train_test_drift/` and `anomalies/`.

## Where `chart_data/` came from

| File | Copied or transcribed from |
| --- | --- |
| `s02_public_score_chain.csv` | `.../r2_research_2026_09_07/kaggle_scores.csv` plus the decision attributions in `pipeline_experiments.md` |
| `s05_cap_scenario_matrix.csv` | `pipeline_experiments.md`, Development Results table |
| `s06_signal_family_ablation.csv` | same file, Signal-Family Ablation table |
| `s07_architecture_studies.csv` | architecture study `run_5`, `run_7`, `run_8` comparison tables |
| `s08_ensemble_methods.csv` | `.../experiments/PE_11/runs/run_1/reporting/summary.csv` and `winner_manifest.json` |
| `s09_pe4_calibration_summary.csv` | verbatim copy of `.../PE_4/runs/run_1/reporting/calibration_summary.csv` |
| `s10_screen_versus_confirmation.csv` | `PE_15`, `PE_20`, `PE_18`, `PE_21`, `PE_24` run 1 summary and promotion tables |
| `s11_error_bands.csv`, `s11_per_uav_error.csv` | verbatim copies from `.../r2_research_2026_09_07/` |
| `s12_kaggle_scores.csv` | verbatim copy of `kaggle_scores.csv` |
| `s12_development_and_public.csv` | Phase 3 `run_6`/`run_7` `report_summary.json`, `diagnostic_summary.json`, `kaggle_scores.csv` |

The Phase 0 and Phase 1 slides read their numbers directly from repository
artifacts rather than from `chart_data/`; the paths are in
`evidence_claims_source.csv`.

## Schematic colours

The native PowerPoint schematics use the template's "UNI COLOUR" theme with
fixed roles: `#00519E` model members, folds and outputs, `#00BEFF` the selected
step, `#FFD500` the held-out fold or the element under discussion, `#9F9998`
intermediate state. The repository figures keep their original colours and are
not restyled.
