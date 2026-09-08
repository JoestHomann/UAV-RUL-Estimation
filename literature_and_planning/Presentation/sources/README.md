# Sources for `UAV_RUL_Project_20min.pptx`

Everything in the deck is rebuilt from this directory.

## Figure policy

Charts in the deck come from the repository's own generated figures wherever
one exists. `figure_manifest.csv` records, for every figure: its provenance,
the repository path it came from, and the slide it is placed on.

| Provenance | Count | Meaning |
| --- | ---: | --- |
| `repository figure, unchanged` | 10 | Byte-identical copy |
| `repository figure, cropped …` | 3 | Same image, cropped to one panel; nothing re-plotted |
| `repository script re-run on a channel subset` | 1 | Same script, same colours, fewer channels |
| `generated for the talk` | 3 | No repository figure exists for that evidence |

Cropping was used only where a generated figure is a tall multi-panel image
that renders at 4–6 pt when scaled onto the template's 10 × 5.625 in canvas.
The three generated figures use the repository's own Phase 0 palette and axis
styling from `0_data_analysis/broad_data_review/plotting_common.py`.

## Files

| File | Purpose |
| --- | --- |
| `build_figures.py` | Copies, crops and generates everything in `figures/`, and writes `figure_manifest.csv` |
| `build_deck.py` | Builds the .pptx on a copy of the supplied template |
| `deck_content.py` | Slide titles, subtitles, source labels and the full narration |
| `build_ledger.py` | Validates `evidence_claims_source.csv` and writes `../evidence_ledger.csv` |
| `build_notes.py` | Writes `../speaker_notes.md` and prints the timing check |
| `evidence_claims_source.csv` | Editable source of truth for the 122 evidence claims |
| `chart_data/` | Measured values behind the generated figures |
| `figures/` | The figures placed in the deck |
| `figures_repo/` | Output of the re-run repository script |

## Rebuild

With `python-pptx`, `matplotlib`, `pandas`, `pillow` and `scipy` installed:

```bash
# 1. re-run the one repository script, restricted to the eight channels shown
python 0_data_analysis/core_data_analysis/temporal_rul_analysis.py \
    --channels telemetry_21 telemetry_19 telemetry_13 telemetry_25 \
               telemetry_16 telemetry_18 telemetry_01 telemetry_20 \
    --output-dir <this dir>/figures_repo/temporal_rul_subset --dpi 160

# 2. assemble the figure set
python build_figures.py --repo-root C:/Users/joest/UAV-RUL-Estimation

# 3. ledger and notes
python build_ledger.py
python build_notes.py

# 4. the deck
python build_deck.py \
    --template <path to UAVRULEstimation_presentation.pptx> \
    --out ../UAV_RUL_Project_20min.pptx

# 5. optional PDF
soffice --headless --convert-to pdf ../UAV_RUL_Project_20min.pptx
```

Step 1 needs one edit to the repository script, which is otherwise untouched:
its `plot_summary` uses a fixed `figsize=(19, 10)` canvas sized for all 28
channels. The presentation copy scales that canvas with the number of plotted
channels when fewer than 21 are requested, so the same plot stays legible at
projection size. The repository's own copy of the script was not modified.

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

## Schematic colours

The two native PowerPoint schematics use the template's "UNI COLOUR" theme with
fixed roles: `#00519E` model members and outputs, `#00BEFF` the selected step,
`#FFD500` the element under discussion, `#9F9998` intermediate state. The
repository figures keep their original colours and are not restyled.
