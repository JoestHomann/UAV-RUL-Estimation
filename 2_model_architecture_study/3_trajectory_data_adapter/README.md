# Step 3b: Trajectory data adapter

This adapter exposes complete causal telemetry prefixes as variable-length
trajectory queries. For fold-based training it also supplies a reference
library containing complete run-to-failure histories from the active training
UAVs only, including cycle-wise remaining life.

It reuses the sequence adapter's byte-verified copies of the raw histories,
endpoint tables, and fold assignments. It does not duplicate those datasets.
Channel scaling uses the same median/IQR implementation and is fitted only on
the current training UAV partition. Validation UAVs never enter the scaler or
the reference library.

The runtime API in `trajectory_data_adapter.py` provides:

- `load_training`, `load_development`, `load_locked`, and `load_test`;
- `get_inner_selection_split` for Phase 2 tuning;
- `get_final_search_split` for Phase 3 tuning;
- `get_locked_outer_evaluation_split` for locked evaluation;
- `TrajectoryDataset`, `TrajectoryReferenceLibrary`, and
  `TrajectoryChannelScaler` as reusable model-facing objects.

Run the builder after the sequence adapter:

```powershell
py 2_model_architecture_study\3_trajectory_data_adapter\build_trajectory_data_adapter.py
```

The builder writes `artifacts/trajectory_dataset_manifest.json` and
`artifacts/trajectory_verification.json`. The verification exercises a real
inner split, checks endpoint cutoffs, checks UAV disjointness, and confirms
that both references and scaling use training UAVs only.

The two new Run 4 architectures still consume fixed sequence windows. This
trajectory interface is deliberately model-independent and ready for a later
similarity-, retrieval-, or full-lifecycle model without changing the split
contract again.
