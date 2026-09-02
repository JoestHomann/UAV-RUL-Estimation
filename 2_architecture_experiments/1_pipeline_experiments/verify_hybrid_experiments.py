"""Verify PE_10 settings resolution and Run 8's static contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import read_experiment_config  # noqa: E402
from experiment_paths import repository_path  # noqa: E402
from run_experiments import _load_interface, _phase2_settings  # noqa: E402


def main() -> None:
    pe10_path = SCRIPT_DIR / "experiments" / "PE_10" / "settings.toml"
    config = read_experiment_config(pe10_path)
    builder = (
        SCRIPT_DIR.parent
        / "2_model_architecture_study"
        / "1_architecture_study_settings"
        / "build_architecture_study_settings.py"
    )
    for name in ("PE10_recent", "PE10_multiresolution"):
        experiment = config["experiments"][name]
        interface_path, interface = _load_interface(experiment)
        resolved = _phase2_settings(config, experiment, interface, interface_path)
        architecture = resolved["architectures"]["hybrid_cnn"]
        assert architecture["representation"] == "heterogeneous"
        assert architecture["feature_sets"] == ["screened_drift_pruned"]
        root = REPOSITORY_ROOT / ".tmp" / "hybrid_verification" / name
        root.mkdir(parents=True, exist_ok=True)
        settings_path = root / "settings.json"
        settings_path.write_text(json.dumps(resolved), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--settings",
                str(settings_path),
                "--output-dir",
                str(root / "artifacts"),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )

    run8_path = (
        SCRIPT_DIR.parent
        / "2_model_architecture_study"
        / "hybrid_run_8_settings.toml"
    )
    with run8_path.open("rb") as stream:
        run8 = tomllib.load(stream)
    assert set(run8["families"]) == {"xgboost", "hybrid_cnn", "hybrid_gru"}
    pe10_source = repository_path(
        REPOSITORY_ROOT,
        run8["sources"]["pe10_settings"],
    )
    assert pe10_source == pe10_path.resolve()
    assert repository_path(
        REPOSITORY_ROOT,
        run8["sources"]["tree_oof_predictions"],
    ).is_file()
    print("Hybrid experiment configuration verification passed")


if __name__ == "__main__":
    main()
