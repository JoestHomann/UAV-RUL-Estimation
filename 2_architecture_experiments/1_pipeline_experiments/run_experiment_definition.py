"""Execute the ordered script plan declared by one PE_X experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import ExperimentConfigError, read_experiment_config
from experiment_paths import pipeline_owner, repository_path, run_directory


ARCHITECTURE_EXPERIMENTS_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = ARCHITECTURE_EXPERIMENTS_ROOT.parent


class DefinitionRunnerError(ValueError):
    """Explain an invalid run execution plan."""


def _repo_path(value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DefinitionRunnerError(f"{label} must be a non-empty path")
    path = repository_path(REPOSITORY_ROOT, value)
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise DefinitionRunnerError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise DefinitionRunnerError(f"{label} does not exist: {path}")
    return path


def _execution_plan(
    definition_path: Path,
    run_name: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = read_experiment_config(definition_path)
    definitions = config.get("run_definitions")
    run = definitions.get(run_name) if isinstance(definitions, dict) else None
    if not isinstance(run, dict):
        raise DefinitionRunnerError(
            f"{definition_path} does not define run_definitions.{run_name}"
        )
    steps = run.get("steps")
    if not isinstance(steps, list) or not steps:
        raise DefinitionRunnerError(f"{run_name} needs at least one execution step")
    names: set[str] = set()
    chains = config.get("script_chains", {})
    if not isinstance(chains, dict):
        raise DefinitionRunnerError("script_chains must be a TOML table")
    validated: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise DefinitionRunnerError(f"{run_name} step {index} must be a table")
        name = step.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise DefinitionRunnerError(
                f"{run_name} step {index} needs a unique non-empty name"
            )
        names.add(name)
        script = _repo_path(step.get("script"), label=f"{run_name}.{name}.script")
        arguments = step.get("arguments", [])
        if not isinstance(arguments, list) or not all(
            isinstance(item, str) for item in arguments
        ):
            raise DefinitionRunnerError(f"{run_name}.{name}.arguments must be strings")
        chain_name = step.get("script_chain")
        chain = chains.get(chain_name) if isinstance(chain_name, str) else None
        if not isinstance(chain, dict):
            raise DefinitionRunnerError(
                f"{run_name}.{name}.script_chain must name a script chain"
            )
        dispatched = chain.get("scripts")
        if not isinstance(dispatched, list) or not dispatched:
            raise DefinitionRunnerError(
                f"script_chains.{chain_name}.scripts must be a non-empty list"
            )
        dispatch_paths = [
            _repo_path(value, label=f"script_chains.{chain_name}.scripts")
            for value in dispatched
        ]
        validated.append(
            {
                **step,
                "name": name,
                "script_path": script,
                "dispatch_paths": dispatch_paths,
            }
        )
    return config, run, validated


def _focused_commands(
    config: dict[str, Any],
    definition_path: Path,
    run_name: str,
) -> dict[str, list[str]]:
    """Return direct commands for named groups and sub-experiments owned by a run."""

    manager = SCRIPT_DIR / "run_experiments.py"
    common = [
        sys.executable,
        str(manager),
        "--config",
        str(definition_path.resolve()),
    ]
    commands: dict[str, list[str]] = {}
    definitions = config.get("run_definitions", {})
    definition = definitions.get(run_name) if isinstance(definitions, dict) else None
    if not isinstance(definition, dict):
        raise DefinitionRunnerError(f"Missing run definition for {run_name}")
    owner = pipeline_owner(run_name, definition)
    for group_name, group in config.get("experiment_groups", {}).items():
        if isinstance(group, dict) and pipeline_owner(group_name, group) == owner:
            commands[str(group_name)] = [*common, "--group", str(group_name)]
    for experiment_name, experiment in config.get("experiments", {}).items():
        if (
            isinstance(experiment, dict)
            and pipeline_owner(experiment_name, experiment) == owner
        ):
            commands[str(experiment_name)] = [*common, "--run", str(experiment_name)]
    for table_name in (
        "experiment_workflows",
        "conditional_calibration_workflows",
        "target_submission_workflows",
    ):
        table = config.get(table_name, {})
        if isinstance(table, dict) and run_name in table:
            commands[run_name] = [*common, "--run", run_name]
    promotions = config.get("promotions", {})
    if isinstance(promotions, dict):
        for promotion_name, promotion in promotions.items():
            if isinstance(promotion, dict) and promotion.get("workflow") == run_name:
                commands[str(promotion_name)] = [
                    *common,
                    "--run",
                    str(promotion_name),
                ]
    return commands


def _run_manager_command(command: list[str], *, label: str) -> None:
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise DefinitionRunnerError(
            f"{label} failed with exit code {completed.returncode}"
        )


def _command(
    definition_path: Path,
    step: dict[str, Any],
    *,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(step["script_path"]),
        "--config",
        str(definition_path.resolve()),
        *step.get("arguments", []),
    ]
    if force and (
        Path(step["script_path"]).name == "run_experiments.py"
        or step.get("supports_force") is True
    ):
        command.append("--force")
    return command


def main(definition_path: Path, run_name: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Execute the reviewable {run_name} experiment plan."
    )
    parser.add_argument(
        "--list",
        "--list-steps",
        "--dry-run",
        dest="list_steps",
        action="store_true",
        help="Print available targets and exact commands without running them.",
    )
    parser.add_argument(
        "--only",
        "--step",
        action="append",
        dest="selected_targets",
        metavar="NAME",
        help=(
            "Run only this step, group, or sub-experiment; repeat to select "
            "multiple targets."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show completion state without running the experiment.",
    )
    parser.add_argument(
        "--collect-figures",
        action="store_true",
        help="Refresh this experiment's top-level figure gallery only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forward --force to shared experiment-manager steps.",
    )
    args = parser.parse_args()

    try:
        config, run, steps = _execution_plan(definition_path.resolve(), run_name)
        focused = _focused_commands(config, definition_path, run_name)
        step_commands = {
            str(step["name"]): _command(definition_path, step, force=args.force)
            for step in steps
        }

        print(f"{run_name}: {run.get('title', run_name)}", flush=True)
        print(f"Settings:  {definition_path.resolve()}", flush=True)
        print(
            f"Artifacts: {run_directory(SCRIPT_DIR / 'experiments', run_name, run)}",
            flush=True,
        )
        if args.status:
            _run_manager_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_experiments.py"),
                    "--config",
                    str(definition_path.resolve()),
                    "--status",
                ],
                label=f"{run_name} status",
            )
            artifact_root = run_directory(
                SCRIPT_DIR / "experiments", run_name, run
            ).resolve()
            for step in steps:
                status_artifact = step.get("status_artifact")
                if not isinstance(status_artifact, str) or not status_artifact:
                    continue
                status_path = (artifact_root / status_artifact).resolve()
                try:
                    status_path.relative_to(artifact_root)
                except ValueError as error:
                    raise DefinitionRunnerError(
                        f"{step['name']}.status_artifact escapes the run folder"
                    ) from error
                state = "complete" if status_path.is_file() else "pending"
                print(
                    f"  {step['name']}: {state} ({status_path})",
                    flush=True,
                )
            return
        if args.collect_figures:
            _run_manager_command(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "run_experiments.py"),
                    "--config",
                    str(definition_path.resolve()),
                    "--collect-figures",
                    "--run",
                    run_name,
                ],
                label=f"{run_name} figure collection",
            )
            return

        commands: list[tuple[str, list[str], list[Path]]] = []
        if args.selected_targets:
            known = set(step_commands) | set(focused)
            unknown = sorted(set(args.selected_targets) - known)
            if unknown:
                raise DefinitionRunnerError(
                    f"Unknown {run_name} target(s): {', '.join(unknown)}; "
                    f"available: {', '.join(sorted(known))}"
                )
            for target in args.selected_targets:
                if target in step_commands:
                    step = next(item for item in steps if item["name"] == target)
                    commands.append(
                        (target, step_commands[target], step["dispatch_paths"])
                    )
                else:
                    command = list(focused[target])
                    if args.force:
                        command.append("--force")
                    commands.append((target, command, []))
        else:
            commands = [
                (
                    str(step["name"]),
                    step_commands[str(step["name"])],
                    step["dispatch_paths"],
                )
                for step in steps
            ]

        description = run.get("description")
        if isinstance(description, str) and description:
            print(description, flush=True)
        print("Execution plan:", flush=True)
        for index, (name, command, dispatch_paths) in enumerate(commands, start=1):
            print(f"  {index}. {name}: {subprocess.list2cmdline(command)}", flush=True)
            if dispatch_paths:
                print("     declared downstream scripts:", flush=True)
            for dispatch_path in dispatch_paths:
                relative = dispatch_path.relative_to(REPOSITORY_ROOT)
                print(f"       - {relative}", flush=True)
        if args.list_steps:
            extra_targets = sorted(set(focused) - set(step_commands))
            if extra_targets:
                print("Focused rerun targets:", flush=True)
                for target in extra_targets:
                    print(f"  - {target}", flush=True)
            return

        for index, (name, command, _) in enumerate(commands, start=1):
            print("\n" + "=" * 80, flush=True)
            print(f"{run_name} target {index}/{len(commands)}: {name}", flush=True)
            _run_manager_command(command, label=f"{run_name} target {name}")
        print(f"{run_name}: all requested steps completed", flush=True)
    except (DefinitionRunnerError, ExperimentConfigError) as error:
        print(f"Experiment definition stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
