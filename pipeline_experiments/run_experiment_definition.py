"""Execute the ordered script plan declared by one PE_run_X definition."""

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


REPOSITORY_ROOT = SCRIPT_DIR.parent


class DefinitionRunnerError(ValueError):
    """Explain an invalid run execution plan."""


def _repo_path(value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DefinitionRunnerError(f"{label} must be a non-empty path")
    path = (REPOSITORY_ROOT / value).resolve()
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    return run, validated


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
    if force and Path(step["script_path"]).name == "run_experiments.py":
        command.append("--force")
    return command


def main(definition_path: Path, run_name: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Execute the reviewable {run_name} experiment plan."
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print the exact script commands without running them.",
    )
    parser.add_argument(
        "--step",
        action="append",
        dest="selected_steps",
        help="Run only this named step; repeat to select multiple steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forward --force to shared experiment-manager steps.",
    )
    args = parser.parse_args()

    try:
        run, steps = _execution_plan(definition_path.resolve(), run_name)
        if args.selected_steps:
            requested = set(args.selected_steps)
            known = {str(step["name"]) for step in steps}
            unknown = sorted(requested - known)
            if unknown:
                raise DefinitionRunnerError(
                    f"Unknown {run_name} step(s): {', '.join(unknown)}"
                )
            steps = [step for step in steps if step["name"] in requested]

        print(f"{run_name}: {run.get('title', run_name)}", flush=True)
        description = run.get("description")
        if isinstance(description, str) and description:
            print(description, flush=True)
        print("Execution plan:", flush=True)
        commands = [
            _command(definition_path, step, force=args.force) for step in steps
        ]
        for index, (step, command) in enumerate(zip(steps, commands), start=1):
            print(f"  {index}. {step['name']}: {subprocess.list2cmdline(command)}", flush=True)
            print("     declared downstream scripts:", flush=True)
            for dispatch_path in step["dispatch_paths"]:
                relative = dispatch_path.relative_to(REPOSITORY_ROOT)
                print(f"       - {relative}", flush=True)
        if args.list_steps:
            return

        for index, (step, command) in enumerate(zip(steps, commands), start=1):
            print("\n" + "=" * 80, flush=True)
            print(f"{run_name} step {index}/{len(steps)}: {step['name']}", flush=True)
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            if completed.returncode != 0:
                raise DefinitionRunnerError(
                    f"{run_name} stopped at {step['name']} with exit code "
                    f"{completed.returncode}"
                )
        print(f"{run_name}: all requested steps completed", flush=True)
    except (DefinitionRunnerError, ExperimentConfigError) as error:
        print(f"Experiment definition stopped:\n{error}", file=sys.stderr)
        raise SystemExit(1) from error
