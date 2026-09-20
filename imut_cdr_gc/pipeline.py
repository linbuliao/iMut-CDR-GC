"""Run a named sequence of existing stages from one portable JSON configuration.

This runner does not infer missing stages, launch shell commands, retry failed
models or turn an endpoint-only sequence into a stepwise-screened V3 library.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import re
import time

from .records import new_output, relative_file, sha256, write_json

SCHEMA = "imut-cdr-gc-pipeline.v1"
COMMANDS = frozenset(("prepare", "prepare-antigen", "generate", "fold",
    "rebuild-complexes", "score", "likelihood", "select", "export"))
PATH_ARGUMENTS = frozenset(("input", "founder", "config", "assets", "policy",
    "excluded", "runtime", "atoms", "features", "calibration", "output_dir"))


def _relative_path(root, value, *, output=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every path must be a nonempty project-relative string")
    relative = Path(value)
    if (relative.is_absolute() or PureWindowsPath(value).drive or "\\" in value
            or ".." in relative.parts or not relative.parts):
        raise ValueError("Use project-relative paths with forward slashes")
    resolved = (root / relative).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Pipeline paths must stay inside the project")
    if output:
        if any(re.search(r"(?:19|20)\d{2}[-_]?\d{2}[-_]?\d{2}", part) for part in relative.parts):
            raise ValueError("Output directories use descriptive names, not dates")
        if (root / relative).exists() or (root / relative).is_symlink():
            raise FileExistsError("Pipeline outputs must be new: " + value)
    return resolved


def load_plan(root, config_path):
    """Validate stage arguments and output collisions without loading models."""
    from .cli import parser

    root = Path(root).resolve(strict=True)
    source = relative_file(root, config_path)
    source_sha = sha256(source)
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"schema", "output_dir", "steps"}:
        raise ValueError("Pipeline requires exactly schema, output_dir and steps")
    if config["schema"] != SCHEMA:
        raise ValueError("Unsupported pipeline schema")
    steps = config["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("The pipeline must contain at least one explicit stage")
    outputs = [_relative_path(root, config["output_dir"], output=True)]
    names, planned = set(), []
    cli = parser()
    subparsers = next(action for action in cli._actions if action.dest == "command")
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"name", "command", "arguments"}:
            raise ValueError("Each stage requires exactly name, command and arguments")
        name, command, arguments = step["name"], step["command"], step["arguments"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("Stage names must be unique short lowercase identifiers")
        names.add(name)
        if not isinstance(command, str) or command not in COMMANDS:
            raise ValueError("Only supported stage commands are allowed; no nested pipeline or shell commands")
        if not isinstance(arguments, dict) or "output_dir" not in arguments:
            raise ValueError("Each stage must declare its own new output_dir")
        actions = {action.dest: action for action in subparsers.choices[command]._actions
                   if action.dest != "help"}
        if set(arguments) - set(actions):
            raise ValueError("Unknown argument for stage " + name)
        if any(action.required and key not in arguments for key, action in actions.items()):
            raise ValueError("Missing required argument for stage " + name)
        argv = ["--project-root", str(root), command]
        for key, value in arguments.items():
            if key in PATH_ARGUMENTS:
                path = _relative_path(root, value, output=key == "output_dir")
                if key == "output_dir":
                    outputs.append(path)
            action = actions[key]
            values = value if isinstance(value, list) else [value]
            if not values or (isinstance(value, list) and action.nargs not in ("+", "*")):
                raise ValueError("Unexpected list argument in stage " + name)
            if any(type(item) not in (str, int, float) for item in values):
                raise ValueError("Stage argument values must be strings or numbers")
            argv.extend([action.option_strings[0], *map(str, values)])
        try:
            parsed = cli.parse_args(argv)
        except SystemExit as error:
            raise ValueError("Invalid arguments for stage " + name) from error
        planned.append((step, parsed))
    for index, path in enumerate(outputs):
        for other in outputs[:index]:
            if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Stage and pipeline receipt output directories must not overlap")
    if sha256(source) != source_sha:
        raise ValueError("Pipeline configuration changed while validating")
    return config, source, source_sha, planned


def run_pipeline(root, config_path, *, dry_run=False, stage_executor=None):
    from .cli import execute

    root = Path(root).resolve(strict=True)
    config, source, source_sha, planned = load_plan(root, config_path)
    plan = dict(schema=SCHEMA, config=config_path, config_sha256=source_sha,
        output_dir=config["output_dir"], stages=[step for step, _ in planned],
        automatic_retry=False, scientific_validation=False)
    if dry_run:
        return dict(plan, status="plan_validated_not_executed", model_calls=0)
    output = new_output(root, config["output_dir"])
    write_json(output / "plan.json", plan)
    stage_executor = stage_executor or execute
    completed, started = [], time.monotonic()
    failed_stage = None
    try:
        for index, (step, parsed) in enumerate(planned, 1):
            failed_stage = step["name"]
            if sha256(source) != source_sha:
                raise ValueError("Pipeline configuration changed during execution")
            before = time.monotonic()
            started_utc = datetime.now(timezone.utc).isoformat()
            result = stage_executor(parsed)
            if not isinstance(result, dict):
                raise TypeError("Every stage must return an explicit result object")
            receipt = dict(name=step["name"], command=step["command"],
                started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - before, result=result)
            if sha256(source) != source_sha:
                raise ValueError("Pipeline configuration changed during execution")
            write_json(output / f"stage_{index:02d}_{step['name']}.json", receipt)
            completed.append(receipt)
            if result.get("status") in ("failed", "interrupted", "insufficient_eligible", "incomplete"):
                raise RuntimeError("Stage " + step["name"] + " did not complete its requested output: " + result["status"])
        completion = dict(status="complete", meaning="All configured software stages completed",
            config_sha256=source_sha, completed_steps=len(completed),
            stages=[dict(name=row["name"], status=row["result"].get("status")) for row in completed],
            elapsed_seconds=time.monotonic() - started,
            completed_utc=datetime.now(timezone.utc).isoformat(), scientific_validation=False)
        write_json(output / "completion.json", completion)
        return completion
    except BaseException as error:
        write_json(output / "failure.json", dict(status="failed", failed_stage=failed_stage,
            completed_steps=len(completed), config_sha256=source_sha,
            elapsed_seconds=time.monotonic() - started,
            error=dict(type=type(error).__name__, message=str(error)), automatic_retry=False))
        raise
