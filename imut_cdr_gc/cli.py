"""Explicit project-local stage commands; use --help without model dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .records import new_output, read_jsonl, relative_file, sha256, write_json, write_jsonl


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    top.add_argument("--project-root", type=Path, default=Path("."),
                     help="Working project directory (default: current directory); paths are relative to it")
    commands = top.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run an explicit JSON pipeline with relative paths")
    run.add_argument("--config", required=True, help="Pipeline JSON; see examples/demo_pipeline.json")
    run.add_argument("--dry-run", action="store_true", help="Validate the plan without running stages or models")
    prepare = commands.add_parser("prepare", help="Validate founder full chains and CDR-to-chain mapping")
    prepare.add_argument("--founder", required=True)
    prepare.add_argument("--output-dir", required=True)
    graph = commands.add_parser("prepare-antigen", help="Materialize an explicitly prepared antigen pocket graph")
    graph.add_argument("--atoms", required=True, help="Safe numeric/Unicode atom NPZ; no pickle")
    graph.add_argument("--atoms-sha256", required=True)
    graph.add_argument("--features", required=True, help="Explicit JM-Epi residue-feature JSON, not an inferred scorer table")
    graph.add_argument("--features-sha256", required=True)
    graph.add_argument("--antigen-id", required=True)
    graph.add_argument("--cutoff", type=float, default=8.0)
    graph.add_argument("--output-dir", required=True)
    generate = commands.add_parser("generate", help="Generate one JM or JM-Epi joint-proposal batch")
    generate.add_argument("--model", choices=("jm", "jm-epi"), default="jm-epi",
                          help="Joint model or antigen-conditioned joint model (default: jm-epi)")
    generate.add_argument("--input", required=True, help="Ordered JSONL proposals with id/fr_cdr_seq/proposal_positions")
    generate.add_argument("--founder", required=True)
    generate.add_argument("--config", required=True, help="Explicit local generator asset and antigen graph config")
    generate.add_argument("--version", required=True, choices=("v1", "v2", "v3"))
    generate.add_argument("--output-dir", required=True)
    fold = commands.add_parser("fold", help="Predict antibody-only folds with explicit pinned native AB2 assets")
    fold.add_argument("--input", required=True)
    fold.add_argument("--assets", required=True, help="Pinned source, weight and numbering asset manifest")
    fold.add_argument("--device", default="cpu", help="Explicit device; no automatic GPU selection")
    fold.add_argument("--output-dir", required=True)
    rebuild = commands.add_parser("rebuild-complexes", help="Rebuild mutant side chains on a fixed founder complex; no docking")
    rebuild.add_argument("--input", required=True)
    rebuild.add_argument("--founder", required=True)
    rebuild.add_argument("--runtime", required=True, help="Explicit previously inspected CPU PDBFixer/OpenMM identity")
    rebuild.add_argument("--output-dir", required=True)
    score = commands.add_parser("score", help="Measure corrected native DeepCDR scores, never silently replace errors")
    score.add_argument("--input", required=True)
    score.add_argument("--config", required=True)
    score.add_argument("--models", nargs="+", choices=("esm2", "3d"), required=True)
    score.add_argument("--output-dir", required=True)
    likelihood = commands.add_parser("likelihood", help="Measure JM-Epi native likelihood for P5 eligibility")
    likelihood.add_argument("--input", required=True)
    likelihood.add_argument("--assets", required=True, help="Pinned native likelihood asset manifest")
    likelihood.add_argument("--calibration", help="Independent reference P5 calibration, with no upper bound")
    likelihood.add_argument("--device", default="cpu")
    likelihood.add_argument("--batch-size", type=int, default=64)
    likelihood.add_argument("--output-dir", required=True)
    select = commands.add_parser("select", help="Apply JM-Epi eligibility, then policy-selected Top-k or random sampling")
    select.add_argument("--input", required=True)
    select.add_argument("--policy", required=True)
    select.add_argument("--excluded", help="Existing full H/L sequence pairs to exclude")
    select.add_argument("--output-dir", required=True)
    export = commands.add_parser("export", help="Verify and copy existing full complexes alongside library.jsonl")
    export.add_argument("--input", required=True)
    export.add_argument("--founder", required=True, help="Prepared founder with fixed full-complex reference identity")
    export.add_argument("--output-dir", required=True)
    return top


def generate_run(args):
    from .generation import load_local_generator, load_antigen_graph
    from .workflow import reconstruct_proposals
    from .preparation import prepare_founder
    root = args.project_root
    source, founder_file = relative_file(root, args.input), relative_file(root, args.founder)
    config_file = relative_file(root, args.config)
    pinned = {path: sha256(path) for path in (source, founder_file, config_file)}
    cfg = json.loads(config_file.read_text())
    model_kind = getattr(args, "model", "jm-epi")
    model_name = {"jm": "iMut-CDR-JM", "jm-epi": "iMut-CDR-JM-Epi"}[model_kind]
    if cfg.get("model", model_kind) != model_kind:
        raise ValueError("Generator config model conflicts with --model")
    founder = prepare_founder(json.loads(founder_file.read_text()))
    proposals = read_jsonl(source)
    if any(row.get("founder_fr_cdr_seq") != founder.get("fr_cdr_seq") for row in proposals):
        raise ValueError("Proposal founders differ from the prepared founder")
    for row in proposals:
        for key, expected in (("design", founder["design"]), ("antigen_id", founder["antigen_id"]), ("version", args.version)):
            if key in row and row[key] != expected:
                raise ValueError("Proposal " + key + " conflicts with explicit founder/version")
    assets = cfg["generator"]
    checkpoint = relative_file(root, assets["checkpoint"]["path"], assets["checkpoint"]["sha256"])
    # Resolve model directories without permitting an implicit external download.
    model_relative = Path(assets["model_dir"])
    if model_relative.is_absolute() or ".." in model_relative.parts:
        raise ValueError("Model directory must be inside the project")
    model_dir = root / model_relative
    if not model_dir.resolve(strict=True).is_relative_to(root) or not model_dir.is_dir():
        raise ValueError("Model directory is missing or escapes the project")
    batch_size = cfg.get("batch_size", 1)
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("Declare a batch_size from 1 to 256; no automatic memory fallback")
    graph = None
    if model_kind == "jm-epi":
        graph_spec = cfg["antigen_graph"]
        if graph_spec.get("antigen_id") != founder["antigen_id"]:
            raise ValueError("The antigen graph must explicitly identify the prepared founder target")
        graph_path = relative_file(root, graph_spec["path"], graph_spec["sha256"])
        graph = load_antigen_graph(graph_path, graph_spec["sha256"])
    elif "antigen_graph" in cfg:
        raise ValueError("Unconditioned JM does not accept an antigen graph; use --model jm-epi")
    generator = load_local_generator(model_dir=model_dir, checkpoint=checkpoint,
        expected_checkpoint_sha256=assets["checkpoint"]["sha256"], asset_sha256=assets["asset_sha256"],
        model_config=cfg.get("model_config"), model_kind=model_kind, **cfg.get("sampling", {}))
    output = new_output(root, args.output_dir)
    try:
        predicted = []
        for begin in range(0, len(proposals), batch_size):
            batch = proposals[begin:begin + batch_size]
            predicted.extend(generator.propose(batch, graphs=[graph] * len(batch) if graph is not None else None))
        if [r["id"] for r in predicted] != [r["id"] for r in proposals]:
            raise ValueError("Generator did not preserve the ordered proposal identities")
        rows = reconstruct_proposals(founder, predicted, version=args.version, model=model_name)
        if any(sha256(path) != value for path, value in pinned.items()):
            raise ValueError("Input/founder/config changed during generation; outputs are not committed")
        write_jsonl(output / "candidates.jsonl", rows)
        receipt = {"status": "generated_not_screened", "count": len(rows), "input_sha256": pinned[source],
                   "founder_sha256": pinned[founder_file], "config_sha256": pinned[config_file],
                   "output_sha256": sha256(output / "candidates.jsonl"),
                   "batch_size": batch_size, "model": model_name, "generator_identity": generator.provenance,
                   "v3_per_step_selection_executed": False, "selected": 0, "pdb_generated": 0}
        write_json(output / "generation.json", receipt)
        return receipt
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error)})
        raise


def execute(args):
    """Run one parsed stage and return its receipt without formatting stdout."""
    args.project_root = args.project_root.resolve(strict=True)
    from . import workflow
    if args.command == "run":
        from .pipeline import run_pipeline
        result = run_pipeline(args.project_root, args.config, dry_run=args.dry_run)
    elif args.command == "prepare":
        result = workflow.prepare_run(args.project_root, args.founder, args.output_dir)
        result = {"status": "prepared_not_generated", "id": result["id"]}
    elif args.command == "prepare-antigen":
        from .antigen_graph import prepare_antigen_graph
        result = prepare_antigen_graph(project_root=args.project_root,
            atom_npz=args.atoms, atom_sha256=args.atoms_sha256,
            feature_json=args.features, feature_sha256=args.features_sha256,
            antigen_id=args.antigen_id, cutoff=args.cutoff, output_dir=args.output_dir)
    elif args.command == "generate":
        result = generate_run(args)
    elif args.command == "fold":
        from .folding import fold_antibodies
        _, result = fold_antibodies(read_jsonl(relative_file(args.project_root, args.input)),
            project_root=args.project_root, output_dir=args.output_dir,
            assets=json.loads(relative_file(args.project_root, args.assets).read_text()), device=args.device)
    elif args.command == "rebuild-complexes":
        from .rebuilding import rebuild_complexes
        _, result = rebuild_complexes(read_jsonl(relative_file(args.project_root, args.input)),
            project_root=args.project_root, output_dir=args.output_dir,
            prepared_founder=json.loads(relative_file(args.project_root, args.founder).read_text()),
            runtime_identity=json.loads(relative_file(args.project_root, args.runtime).read_text()))
    elif args.command == "score":
        if len(set(args.models)) != len(args.models):
            raise ValueError("Choose each scoring model only once")
        result = workflow.score_run(args.project_root, args.input, args.config, args.output_dir, args.models)
    elif args.command == "likelihood":
        from .naturalness import score_campaign_likelihood_pairs
        source = relative_file(args.project_root, args.input)
        assets = relative_file(args.project_root, args.assets)
        calibration = relative_file(args.project_root, args.calibration) if args.calibration else None
        _, result = score_campaign_likelihood_pairs(read_jsonl(source), args.output_dir,
            project_root=args.project_root, assets=json.loads(assets.read_text()),
            calibration=json.loads(calibration.read_text()) if calibration else None,
            device=args.device, batch_size=args.batch_size)
    elif args.command == "select":
        result = workflow.select_run(args.project_root, args.input, args.policy, args.output_dir, args.excluded)
        result = {key: value for key, value in result.items() if key != "decisions"}
    elif args.command == "export":
        from .complexes import export_library
        result = export_library(read_jsonl(relative_file(args.project_root, args.input)),
                                project_root=args.project_root, output_dir=args.output_dir,
                                prepared_founder=json.loads(relative_file(args.project_root, args.founder).read_text()))
    else:
        raise ValueError("Unknown workflow command")
    return result


def main(argv=None):
    result = execute(parser().parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
