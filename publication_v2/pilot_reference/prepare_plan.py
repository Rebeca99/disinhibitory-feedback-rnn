"""Select architecture-matched parent archives before training; create a job plan."""
import argparse
import json
from pathlib import Path
import re

from core import (CORE_ROUTES, EXTRA_ROUTES, DEFAULTS, VERSION, architecture_signature,
                  digest_file, parameter_digest, trusted_load, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", required=True, help="Directory containing original pretrained archives")
    p.add_argument("--anchor", required=True, help="Original Chapter 6 parent; included in selected parents")
    p.add_argument("--parent-count", type=int, default=2)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--routes", nargs="+", choices=CORE_ROUTES + EXTRA_ROUTES, default=CORE_ROUTES)
    p.add_argument("--with-motif-training-control", action="store_true")
    p.add_argument("--with-normmatched-combined", action="store_true")
    p.add_argument("--with-dense-direct-control", action="store_true")
    p.add_argument("--with-restricted-control", action="store_true")
    p.add_argument("--updates", type=int, default=DEFAULTS["updates"])
    p.add_argument("--validation-every", type=int, default=DEFAULTS["validation_every"])
    p.add_argument("--validation-trials", type=int, default=DEFAULTS["validation_trials"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--block-trials", type=int, default=DEFAULTS["block_trials"])
    p.add_argument("--jitter-ms", type=int, default=DEFAULTS["jitter_ms"])
    p.add_argument("--dynamics", choices=["legacy_additive", "pure_additive"], default="legacy_additive")
    p.add_argument("--target-seed", type=int, default=DEFAULTS["target_seed"])
    p.add_argument("--output-root", required=True)
    p.add_argument("--plan", default="plan.json")
    args = p.parse_args()
    if args.parent_count < 1 or min(args.updates, args.validation_every, args.validation_trials,
                                    args.block_trials, args.batch_size) < 1:
        p.error("Counts must be positive")
    anchor_path = Path(args.anchor).resolve()
    anchor = trusted_load(anchor_path)
    signature = architecture_signature(anchor)
    anchor_digest = parameter_digest(anchor)
    parents = []
    audit = []
    seen = set()
    candidates = [anchor_path] + sorted(path for path in Path(args.parent_dir).resolve().rglob("*")
                                       if path.is_file() and not path.name.startswith("."))
    for path in candidates:
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            saved = anchor if path == anchor_path else trusted_load(path)
            if saved.get("parent_stage") == "untrained_random":
                raise ValueError("Untrained random archive is not a pretrained parent")
            compatible = architecture_signature(saved) == signature
            weights_digest = anchor_digest if path == anchor_path else parameter_digest(saved)
            duplicate = any(item["parameter_sha256"] == weights_digest for item in parents)
            source_seed = saved["hp"].get("torch_seed", saved["hp"].get("seed"))
            if source_seed is None:
                match = re.search(r"wcst_(\d+)", path.name)
                source_seed = int(match.group(1)) if match else None
            if source_seed is not None:
                source_seed = int(source_seed)
            audit.append(dict(path=str(path), compatible=compatible, parameter_sha256=weights_digest,
                              duplicate=duplicate, source_training_seed=source_seed))
            if compatible and not duplicate:
                parents.append(dict(id="parent_{:02d}".format(len(parents) + 1), path=str(path),
                                    sha256=digest_file(path), parameter_sha256=weights_digest,
                                    source_training_seed=source_seed, source_name=path.name))
        except Exception as exc:
            audit.append(dict(path=str(path), error=str(exc)))
    write_json(Path(args.plan).with_suffix(".parent_audit.json"), audit)
    if len(parents) < args.parent_count:
        raise SystemExit("Only {} nonduplicate architecture-matched parents found; requested {}. "
                         "See parent_audit.json. Do not treat fine-tuned copies as independent parents."
                         .format(len(parents), args.parent_count))
    parents = parents[:args.parent_count]
    source_seeds = [r["source_training_seed"] for r in parents]
    if None in source_seeds or len(set(source_seeds)) != len(source_seeds):
        raise SystemExit("Independent source training seeds could not be verified; inspect archives/filenames")
    routes = list(dict.fromkeys(args.routes))
    if args.with_motif_training_control:
        routes += [r for r in ["baseline_motif_blocked", "sr_vip_motif_blocked"] if r not in routes]
    if args.with_normmatched_combined and "both_vip_normmatched" not in routes:
        routes.append("both_vip_normmatched")
    if args.with_dense_direct_control and "sr_direct_dense_normmatched" not in routes:
        routes.append("sr_direct_dense_normmatched")
    cfg = dict(DEFAULTS)
    for key in ["updates", "validation_every", "validation_trials", "batch_size", "block_trials",
                "jitter_ms", "target_seed", "dynamics"]:
        cfg[key] = getattr(args, key)
    root = Path(args.output_root).resolve()
    jobs = []
    for parent in parents:
        for seed in args.seeds:
            conditions = [(r, "full") for r in routes]
            if args.with_restricted_control:
                conditions += [(r, "feedback_readout") for r in ["baseline", "pfc_vip", "sr_vip", "both_vip"]]
            for route, adaptation in conditions:
                identity = "{}/{}/{}/seed_{}".format(parent["id"], adaptation, route, seed)
                jobs.append(dict(id=identity, parent=parent["path"], parent_id=parent["id"],
                                 parent_sha256=parent["sha256"], seed=seed, route=route,
                                 adaptation=adaptation, config=cfg, output=str(root / identity)))
    plan = dict(version=VERSION, parents=parents, architecture_signature=json.loads(signature),
                selection="anchor then filename order within architecture stratum, no new-outcome selection",
                source_seed_provenance="archive torch_seed/seed, with original wcst filename fallback",
                jobs=jobs)
    plan_path = Path(args.plan).resolve()
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("Plan already exists and differs. Use a new --plan filename and output root")
    write_json(plan_path, plan)
    print("Plan: {}\nParents: {}\nJobs: {}\nUpdates/job: {}".format(plan_path, len(parents), len(jobs), args.updates))
    print("Cluster preflight is required before launching the full matrix.")


if __name__ == "__main__":
    main()
