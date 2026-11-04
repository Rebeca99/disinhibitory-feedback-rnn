"""Initialize and train genuinely new parent circuits, then export completed parents."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch
from core import DEFAULTS, VERSION, checkpoint_state, digest_file, trusted_load, write_json
from model_working import Net_readoutSR_working


def initialize(args):
    template = trusted_load(args.template)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS, updates=args.updates, lr=args.lr, pfc_gain=1.0,
               history_stim=1.0, history_choice=1.0, feedback_noise=0.0,
               phase="parent_pretraining", train_hazard=[20, 20, 0.0])
    jobs, parents = [], []
    for seed in args.seeds:
        target = root / ("untrained_parent_{}.pt".format(seed))
        if target.exists():
            raise FileExistsError("Refusing to overwrite initialized parent: " + str(target))
        hp = dict(template["hp"], torch_seed=seed, batch_size=cfg["batch_size"],
                  prev_rew_to_pfc_scale=1.0, prev_rew_to_vip_scale=1.0,
                  prev_rew_to_pfc_default=True, prev_rew_to_pfc_vip=False,
                  prev_rew_to_sr_vip=False, fdbk_to_vip=False)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            model = Net_readoutSR_working(hp)
        # Same wiring, genuinely new learned parameters; no trained weights copied.
        template_state = checkpoint_state(template)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "mask" in name or name in ["rnn.w_fix", "rnn.dend2soma"]:
                    parameter.copy_(template_state[name])
        saved = dict(hp=hp, model_state_dict=model.state_dict(), parent_stage="untrained_random",
                     provenance=dict(random_initialization_seed=seed, template_architecture=args.template,
                                     no_pretrained_learned_weights_copied=True))
        torch.save(saved, target)
        parent_id = "fresh_parent_{}".format(seed)
        parents.append(dict(id=parent_id, path=str(target), source_name=target.name,
                            source_training_seed=seed, sha256=digest_file(target)))
        identity = parent_id + "/full/baseline/seed_" + str(seed)
        jobs.append(dict(id=identity, parent=str(target), parent_id=parent_id, parent_sha256=digest_file(target),
                         route="baseline", seed=seed, adaptation="full", config=cfg,
                         output=str(root / identity)))
    plan = dict(version=VERSION, phase="parent_pretraining", parents=parents, jobs=jobs,
                task="fixed 20-trial uncued rule blocks, full informative history, uncorrupted feedback",
                note="Controlled new parent task; not a claim to reproduce all original Liu-Wang training details")
    write_json(args.plan, plan)
    print("Fresh parent plan:", args.plan, "jobs:", len(jobs))


def promote(args):
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("phase") != "parent_pretraining":
        raise ValueError("Not a fresh-parent pretraining plan")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    for job in plan["jobs"]:
        checkpoint = Path(job["output"]) / "final.pt"
        saved = trusted_load(checkpoint)
        if saved["updates"] != job["config"]["updates"]:
            raise ValueError("Parent pretraining incomplete")
        final = saved["validation_records"][-1]
        passed = final["choice_accuracy"] >= args.min_choice and final["rule_accuracy"] >= args.min_rule
        audit.append(dict(parent=job["parent_id"], source=str(checkpoint),
                          final_validation=final, passes_declared_quality_gate=passed))
        if not passed:
            continue
        target = output / (job["parent_id"] + ".pt")
        compact = dict(hp=saved["hp"], model_state_dict=saved["model_state_dict"],
                       parent_stage="pretrained", provenance=dict(source=str(checkpoint), seed=job["seed"],
                       updates=saved["updates"], validation=final, quality_gate=dict(choice=args.min_choice, rule=args.min_rule)))
        torch.save(compact, target)
    write_json(output / "parent_quality_audit.json", audit)
    print("Quality gate uses the parent's own held-out pretraining task, before any route comparisons.")
    print("Exported", sum(r["passes_declared_quality_gate"] for r in audit), "of", len(audit), "parents.")
    if not all(r["passes_declared_quality_gate"] for r in audit):
        raise SystemExit("Some parents need further pretraining. Inspect learning curves; do not silently discard failed seeds.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--template", required=True)
    init.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    init.add_argument("--updates", type=int, default=5000)
    init.add_argument("--lr", type=float, default=1e-3)
    init.add_argument("--output-root", required=True)
    init.add_argument("--plan", default="parent_training_plan.json")
    export = sub.add_parser("promote")
    export.add_argument("--plan", required=True)
    export.add_argument("--output", default="additional_parents")
    export.add_argument("--min-choice", type=float, default=.75)
    export.add_argument("--min-rule", type=float, default=.75)
    args = p.parse_args()
    if args.command == "initialize":
        initialize(args)
    else:
        promote(args)


if __name__ == "__main__":
    main()
