"""Equal-update, resumable supervised fine-tuning from a trusted parent archive."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from core import VERSION, build_model, code_digest, digest_file, feedback_audit, stream_seed, trusted_load, write_json
from protocol import run_sequence, validation
from continuation import fingerprint_for, load_continuation_source


def atomic_save(path, value):
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def train(job, device):
    cfg = job["config"]
    root = Path(job["output"])
    root.mkdir(parents=True, exist_ok=True)
    if digest_file(job["parent"]) != job["parent_sha256"]:
        raise ValueError("Parent checksum changed since plan was created")
    fingerprint = fingerprint_for(job)
    if (root / "job.json").exists() and json.loads((root / "job.json").read_text()) != fingerprint:
        raise ValueError("Existing run has a different config; choose a new output directory")
    write_json(root / "job.json", fingerprint)
    saved = trusted_load(job["parent"])
    model, hp = build_model(saved, job["route"], cfg, job["adaptation"], device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    update, elapsed, records, val_records = 0, 0.0, [], []
    latest = root / "latest.pt"
    origin = job.get("continuation")
    source = load_continuation_source(job) if origin else None
    if latest.exists():
        resume = trusted_load(latest)
        if resume["fingerprint"] != fingerprint:
            raise ValueError("Resume checkpoint fingerprint mismatch")
    else:
        resume = source
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        update, elapsed = resume["updates"], resume["elapsed_seconds"]
        records, val_records = resume["training_records"], resume["validation_records"]
    if origin:
        if not origin["source_updates"] <= update <= cfg["updates"]:
            raise ValueError("Continuation update counter is outside its declared range")
        write_json(root / "continuation.json", origin)
        if not latest.exists():
            write_json(root / "continuation_start_feedback.json", feedback_audit(model))
        print("CONTINUE {}: completed {}, next update {}, total budget {}".format(
              job["id"], update, update + 1, cfg["updates"]), flush=True)
    del source, resume
    initial_audit = feedback_audit(build_model(saved, job["route"], cfg, job["adaptation"], "cpu")[0])
    write_json(root / "initial_feedback.json", initial_audit)
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *args: stop.__setitem__(0, True))
    start = time.monotonic()

    def persist(reason):
        state = dict(version=VERSION, model_state_dict=model.state_dict(), hp=hp, config=cfg,
                     route=job["route"], adaptation=job["adaptation"], fingerprint=fingerprint,
                     parent_id=job["parent_id"], seed=job["seed"], updates=update,
                     elapsed_seconds=elapsed + time.monotonic() - start,
                     optimizer_state_dict=optimizer.state_dict(), training_records=records,
                     validation_records=val_records, termination_reason=reason,
                     populations={k: v.tolist() for k, v in model.rnn.cg_idx.items()},
                     feedback_audit=feedback_audit(model))
        if origin:
            state["continuation"] = origin
        atomic_save(latest, state)
        if reason == "equal_update_budget_completed":
            atomic_save(root / "final.pt", state)
        for filename, data in [("training.csv", records), ("validation.csv", val_records)]:
            if data:
                with open(root / filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
        diagnostics = dict(parent=job["parent_id"], route=job["route"], seed=job["seed"],
                           adaptation=job["adaptation"], updates=update, planned_updates=cfg["updates"],
                           termination_reason=reason, final_state_selected=True,
                           evaluation_based_selection=False, elapsed_seconds=state["elapsed_seconds"],
                           inherited_criterion_scope="diagnostic only, not an early stopping rule",
                           criterion_reached=any(v["inherited_criterion_met"] for v in val_records),
                           final_training=records[-1] if records else None,
                           last_validation=val_records[-1] if val_records else None,
                           feedback=state["feedback_audit"])
        if origin:
            diagnostics.update(continuation=origin,
                continuation_updates=update - origin["source_updates"],
                continuation_elapsed_seconds=state["elapsed_seconds"] - origin["source_elapsed_seconds"])
        write_json(root / "audit.json", diagnostics)

    if update >= cfg["updates"]:
        persist("equal_update_budget_completed")
        print("Already complete: " + job["id"], flush=True)
        return
    for update_index in range(update, cfg["updates"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seed = stream_seed(job["seed"], "train", update_index)
        out = run_sequence(model, cfg, seed, cfg["block_trials"], cfg["train_hazard"], training=True,
                           initial_rule=stream_seed(seed, "initial_rule") % 2)
        if not torch.isfinite(out["loss"]):
            persist("nonfinite_loss")
            raise FloatingPointError("Nonfinite loss: inspect latest checkpoint")
        out["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
        if not torch.isfinite(grad):
            persist("nonfinite_gradient")
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        update = update_index + 1
        record = dict(update=update, loss=float(out["loss"].detach()),
                      choice_loss=float(out["choice_loss"].detach()), rule_loss=float(out["rule_loss"].detach()),
                      choice_accuracy=out["metrics"][0], rule_accuracy=out["metrics"][1],
                      choice_tie_rate=out["metrics"][2], rule_tie_rate=out["metrics"][3],
                      gradient_norm_before_clip=float(grad))
        # A training block starts from zero; the graph is retained only within it.
        del out
        records.append(record)
        print(json.dumps(record), flush=True)
        if update % cfg["validation_every"] == 0 or update == cfg["updates"]:
            val = validation(model, cfg, stream_seed(job["seed"], "validation"),
                             cfg["validation_sequences"], cfg["validation_trials"], cfg["train_hazard"])
            rows = val["rows"]
            choice_acc = float(np.mean([r["choice_accuracy"] for r in rows]))
            rule_acc = float(np.mean([r["rule_accuracy"] for r in rows]))
            switch_rate = float(np.mean([r["true_switch"] for r in rows]))
            threshold = 1 - switch_rate - 0.05
            v = dict(update=update, loss=val["losses"][0], choice_loss=val["losses"][1],
                     rule_loss=val["losses"][2], choice_accuracy=choice_acc, rule_accuracy=rule_acc,
                     choice_tie_rate=float(np.mean([r["choice_tie"] for r in rows])),
                     rule_tie_rate=float(np.mean([r["rule_tie"] for r in rows])),
                     inherited_threshold=threshold, inherited_criterion_met=False)
            recent = val_records[-4:] + [v]
            v["inherited_criterion_met"] = bool(len(recent) == 5 and
                np.mean([r["choice_accuracy"] for r in recent]) >= threshold and
                np.mean([r["rule_accuracy"] for r in recent]) >= threshold)
            val_records.append(v)
            print("VALIDATION " + json.dumps(v), flush=True)
        if stop[0]:
            persist("interrupted_resumable")
            return
        if update % cfg["checkpoint_every"] == 0:
            persist("running_resumable")
    persist("equal_update_budget_completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for smoke tests")
    plan = json.loads(Path(args.plan).read_text())
    job = plan["jobs"][args.job_index]
    Path(job["output"]).mkdir(parents=True, exist_ok=True)
    with open(Path(job["output"]) / "training.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This job is already running; duplicate workers are not allowed")
        train(job, args.device)


if __name__ == "__main__":
    main()
