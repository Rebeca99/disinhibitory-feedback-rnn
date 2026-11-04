"""Create a separate immutable plan continuing every completed pilot condition."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

from core import VERSION, code_digest, digest_file, trusted_load, write_json
from continuation import SCHEMA, validate_completed_source, verify_source_code


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def prepare(source_plan_path, source_code_directory, output_root, updates, plan_path=None):
    source_path = Path(source_plan_path).expanduser().resolve()
    source_directory = Path(source_code_directory).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    destination = Path(plan_path).expanduser().resolve() if plan_path else root / "plan.json"
    if overlaps(root, source_path.parent) or overlaps(root, source_directory):
        raise ValueError("Continuation output must be separate from the pilot data and original code.")
    if root != destination.parent and root not in destination.parents:
        raise ValueError("The continuation plan must be inside its output root")
    original = json.loads(source_path.read_text())
    if original.get("version") != VERSION or not original.get("jobs"):
        raise ValueError("Source is not a supported publication plan")
    source_hash = verify_source_code(source_directory)
    plan_hash = digest_file(source_path)
    jobs, budgets, seen = [], set(), set()
    for source in original["jobs"]:
        identity = "{}/{}/{}/seed_{}".format(source["parent_id"], source["adaptation"], source["route"], source["seed"])
        if source["id"] != identity or identity in seen:
            raise ValueError("Source identities must be unique, canonical experiment IDs")
        seen.add(identity)
        if overlaps(root, Path(source["output"]).resolve()):
            raise ValueError("Continuation destination overlaps a source job")
        checkpoint = (Path(source["output"]) / "final.pt").resolve()
        if not checkpoint.exists():
            raise ValueError("Source job has not completed: " + identity)
        checksum = digest_file(checkpoint)
        saved = trusted_load(checkpoint)
        validate_completed_source(source, saved, source_hash)
        if updates <= saved["updates"]:
            raise ValueError("Total updates must exceed source updates for every job")
        if digest_file(checkpoint) != checksum:
            raise ValueError("Source checkpoint changed while being inspected")
        budgets.add(saved["updates"])
        job = copy.deepcopy(source)
        job["config"]["updates"] = updates
        job["output"] = str(root / identity)
        job["continuation"] = dict(schema=SCHEMA, source_plan=str(source_path),
            source_plan_sha256=plan_hash, source_code_directory=str(source_directory),
            source_code_sha256=source_hash, source_job_id=identity,
            source_checkpoint=str(checkpoint), source_checkpoint_sha256=checksum,
            source_updates=saved["updates"], source_elapsed_seconds=saved["elapsed_seconds"])
        jobs.append(job)
    if len(budgets) != 1:
        raise ValueError("All source jobs must have the same completed update budget")
    plan = copy.deepcopy(original)
    plan["jobs"] = jobs
    plan["continuation"] = dict(schema=SCHEMA, source_plan=str(source_path),
        source_plan_sha256=plan_hash, source_code_sha256=source_hash,
        code_sha256=code_digest(), source_updates=next(iter(budgets)), total_updates=updates,
        policy="all source jobs, final source state, no performance-based selection")
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "preparation.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if json.loads(destination.read_text()) != plan:
                raise ValueError("Continuation plan already differs; choose a new output root")
        else:
            if any((Path(j["output"]) / name).exists() for j in jobs
                   for name in ["job.json", "latest.pt", "final.pt"]):
                raise ValueError("Destination already contains runs without this continuation plan")
            write_json(destination, plan)
    start = plan["continuation"]["source_updates"]
    print("Verified source jobs: {}\nContinue updates: {} -> {} ({} additional/job)\nPlan: {}".format(
          len(jobs), start, updates, updates - start, destination), flush=True)
    print("Model, Adam state, seed streams and prior learning history will be restored; source files are read-only.", flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--source-code-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--updates", type=int, default=1000, help="TOTAL updates, not additional updates")
    parser.add_argument("--plan")
    args = parser.parse_args()
    prepare(args.source_plan, args.source_code_dir, args.output_root, args.updates, args.plan)


if __name__ == "__main__":
    main()
