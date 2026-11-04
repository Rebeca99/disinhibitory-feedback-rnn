"""Merge completed seed-1 and replication plans into an analysis-only plan."""

import argparse
import json
from pathlib import Path

from core import write_json


def parent_identity(parent):
    return (
        parent["id"],
        parent["sha256"],
        parent.get("parameter_sha256"),
        parent.get("source_training_seed"),
    )


def job_identity(job):
    return job["parent_id"], job["seed"], job["adaptation"], job["route"]


def comparable_job(job):
    return {
        "parent_id": job["parent_id"],
        "parent_sha256": job["parent_sha256"],
        "adaptation": job["adaptation"],
        "route": job["route"],
        "config": job["config"],
    }


def load_plan(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--replication-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path, base = load_plan(args.base_plan)
    replication_path, replication = load_plan(args.replication_plan)
    output = Path(args.output).expanduser().resolve()

    if base["version"] != replication["version"]:
        raise ValueError("Plan versions differ")
    if base["architecture_signature"] != replication["architecture_signature"]:
        raise ValueError("Architecture signatures differ")
    if [parent_identity(p) for p in base["parents"]] != [parent_identity(p) for p in replication["parents"]]:
        raise ValueError("Parent identities or ordering differ")

    base_templates = {
        (job["parent_id"], job["adaptation"], job["route"]): comparable_job(job)
        for job in base["jobs"]
    }
    for job in replication["jobs"]:
        key = job["parent_id"], job["adaptation"], job["route"]
        if key not in base_templates:
            raise ValueError("Replication condition absent from base plan: {}".format(key))
        if comparable_job(job) != base_templates[key]:
            raise ValueError("Replication condition differs from base protocol: {}".format(key))

    jobs = list(base["jobs"]) + list(replication["jobs"])
    identities = [job_identity(job) for job in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError("Plans contain duplicate parent/seed/adaptation/route identities")
    missing = [job["id"] for job in jobs if not (Path(job["output"]) / "final.pt").is_file()]
    if missing:
        raise ValueError("Cannot merge incomplete jobs: {}".format(missing))

    combined = dict(base)
    combined["jobs"] = sorted(jobs, key=job_identity)
    combined["analysis_only"] = True
    combined["component_plans"] = [str(base_path), str(replication_path)]
    combined["selection"] = (
        "Completed seed-1 jobs from the fixed 2,500-update continuation plus prospectively declared "
        "fine-tuning seeds from the matched replication plan; analysis-only, do not launch."
    )

    if output.exists() and json.loads(output.read_text()) != combined:
        raise SystemExit("Combined plan already exists and differs; use a new output path")
    write_json(output, combined)
    parents = sorted({job["parent_id"] for job in jobs})
    seeds = sorted({job["seed"] for job in jobs})
    routes = sorted({job["route"] for job in jobs})
    print("Combined analysis plan:", output)
    print("Parents: {}; seeds: {}; routes: {}; jobs: {}".format(parents, seeds, routes, len(jobs)))


if __name__ == "__main__":
    main()
