"""Verified initialization from completed publication fine-tuning runs."""
import json
from pathlib import Path

from core import VERSION, code_digest, digest_file, trusted_load


SCHEMA = "publication-continuation-v1"
PILOT_CODE_SHA256 = "c424e037efce9c1618a42263765ff9e518680c216138b50e00bcaee2e6073db8"


def snapshot_digest(directory):
    import hashlib
    root = Path(directory).resolve()
    paths = sorted(list(root.glob("*.py")) + list((root / "legacy").glob("*.py")))
    if not paths:
        raise ValueError("No experiment Python files found in " + str(root))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_source_code(directory):
    source = Path(directory).resolve()
    current = Path(__file__).resolve().parent
    digest = snapshot_digest(source)
    if digest not in {PILOT_CODE_SHA256, code_digest()}:
        raise ValueError("Source code is not a verified pilot or current continuation snapshot. "
                         "Keep the original pilot code unchanged; do not bypass this check.")
    # The trainer adds initialization/provenance only; model and task math must match.
    for relative in ["core.py", "protocol.py"] + [str(p.relative_to(current))
                                                  for p in sorted((current / "legacy").glob("*.py"))]:
        if digest_file(source / relative) != digest_file(current / relative):
            raise ValueError("Model/task implementation differs from the source: " + relative)
    return digest


def fingerprint_for(job, code_sha256=None):
    result = {k: v for k, v in job.items() if k not in ["id", "output"]}
    result["version"] = VERSION
    result["code_sha256"] = code_sha256 or code_digest()
    return result


def validate_completed_source(job, saved, code_sha256):
    if saved.get("fingerprint") != fingerprint_for(job, code_sha256):
        raise ValueError("Source checkpoint does not match its plan/code: " + job["id"])
    if saved.get("termination_reason") != "equal_update_budget_completed":
        raise ValueError("Source is not a completed equal-budget run: " + job["id"])
    if saved.get("updates") != job["config"]["updates"] or saved.get("config") != job["config"]:
        raise ValueError("Source update count/configuration mismatch: " + job["id"])
    for key, expected in [("route", job["route"]), ("adaptation", job["adaptation"]),
                          ("seed", job["seed"]), ("parent_id", job["parent_id"])]:
        if saved.get(key) != expected:
            raise ValueError("Source metadata mismatch for " + key + ": " + job["id"])
    if "model_state_dict" not in saved or not saved.get("optimizer_state_dict", {}).get("state"):
        raise ValueError("Source lacks learned parameters or Adam state: " + job["id"])
    groups = saved["optimizer_state_dict"].get("param_groups", [])
    if not groups or any(group["lr"] != job["config"]["lr"] for group in groups):
        raise ValueError("Source optimizer learning rate differs from its plan: " + job["id"])
    updates = saved["updates"]
    if [record["update"] for record in saved.get("training_records", [])] != list(range(1, updates + 1)):
        raise ValueError("Source training history is incomplete: " + job["id"])
    validations = saved.get("validation_records", [])
    if not validations or validations[-1]["update"] != updates:
        raise ValueError("Source final validation/history is missing: " + job["id"])
    if updates % job["config"]["validation_every"]:
        raise ValueError("Continue at a scheduled validation boundary to preserve the validation history.")
    if digest_file(job["parent"]) != job["parent_sha256"]:
        raise ValueError("Source parent archive checksum changed: " + job["id"])


def load_continuation_source(job):
    origin = job["continuation"]
    if origin.get("schema") != SCHEMA:
        raise ValueError("Unknown continuation schema")
    if digest_file(origin["source_plan"]) != origin["source_plan_sha256"]:
        raise ValueError("Source plan changed after continuation preparation")
    source_plan = json.loads(Path(origin["source_plan"]).read_text())
    matches = [j for j in source_plan["jobs"] if j["id"] == origin["source_job_id"]]
    if len(matches) != 1:
        raise ValueError("Cannot identify a unique source job")
    source = matches[0]
    source_hash = verify_source_code(origin["source_code_directory"])
    if source_hash != origin["source_code_sha256"]:
        raise ValueError("Source code changed after continuation preparation")
    path = Path(source["output"]) / "final.pt"
    if str(path.resolve()) != origin["source_checkpoint"] or digest_file(path) != origin["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed after continuation preparation")
    expected_config = dict(source["config"], updates=job["config"]["updates"])
    if expected_config != job["config"]:
        raise ValueError("A continuation may change only the total update budget")
    for key in ["id", "parent", "parent_id", "parent_sha256", "seed", "route", "adaptation"]:
        if source[key] != job[key]:
            raise ValueError("Continuation changed " + key)
    saved = trusted_load(path)
    validate_completed_source(source, saved, source_hash)
    if saved["updates"] != origin["source_updates"] or saved["elapsed_seconds"] != origin["source_elapsed_seconds"]:
        raise ValueError("Continuation start metadata mismatch")
    if job["config"]["updates"] <= saved["updates"]:
        raise ValueError("Total update budget must exceed the completed source budget")
    return saved
