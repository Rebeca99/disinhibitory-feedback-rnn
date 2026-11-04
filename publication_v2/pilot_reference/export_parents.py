"""Export compact, unchanged parent parameters with source provenance."""
import argparse
import json
from pathlib import Path
import torch
from core import checkpoint_state, digest_file, trusted_load, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-plan", required=True)
    p.add_argument("--output", default="parents")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.source_plan).read_text())
    rows = []
    for parent in plan["parents"]:
        saved = trusted_load(parent["path"])
        target = output / (parent["id"] + "_wcst_" + str(parent["source_training_seed"]) + ".pt")
        compact = dict(hp=saved["hp"], hp_task=saved.get("hp_task"),
                       model_state_dict={k: v.detach().cpu().clone() for k, v in checkpoint_state(saved).items()},
                       provenance=parent)
        torch.save(compact, target)
        rows.append(dict(filename=target.name, sha256=digest_file(target), source=parent))
    write_json(output / "provenance.json", rows)
    print("Exported unchanged parameter states:", len(rows), "to", output.resolve())


if __name__ == "__main__":
    main()
