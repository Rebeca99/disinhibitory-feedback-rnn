"""Check inherited forward-pass equivalence, route matching, and baseline metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from core import DEFAULTS, CORE_ROUTES, build_model, feedback_audit, make_trial, trusted_load, write_json
from protocol import validation


def equivalence(parent, device="cpu"):
    cfg = dict(DEFAULTS, pfc_gain=1.0, history_stim=1.0, history_choice=1.0)
    model, _ = build_model(parent, "baseline", cfg, device=device)
    model.rnn.network_noise = 0
    trial = make_trial(0, cfg["batch_size"], model.rnn.dt, np.random.RandomState(11), jitter_ms=0)
    x = trial["x"].to(device)
    history = {k: torch.rand(x.shape[0], x.shape[1], d, device=device)
               for k, d in [("feedback", 2), ("stimulus", 16), ("choice", 3)]}
    h = torch.rand(cfg["batch_size"], model.rnn.total_n_neurons, device=device)
    with torch.no_grad():
        out_new = model.simulate(x, history, h, torch.zeros(x.shape[0], *h.shape, device=device))
        out_legacy, data = super(type(model), model).forward(x, dict(h=h, i_me=None),
            trial_history={"i_prev_rew": history["feedback"], "i_prev_stim": history["stimulus"],
                           "i_prev_choice": history["choice"]})
    errors = dict(choice=float((out_new[0] - out_legacy["out"]).abs().max()),
                  rule=float((out_new[1] - out_legacy["out_rule"]).abs().max()),
                  hidden=float((out_new[2] - data["last_states"]["hidden"]).abs().max()))
    if max(errors.values()) > 1e-4:
        raise AssertionError("Forward pass differs from inherited additive implementation: " + str(errors))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trials", type=int, default=100)
    args = p.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(Path(args.plan).read_text())
    result = []
    for parent_info in plan["parents"]:
        parent = trusted_load(parent_info["path"])
        forward_error = equivalence(parent, args.device)
        cfg = dict(plan["jobs"][0]["config"])
        routes = {route: feedback_audit(build_model(parent, route, cfg)[0]) for route in CORE_ROUTES}
        single = ["pfc_vip", "sr_vip", "sr_dend10", "sr_pv10", "sr_dend_pv10"]
        assert all(routes[r]["added_n"] == 10 for r in single)
        assert np.allclose([routes[r]["added_l2"] for r in single], routes["sr_vip"]["added_l2"])
        assert routes["both_vip"]["added_n"] == 20
        metrics = {}
        for name, overrides in [
            ("chapter6_bottleneck_before_finetuning", {}),
            ("parent_like_informative_history", dict(pfc_gain=1.0, history_stim=1.0, history_choice=1.0,
                                                   feedback_noise=0.0))]:
            test_cfg = dict(cfg, **overrides)
            model, _ = build_model(parent, "baseline", test_cfg, device=args.device)
            evaluation = validation(model, test_cfg, 91010, 2, args.trials, cfg["train_hazard"])
            rows = evaluation["rows"]
            metrics[name] = dict(choice_accuracy=float(np.mean([r["choice_accuracy"] for r in rows])),
                                 rule_accuracy=float(np.mean([r["rule_accuracy"] for r in rows])),
                                 legacy_choice_accuracy=float(np.mean([r["legacy_choice_accuracy"] for r in rows])),
                                 legacy_rule_accuracy=float(np.mean([r["legacy_rule_accuracy"] for r in rows])),
                                 choice_tie_rate=float(np.mean([r["choice_tie"] for r in rows])),
                                 rule_tie_rate=float(np.mean([r["rule_tie"] for r in rows])), loss=evaluation["losses"][0])
        result.append(dict(parent=parent_info["id"], source=parent_info["path"],
                           forward_pass_max_abs_error=forward_error, routes=routes, initial_metrics=metrics))
        print(parent_info["id"], json.dumps(metrics), flush=True)
    path = Path(args.plan).with_suffix(".preflight.json")
    write_json(path, result)
    print("PASS: inherited dynamics and initial single-route norms/counts verified.")
    print("Inspect baseline accuracy and tie rates in", path)
    print("Low bottleneck accuracy is a diagnostic result, not automatically an implementation failure.")


if __name__ == "__main__":
    main()
