"""Publication experiments: inherited additive dynamics, explicit task streams."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/biornn_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))
import numpy as np
import torch
from model_working import Net_readoutSR_working

VERSION = "publication-v2.2"
CORE_ROUTES = ["baseline", "pfc_vip", "sr_vip", "both_vip",
               "sr_dend10", "sr_pv10", "sr_dend_pv10"]
EXTRA_ROUTES = ["both_vip_normmatched", "baseline_motif_blocked",
                "sr_vip_motif_blocked", "sr_direct_dense_normmatched"]
DEFAULTS = dict(updates=1000, batch_size=10, block_trials=30, lr=1e-4,
                dynamics="legacy_additive",
                clip_norm=1.0, pfc_gain=0.1, added_gain=5.0, added_raw_init=0.02,
                history_stim=0.1, history_choice=0.1, history_feedback=1.0,
                feedback_noise=0.1, flip_fraction=0.5, jitter_ms=200,
                train_hazard=[8, 22, 0.1], ood_hazard=[8, 35, 0.08],
                validation_every=50, validation_trials=100, validation_sequences=2,
                target_seed=20261008, checkpoint_every=50)


def trusted_load(path):
    # Parent archives include Python model objects: only load trusted archives.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    paths = sorted(list(root.glob("*.py")) + list((root / "legacy").glob("*.py")))
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def stream_seed(seed, *parts):
    payload = json.dumps([int(seed)] + list(parts), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError("Not JSON serializable: " + type(v).__name__)
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")
    tmp.replace(path)


def checkpoint_state(saved):
    if "model_state_dict" in saved:
        return saved["model_state_dict"]
    if "model" in saved:
        return saved["model"].state_dict()
    raise ValueError("Checkpoint lacks model_state_dict/model")


def architecture_signature(saved):
    hp, state = saved["hp"], checkpoint_state(saved)
    keys = ["cell_group_list", "n_input", "n_output", "n_output_rule",
            "n_branches", "dt", "tau", "activation", "dendrite_type", "dend_nonlinearity", "mglur",
            "trainable_dend2soma", "pos_wout", "pos_wout_rule", "exc_to_vip",
            "divide_sr_sst_vip", "sparse_srsst_to_sredend",
            "sparse_pfcesoma_to_srvip", "no_pfcesoma_to_srsst",
            "no_pfcesoma_to_sredend", "no_pfcesoma_to_srpv",
            "no_srsst_to_srvip", "structured_sr_sst_to_sr_edend",
            "structured_sr_sst_to_sr_edend_branch_specific", "sr_sst_high_bias"]
    keys += ["n_" + g for g in hp["cell_group_list"]]
    signature = {k: hp.get(k, False if k == "exc_to_vip" else None) for k in keys}
    signature["mask_connection_counts"] = {str(v): int((state["rnn.mask"] == v).sum())
                                            for v in [-1, 0, 1]}
    return json.dumps(signature, sort_keys=True)


def parameter_digest(saved):
    h = hashlib.sha256()
    for name, value in sorted(checkpoint_state(saved).items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(parent, route, cfg, adaptation="full", device="cpu",
                initialize_added=True):
    hp = dict(parent["hp"])
    hp.update(batch_size=cfg["batch_size"], fdbk_to_vip=False,
              prev_rew_to_pfc_default=True, prev_rew_to_pfc_vip=False,
              prev_rew_to_sr_vip=False, prev_rew_to_pfc_scale=1.0,
              prev_rew_to_vip_scale=1.0, timeit_print=False, learning_rate=cfg["lr"])
    if hp["dendrite_type"] != "additive" or hp["activation"] != "relu" or hp["mglur"]:
        raise ValueError("This version deliberately supports additive/ReLU/no-mGluR parents only")
    if hp.get("output_noise", 0) != 0:
        raise ValueError("Nonzero output noise requires an explicit paired output-noise stream")
    random.seed(cfg["target_seed"])
    np.random.seed(cfg["target_seed"])
    torch.manual_seed(cfg["target_seed"])
    with contextlib.redirect_stdout(io.StringIO()):
        model = PublicationNet(hp)
    model.load_state_dict(checkpoint_state(parent), strict=True)
    rnn = model.rnn
    inherited = torch.zeros_like(rnn.mask_rew)
    inherited[:, np.concatenate([rnn.cg_idx["pfc_edend"], rnn.cg_idx["pfc_pv"]])] = 1
    if not torch.equal((rnn.mask_rew != 0).float(), inherited):
        raise ValueError("Parent feedback mask is not the inherited PFC dendrite/PV mask")
    target_rng = np.random.RandomState(cfg["target_seed"])
    dend = np.sort(target_rng.choice(rnn.cg_idx["sr_edend"], 10, replace=False))
    pv = np.sort(target_rng.choice(rnn.cg_idx["sr_pv"], 10, replace=False))
    mixed_dend = np.sort(target_rng.choice(dend, 5, replace=False))
    mixed_pv = np.sort(target_rng.choice(pv, 5, replace=False))
    targets = {
        "baseline": [], "baseline_motif_blocked": [],
        "pfc_vip": rnn.cg_idx["pfc_vip"], "sr_vip": rnn.cg_idx["sr_vip"],
        "sr_vip_motif_blocked": rnn.cg_idx["sr_vip"],
        "both_vip": np.concatenate([rnn.cg_idx["pfc_vip"], rnn.cg_idx["sr_vip"]]),
        "both_vip_normmatched": np.concatenate([rnn.cg_idx["pfc_vip"], rnn.cg_idx["sr_vip"]]),
        "sr_dend10": dend, "sr_pv10": pv,
        "sr_dend_pv10": np.concatenate([mixed_dend, mixed_pv]),
        "sr_direct_dense_normmatched": np.concatenate([rnn.cg_idx["sr_edend"], rnn.cg_idx["sr_pv"]])}
    if route not in targets:
        raise ValueError("Unknown route: " + route)
    selected = np.asarray(targets[route], dtype=int)
    added_mask = torch.zeros_like(inherited)
    gain = cfg["added_gain"] / (np.sqrt(2) if route == "both_vip_normmatched" else 1)
    if route == "sr_direct_dense_normmatched":
        gain *= np.sqrt(10 / len(selected))
    added_mask[1, selected] = gain
    with torch.no_grad():
        rnn.mask_rew.copy_(cfg["pfc_gain"] * inherited + added_mask)
        if initialize_added:
            rnn.w_rew[1, selected] = cfg["added_raw_init"]
    model.route = route
    model.dynamics = cfg["dynamics"]
    if model.dynamics not in ["legacy_additive", "pure_additive"]:
        raise ValueError("Unknown dynamics mode")
    model.added_targets = selected.tolist()
    model.inherited_targets = torch.where(inherited[0] != 0)[0].tolist()
    model.train_motif_block = route.endswith("motif_blocked")
    # Frozen architecture/masks, unused h0 and zero bias are never optimized.
    allowed = {"rnn.w_rec", "rnn.w_in", "rnn.w_prev_stim", "rnn.w_prev_choice",
               "rnn.w_rew", "rnn.w_out", "rnn.w_out_rule"}
    if adaptation == "feedback_readout":
        allowed = {"rnn.w_rew", "rnn.w_out", "rnn.w_out_rule"}
    elif adaptation != "full":
        raise ValueError("Unknown adaptation mode")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    if model.train_motif_block:
        with torch.no_grad():
            rnn.mask[np.ix_(rnn.cg_idx["sr_vip"], rnn.cg_idx["sr_sst"])] = 0
    return model.to(device), hp


class PublicationNet(Net_readoutSR_working):
    def weights(self):
        r = self.rnn
        return dict(rec=r.w_rec.abs() * r.mask + r.dend2soma.abs() * r.w_fix,
                    sensory=r.w_in.abs() * r.mask_in,
                    stimulus=r.w_prev_stim.abs() * r.mask_prev_stim,
                    choice=r.w_prev_choice.abs() * r.mask_prev_choice,
                    feedback=r.w_rew.abs() * r.mask_rew,
                    read_choice=(r.w_out.abs() if self.pos_wout else r.w_out) * self.mask_out,
                    read_rule=(r.w_out_rule.abs() if self.pos_wout_rule else r.w_out_rule) * self.mask_out_rule)

    def simulate(self, x, history, hidden=None, noise=None, intervention=None):
        r = self.rnn
        w = self.weights()
        if hidden is None:
            hidden = x.new_zeros((x.shape[1], r.total_n_neurons))
        fw = w["feedback"]
        if intervention and intervention["kind"] == "route_ablation":
            keep = torch.ones_like(fw)
            keep[1, intervention["targets"]] = 0
            fw = fw * keep
        external = x @ w["sensory"] + (history["feedback"] @ fw + history["choice"] @ w["choice"]
                                       + history["stimulus"] @ w["stimulus"])
        states = []
        rec_exc, rec_inh = w["rec"][r.e_idx], w["rec"][r.i_idx]
        for t in range(x.shape[0]):
            active = intervention and intervention["kind"] in {"edge_block", "silence"}
            active = active and intervention["start"] <= t < intervention["stop"]
            # Legacy next_h aliases hidden after step zero. Its dendritic write
            # consequently changes the state used by the subsequent soma update
            # and the preceding recorded sample. Reproduce this functionally,
            # without mutating tensors retained by autograd.
            if t and self.dynamics == "legacy_additive":
                i_exc = hidden[:, r.e_idx] @ rec_exc + external[t]
                i_inh = hidden[:, r.i_idx] @ rec_inh
                dend = r.nonlinearity_dend(g_e=i_exc[:, r.dend_idx], g_i=i_inh[:, r.dend_idx].abs(),
                                          nonlinearity=r.dend_nonlinearity)
                replaced = hidden.clone()
                replaced[:, r.dend_idx] = dend
                hidden = replaced
                states[-1] = replaced
            current = hidden @ w["rec"] + external[t] + r.bias * r.mask_bias
            if active and intervention["kind"] == "edge_block":
                src, dst = intervention["src"], intervention["dst"]
                removed = hidden[:, src] @ w["rec"][src][:, dst]
                removed = removed * intervention["examples"].to(x.dtype)[:, None]
                correction = torch.zeros_like(current)
                correction[:, dst] = removed
                current = current - correction
            hidden_next = (1 - r.decay) * hidden + r.decay * torch.relu(current)
            if active and intervention["kind"] == "silence":
                keep = torch.ones_like(hidden_next)
                keep[:, intervention["targets"]] = 1 - intervention["examples"].to(x.dtype)[:, None]
                hidden_next = hidden_next * keep
            epsilon = torch.randn_like(hidden_next) if noise is None else noise[t]
            hidden = hidden_next + r.network_noise * epsilon
            states.append(hidden)
        activity = torch.stack(states)
        return activity @ w["read_choice"], activity @ w["read_rule"], hidden, activity


def hazard_rules(n, hazard, seed, initial_rule=0):
    lo, hi, prob = hazard
    if not (1 <= lo <= hi and 0 <= prob <= 1):
        raise ValueError("Invalid bounded hazard")
    rng = np.random.RandomState(seed)
    rules, ages = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    rule, age = initial_rule, 0
    for trial in range(n):
        if trial and (age >= hi or (age >= lo and rng.rand() < prob)):
            rule, age = 1 - rule, 0
        rules[trial], ages[trial] = rule, age
        age += 1
    return rules, ages


def make_trial(rule, batch_size, dt, rng, training=False, jitter_ms=200):
    # Cards do not depend on the hidden rule; only supervised targets do.
    center = rng.randint(0, 2, (batch_size, 2))
    candidates = np.empty((batch_size, 3, 2), dtype=int)
    rule_locations = np.empty((batch_size, 2), dtype=int)
    for b in range(batch_size):
        slots = rng.permutation(3)
        candidates[b, slots[0]] = [center[b, 0], 1 - center[b, 1]]
        candidates[b, slots[1]] = [1 - center[b, 0], center[b, 1]]
        candidates[b, slots[2]] = 1 - center[b]
        rule_locations[b] = slots[:2]
    jitter = int(rng.uniform(-jitter_ms, jitter_ms) // dt) * dt if jitter_ms else 0
    reference = 1100 + (jitter if training else 0)
    response = 1600 + jitter
    stop = response + 500
    steps = stop // dt
    x = torch.zeros(steps, batch_size, 16)
    for b in range(batch_size):
        for dimension in range(2):
            x[reference // dt:, b, dimension * 2 + center[b, dimension]] = 1
            for card in range(3):
                index = 4 + card * 4 + dimension * 2 + candidates[b, card, dimension]
                x[response // dt:, b, index] = 1
    correct = torch.as_tensor(rule_locations[:, rule], dtype=torch.long)
    yc = torch.zeros(steps, batch_size, 3)
    yc[response // dt:, torch.arange(batch_size), correct] = 1
    yr = torch.zeros(steps, batch_size, 2)
    yr[:, :, rule] = 1
    return dict(x=x, choice_target=yc, rule_target=yr, correct=correct,
                rule_locations=rule_locations, response=(response // dt, steps),
                reference=reference // dt, history_stop=100 // dt, dt=dt, rule=rule,
                jitter_ms=jitter, center=center, candidates=candidates)


def corrupt_outcome(correct, draws, cfg):
    truth = correct.detach().cpu().numpy().astype(bool)
    noisy = draws[:, 0] < cfg["feedback_noise"]
    flip = noisy & (draws[:, 1] < cfg["flip_fraction"])
    omit = noisy & ~flip
    observed_correct = np.logical_xor(truth, flip)
    observed = np.stack([observed_correct, ~observed_correct], axis=1).astype(np.float32)
    observed[omit] = 0
    event = np.where(omit, "omitted", np.where(flip, "reversed", "veridical"))
    return torch.from_numpy(observed), event


def trial_history(trial, previous, cfg, device):
    t, b = trial["x"].shape[:2]
    result = {k: torch.zeros(t, b, d, device=device)
              for k, d in [("stimulus", 16), ("choice", 3), ("feedback", 2)]}
    if previous is not None:
        for key, scale in [("stimulus", cfg["history_stim"]),
                           ("choice", cfg["history_choice"]),
                           ("feedback", cfg["history_feedback"])]:
            result[key][:trial["history_stop"]] = scale * previous[key].detach().to(device)
    return result


def decisions(choice_outputs, rule_outputs, trial):
    start, stop = trial["response"]
    cm, rm = choice_outputs[start:stop].mean(0), rule_outputs.mean(0)
    c, r = cm.argmax(-1), rm.argmax(-1)
    cties = (cm == cm.max(-1, keepdim=True).values).sum(-1) > 1
    rties = (rm == rm.max(-1, keepdim=True).values).sum(-1) > 1
    return c, r, cties, rties


def readout_loss(choice_outputs, rule_outputs, trial, device):
    lc = torch.mean((choice_outputs - trial["choice_target"].to(device)) ** 2)
    lr = torch.mean((rule_outputs - trial["rule_target"].to(device)) ** 2)
    return lc + lr, lc, lr


def make_previous(trial, choice, correct, draws, cfg):
    feedback, event = corrupt_outcome(correct, draws, cfg)
    start, stop = trial["response"]
    return dict(stimulus=trial["x"][start:stop].mean(0).detach(),
                choice=torch.nn.functional.one_hot(choice.detach().cpu(), 3).float(),
                feedback=feedback, event=event, true_correct=correct.detach().cpu())


def noise_for(trial, n_units, seed, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(trial["x"].shape[0], trial["x"].shape[1], n_units,
                       generator=gen).to(device)


def window_features(activity, trial):
    dt = trial["dt"]
    windows = {"feedback": (0, 100 // int(dt)), "early": (100 // int(dt), 200 // int(dt)),
               "delay": (100 // int(dt), trial["reference"]), "response": trial["response"]}
    return {name: activity[start:stop].mean(0).detach().cpu().numpy()
            for name, (start, stop) in windows.items()}


def feedback_audit(model):
    r = model.rnn
    effective = (r.w_rew.abs() * r.mask_rew).detach().cpu()
    rows = []
    for channel in range(2):
        for name in r.cell_group_list:
            idx = r.cg_idx[name]
            weights = effective[channel, idx]
            enabled = r.mask_rew.detach().cpu()[channel, idx] != 0
            values = weights[enabled]
            rows.append(dict(channel="positive" if channel == 0 else "negative", population=name,
                             n_enabled=int(enabled.sum()), l1=float(weights.abs().sum()),
                             l2=float(weights.norm()), mean_enabled=float(values.mean()) if len(values) else 0.0))
    added = effective[1, model.added_targets]
    active_parameters = sum(int((mask != 0).sum()) for weight, mask in [
        (r.w_rec, r.mask), (r.w_in, r.mask_in), (r.w_prev_stim, r.mask_prev_stim),
        (r.w_prev_choice, r.mask_prev_choice), (r.w_rew, r.mask_rew),
        (r.w_out, model.mask_out), (r.w_out_rule, model.mask_out_rule)] if weight.requires_grad)
    return dict(routes=rows, added_n=len(model.added_targets), added_l1=float(added.abs().sum()),
                added_l2=float(added.norm()), added_targets=model.added_targets,
                trainable_effective_weight_entries=active_parameters,
                trainable_tensor_entries=sum(p.numel() for p in model.parameters() if p.requires_grad))


def intervention_spec(model, name, trial, previous, device):
    if name == "intact":
        return None
    r = model.rnn
    if name.startswith("ablate_"):
        group = name[len("ablate_"):]
        targets = {"added": model.added_targets, "pfc_vip": r.cg_idx["pfc_vip"].tolist(),
                   "sr_vip": r.cg_idx["sr_vip"].tolist()}[group]
        return dict(kind="route_ablation", targets=targets)
    negative = (torch.zeros(trial["x"].shape[1], dtype=torch.bool, device=device) if previous is None
                else previous["feedback"][:, 1].to(device) > 0)
    windows = {"feedback": (0, trial["history_stop"]),
               "early": (trial["history_stop"], 2 * trial["history_stop"]),
               "both": (0, 2 * trial["history_stop"])}
    if name.startswith("block_sr_vip_sst_"):
        suffix = name[len("block_sr_vip_sst_"):]
        negative_only = suffix.endswith("_negative")
        window = suffix.removesuffix("_negative") if hasattr(str, "removesuffix") else suffix.replace("_negative", "")
        start, stop = windows[window]
        return dict(kind="edge_block", src=r.cg_idx["sr_vip"].tolist(), dst=r.cg_idx["sr_sst"].tolist(),
                    start=start, stop=stop, examples=negative if negative_only else torch.ones_like(negative))
    if name.startswith("silence_"):
        population, window = name[len("silence_"):].rsplit("_", 1)
        start, stop = windows[window]
        return dict(kind="silence", targets=r.cg_idx[population].tolist(), start=start, stop=stop,
                    examples=torch.ones_like(negative))
    raise ValueError("Unknown intervention: " + name)
