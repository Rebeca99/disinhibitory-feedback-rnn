"""Paired training/evaluation sequences, event labels, and common-state replay."""
import numpy as np
import torch
from core import (decisions, hazard_rules, intervention_spec, make_previous, make_trial,
                  noise_for, readout_loss, stream_seed, trial_history, window_features)


class BayesianObserver:
    """Bounded-hazard filter conditioned on known cards, actual choice and feedback.

    This is an information-access benchmark, not a competing choice policy: it sees
    the unattenuated cards and choice and knows the hazard/corruption parameters.
    """
    def __init__(self, hazard, cfg, batch_size):
        self.lo, self.hi, self.prob = hazard
        self.cfg = cfg
        self.belief = np.zeros((batch_size, 2, self.hi), dtype=float)
        self.belief[:, :, 0] = 0.5

    def current(self):
        return self.belief.sum(-1)

    def advance(self, rule_locations, choice, observed):
        match = rule_locations == np.asarray(choice)[:, None]
        noise, flip = self.cfg["feedback_noise"], self.cfg["flip_fraction"]
        p_flip, p_omit = noise * flip, noise * (1 - flip)
        positive = observed[:, 0] > 0
        negative = observed[:, 1] > 0
        like = np.where(match, 1 - noise, p_flip)
        like = np.where(negative[:, None], np.where(match, p_flip, 1 - noise), like)
        like = np.where((~positive & ~negative)[:, None], p_omit, like)
        posterior = self.belief * like[:, :, None]
        total = posterior.sum((1, 2), keepdims=True)
        # Zero-probability observations mean the configured model is inconsistent.
        if (total <= 0).any():
            raise ValueError("Feedback has zero likelihood under observer model")
        posterior /= total
        new = np.zeros_like(posterior)
        for age in range(self.hi):
            length = age + 1
            h = 0 if length < self.lo else (1 if length >= self.hi else self.prob)
            new[:, :, 0] += h * posterior[:, ::-1, age]
            if length < self.hi:
                new[:, :, age + 1] += (1 - h) * posterior[:, :, age]
        self.belief = new


def run_sequence(model, cfg, seed, n_trials, hazard, training=False, intervention="intact",
                 replay=False, collect_features=False, sequence_id=0, initial_rule=0):
    device = next(model.parameters()).device
    rules, ages = hazard_rules(n_trials, hazard, stream_seed(seed, "hazard"), initial_rule)
    card_rng = np.random.RandomState(stream_seed(seed, "cards"))
    feedback_rng = np.random.RandomState(stream_seed(seed, "feedback"))
    hidden, previous = None, None
    rows, feature_list, label_list = [], [], []
    losses, choice_losses, rule_losses, training_metrics = [], [], [], []
    observer = BayesianObserver(hazard, cfg, cfg["batch_size"]) if not training else None
    for n in range(n_trials):
        trial = make_trial(int(rules[n]), cfg["batch_size"], model.rnn.dt, card_rng,
                           training=training, jitter_ms=cfg["jitter_ms"])
        history = trial_history(trial, previous, cfg, device)
        noise = noise_for(trial, model.rnn.total_n_neurons, stream_seed(seed, "noise", n), device)
        spec = intervention_spec(model, intervention, trial, previous, device)
        choice_out, rule_out, next_hidden, activity = model.simulate(
            trial["x"].to(device), history, hidden, noise, None if replay else spec)
        choice, read_rule, cties, rties = decisions(choice_out, rule_out, trial)
        choice_correct = choice == trial["correct"].to(device)
        rule_correct = read_rule == rules[n]
        training_metrics.append([float(choice_correct.float().mean()), float(rule_correct.float().mean()),
                                 float(cties.float().mean()), float(rties.float().mean())])
        loss, lc, lr = readout_loss(choice_out, rule_out, trial, device)
        losses.append(loss)
        choice_losses.append(lc)
        rule_losses.append(lr)
        if replay and intervention != "intact":
            # Fork one trial from the intact state/history; resume the intact run.
            pc, pr, _, _ = model.simulate(trial["x"].to(device), history, hidden, noise, spec)
            c_alt, r_alt, _, _ = decisions(pc, pr, trial)
            c_alt_correct = c_alt == trial["correct"].to(device)
            r_alt_correct = r_alt == rules[n]
        else:
            c_alt_correct, r_alt_correct = choice_correct, rule_correct
        windows = window_features(activity, trial) if collect_features else {}
        if collect_features:
            feature_list.append(np.stack([windows[k] for k in ["feedback", "early", "delay", "response"]], 1))
            label_list.extend([rules[n]] * cfg["batch_size"])
        if not training:
            beliefs = observer.current()
            feedback = history["feedback"][0].detach().cpu().numpy()
            w = model.weights()["feedback"].detach().cpu().numpy()
            intervened_w = w.copy()
            if spec and spec["kind"] == "route_ablation":
                intervened_w[1, spec["targets"]] = 0
            intervened_currents = feedback @ intervened_w
            currents = feedback @ w if replay else intervened_currents
            for b in range(cfg["batch_size"]):
                prev_correct = None if previous is None else bool(previous["true_correct"][b])
                observed = "none_first" if previous is None else (
                    "negative" if previous["feedback"][b, 1] > 0 else
                    "positive" if previous["feedback"][b, 0] > 0 else "omitted")
                row = dict(sequence=sequence_id, trial=n, example=b, initial_rule=initial_rule,
                           rule=int(rules[n]), age=int(ages[n]), true_switch=bool(n and rules[n] != rules[n-1]),
                           nonfirst=bool(n), stable=bool(ages[n] >= 5),
                           previous_error=bool(previous is not None and not prev_correct),
                           previous_correct=prev_correct, observed_feedback=observed,
                           corruption="none_first" if previous is None else str(previous["event"][b]),
                           choice=int(choice[b]), correct_choice=int(trial["correct"][b]),
                           rule_readout=int(read_rule[b]), choice_accuracy=float(choice_correct[b]),
                           rule_accuracy=float(rule_correct[b]),
                           legacy_choice_accuracy=float(choice_correct[b] & ~cties[b]),
                           legacy_rule_accuracy=float(rule_correct[b] & ~rties[b]),
                           choice_tie=bool(cties[b]), rule_tie=bool(rties[b]),
                           delta_choice=float(c_alt_correct[b]) - float(choice_correct[b]) if replay else 0.0,
                           delta_rule=float(r_alt_correct[b]) - float(rule_correct[b]) if replay else 0.0,
                           chose_old_rule=bool(n and ages[n] < hazard[1] and
                               int(choice[b]) == trial["rule_locations"][b, 1-int(rules[n])]),
                           chose_neither=bool(int(choice[b]) not in trial["rule_locations"][b]),
                           response_start_ms=trial["response"][0] * model.rnn.dt,
                           reference_start_ms=trial["reference"] * model.rnn.dt,
                           added_current_l2=float(np.linalg.norm(currents[b, model.added_targets])),
                           intervention_added_current_l2=float(np.linalg.norm(intervened_currents[b, model.added_targets])),
                           inherited_current_l2=float(np.linalg.norm(currents[b, model.inherited_targets])),
                           bayes_rule_accuracy=float(np.argmax(beliefs[b]) == rules[n]),
                           bayes_true_rule_probability=float(beliefs[b, rules[n]]),
                           bayes_entropy=float(-np.sum(beliefs[b] * np.log(beliefs[b] + 1e-15))))
                for window, values in windows.items():
                    for population in model.rnn.cell_group_list:
                        row["activity_" + window + "_" + population] = float(values[b, model.rnn.cg_idx[population]].mean())
                rows.append(row)
        draws = feedback_rng.rand(cfg["batch_size"], 2)
        next_previous = make_previous(trial, choice, choice_correct, draws, cfg)
        if observer is not None:
            observer.advance(trial["rule_locations"], choice.detach().cpu().numpy(),
                             next_previous["feedback"].numpy())
        hidden, previous = next_hidden, next_previous
    features = np.concatenate(feature_list, 0) if collect_features else None
    return dict(loss=torch.stack(losses).mean(), choice_loss=torch.stack(choice_losses).mean(),
                rule_loss=torch.stack(rule_losses).mean(), rows=rows, features=features,
                labels=np.asarray(label_list), hidden=hidden, metrics=np.mean(training_metrics, 0).tolist())


def validation(model, cfg, seed, n_sequences, n_trials, hazard, intervention="intact",
               replay=False, collect_features=False):
    rows, features, labels, losses = [], [], [], []
    model.eval()
    with torch.no_grad():
        for sequence in range(n_sequences):
            out = run_sequence(model, cfg, stream_seed(seed, "sequence", sequence), n_trials,
                               hazard, intervention=intervention, replay=replay,
                               collect_features=collect_features, sequence_id=sequence,
                               initial_rule=sequence % 2)
            rows.extend(out["rows"])
            if collect_features:
                features.append(out["features"])
                labels.append(out["labels"])
            losses.append([float(out[k]) for k in ["loss", "choice_loss", "rule_loss"]])
    result = dict(rows=rows, losses=np.mean(losses, 0).tolist())
    if collect_features:
        result.update(features=np.concatenate(features), labels=np.concatenate(labels))
    return result
