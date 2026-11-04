import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core import (CORE_ROUTES, DEFAULTS, build_model, corrupt_outcome, decisions, feedback_audit,
                  hazard_rules, make_previous, make_trial, noise_for, trial_history, window_features)
from protocol import BayesianObserver, run_sequence, validation
from preflight import equivalence
from functions import get_default_hp
from model_working import Net_readoutSR_working


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        hp = get_default_hp()[0]
        hp.update(task="wcst", batch_size=2, train_rule=True, mglur=False,
                  dendrite_type="additive", dend_nonlinearity="subtractive", activation="relu", pos_wout=True, pos_wout_rule=True)
        for module in ["sr", "pfc"]:
            hp["n_" + module + "_esoma"] = 5
            hp["n_" + module + "_edend"] = 10
        with contextlib.redirect_stdout(io.StringIO()):
            model = Net_readoutSR_working(hp)
        cls.parent = dict(hp=hp, model_state_dict=model.state_dict())
        cls.cfg = dict(DEFAULTS, batch_size=2, jitter_ms=0)

    def model(self, route="sr_vip", **overrides):
        cfg = dict(self.cfg, **overrides)
        return build_model(self.parent, route, cfg)[0]

    def test_original_forward_pass_equivalence(self):
        errors = equivalence(self.parent)
        self.assertLess(max(errors.values()), 1e-4)

    def test_recorded_activity_matches_legacy_aliasing(self):
        model = self.model("baseline")
        model.rnn.network_noise = 0
        t = make_trial(0, 2, 10, np.random.RandomState(31), jitter_ms=0)
        t["x"] = t["x"][:8]
        h = trial_history(t, None, self.cfg, "cpu")
        state = torch.ones(2, model.rnn.total_n_neurons)
        with torch.no_grad():
            activity = model.simulate(t["x"], h, state)[3]
            _, legacy = super(type(model), model).forward(t["x"], dict(h=state, i_me=None),
                trial_history={"i_prev_stim": h["stimulus"], "i_prev_choice": h["choice"], "i_prev_rew": h["feedback"]})
        self.assertTrue(torch.allclose(activity, torch.stack(legacy["record"]["hiddens"]), atol=1e-5))

    def test_pure_additive_is_a_declared_different_model(self):
        legacy = self.model("baseline")
        pure = self.model("baseline", dynamics="pure_additive")
        t = make_trial(0, 2, 10, np.random.RandomState(31), jitter_ms=0)
        t["x"] = t["x"][:8]
        h = trial_history(t, None, self.cfg, "cpu")
        state = torch.ones(2, legacy.rnn.total_n_neurons)
        noise = torch.zeros(8, *state.shape)
        with torch.no_grad():
            a = legacy.simulate(t["x"], h, state, noise)[3]
            b = pure.simulate(t["x"], h, state, noise)[3]
        self.assertFalse(torch.allclose(a, b))

    def test_gradients_match_legacy_forward(self):
        new = self.model("baseline")
        legacy = self.model("baseline")
        new.rnn.network_noise = legacy.rnn.network_noise = 0
        torch.manual_seed(15)
        x = torch.rand(12, 2, 16)
        state = torch.ones(2, new.rnn.total_n_neurons)
        history = {k: torch.rand(12, 2, d) for k, d in [("feedback", 2), ("stimulus", 16), ("choice", 3)]}
        nc, nr, _, _ = new.simulate(x, history, state)
        (nc.square().mean() + nr.square().mean()).backward()
        out, _ = super(type(legacy), legacy).forward(x, dict(h=state.clone(), i_me=None),
            trial_history={"i_prev_rew": history["feedback"], "i_prev_stim": history["stimulus"], "i_prev_choice": history["choice"]})
        (out["out"].square().mean() + out["out_rule"].square().mean()).backward()
        for name, parameter in new.named_parameters():
            other = dict(legacy.named_parameters())[name]
            if parameter.grad is not None:
                self.assertTrue(torch.allclose(parameter.grad, other.grad, atol=2e-5, rtol=2e-4), name)

    def test_resume_matches_uninterrupted_training(self):
        import signal
        from unittest.mock import patch
        import train_publication
        from core import digest_file, trusted_load
        cfg = dict(self.cfg, updates=2, block_trials=2, validation_every=1,
                   validation_trials=4, validation_sequences=2, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "parent.pt"
            torch.save(self.parent, parent)
            job = dict(id="test", parent=str(parent), parent_id="parent_01", parent_sha256=digest_file(parent),
                       seed=17, route="sr_vip", adaptation="full", config=cfg, output=str(Path(temp) / "full"))
            with contextlib.redirect_stdout(io.StringIO()):
                train_publication.train(job, "cpu")
                resumed = dict(job, output=str(Path(temp) / "resumed"))
                original = train_publication.run_sequence
                def interrupt_once(*args, **kwargs):
                    result = original(*args, **kwargs)
                    signal.raise_signal(signal.SIGTERM)
                    return result
                with patch.object(train_publication, "run_sequence", interrupt_once):
                    train_publication.train(resumed, "cpu")
                self.assertEqual(trusted_load(Path(resumed["output"]) / "latest.pt")["updates"], 1)
                train_publication.train(resumed, "cpu")
            reference = trusted_load(Path(job["output"]) / "final.pt")
            result = trusted_load(Path(resumed["output"]) / "final.pt")
            for name, parameter in reference["model_state_dict"].items():
                self.assertTrue(torch.equal(parameter, result["model_state_dict"][name]), name)
            self.assertEqual(reference["training_records"], result["training_records"])

    def test_matched_routes(self):
        audits = {r: feedback_audit(self.model(r)) for r in CORE_ROUTES}
        for route in ["pfc_vip", "sr_vip", "sr_dend10", "sr_pv10", "sr_dend_pv10"]:
            self.assertEqual(audits[route]["added_n"], 10)
            self.assertAlmostEqual(audits[route]["added_l2"], np.sqrt(10) * .1, places=6)
        self.assertEqual(audits["both_vip"]["added_n"], 20)
        self.assertAlmostEqual(feedback_audit(self.model("both_vip_normmatched"))["added_l2"], audits["sr_vip"]["added_l2"], places=6)

    def test_positive_feedback_and_history_unchanged(self):
        baseline = self.model("baseline")
        for route in CORE_ROUTES[1:]:
            model = self.model(route)
            self.assertTrue(torch.equal(model.weights()["feedback"][0], baseline.weights()["feedback"][0]))
            self.assertTrue(torch.equal(model.rnn.mask_prev_stim, baseline.rnn.mask_prev_stim))
            self.assertTrue(torch.equal(model.rnn.mask_prev_choice, baseline.rnn.mask_prev_choice))
            self.assertEqual(int((model.rnn.mask_rew[0] != 0).sum()), 20)

    def test_fixed_backbone_is_fixed(self):
        model = build_model(self.parent, "sr_vip", self.cfg, adaptation="feedback_readout")[0]
        trained = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trained, {"rnn.w_rew", "rnn.w_out", "rnn.w_out_rule"})

    def test_hazard_bounds(self):
        rules, ages = hazard_rules(10000, [8, 22, .1], 4)
        switches = np.flatnonzero(rules[1:] != rules[:-1]) + 1
        lengths = np.diff(np.r_[0, switches])
        self.assertGreaterEqual(lengths.min(), 8)
        self.assertLessEqual(lengths.max(), 22)
        rules2, _ = hazard_rules(80, [8, 22, .1], 4)
        np.testing.assert_array_equal(rules[:80], rules2)

    def test_cards_do_not_reveal_rule(self):
        a = make_trial(0, 20, 10, np.random.RandomState(17))
        b = make_trial(1, 20, 10, np.random.RandomState(17))
        self.assertTrue(torch.equal(a["x"], b["x"]))
        self.assertTrue(torch.all(a["correct"] != b["correct"]))
        for example in range(20):
            self.assertEqual(int(np.all(a["candidates"][example] == a["center"][example], axis=1).sum()), 0)
            self.assertEqual(int((a["candidates"][example, :, 0] == a["center"][example, 0]).sum()), 1)
            self.assertEqual(int((a["candidates"][example, :, 1] == a["center"][example, 1]).sum()), 1)

    def test_unique_tie_decision_and_targets(self):
        trial = make_trial(0, 2, 10, np.random.RandomState(1), jitter_ms=0)
        c, r, ct, rt = decisions(torch.zeros(210, 2, 3), torch.zeros(210, 2, 2), trial)
        self.assertEqual(c.tolist(), [0, 0])
        self.assertTrue(ct.all() and rt.all())
        self.assertTrue(torch.all(trial["choice_target"][:160] == 0))
        self.assertTrue(torch.all(trial["choice_target"][160:].sum(-1) == 1))
        self.assertTrue(torch.all(trial["rule_target"].sum(-1) == 1))

    def test_corruption_events(self):
        correct = torch.tensor([True, False, True, False])
        draws = np.array([[.01, .1], [.01, .1], [.01, .9], [.8, .1]])
        observed, events = corrupt_outcome(correct, draws, self.cfg)
        self.assertEqual(observed.tolist(), [[0, 1], [1, 0], [0, 0], [0, 1]])
        self.assertEqual(events.tolist(), ["reversed", "reversed", "omitted", "veridical"])

    def test_noise_and_validation_do_not_change_task_stream(self):
        model = self.model()
        t = make_trial(0, 2, 10, np.random.RandomState(4), jitter_ms=0)
        a = noise_for(t, model.rnn.total_n_neurons, 17, "cpu")
        torch.randn(123)
        b = noise_for(t, model.rnn.total_n_neurons, 17, "cpu")
        self.assertTrue(torch.equal(a, b))
        with torch.no_grad():
            x = run_sequence(model, self.cfg, 9, 3, [1, 2, .3])
            validation(model, self.cfg, 91, 1, 2, [1, 2, .3])
            y = run_sequence(model, self.cfg, 9, 3, [1, 2, .3])
        self.assertEqual(x["rows"], y["rows"])

    def test_history_uses_previous_actual_response_window(self):
        t = make_trial(0, 2, 10, np.random.RandomState(2), training=True)
        p = make_previous(t, torch.tensor([0, 1]), torch.tensor([True, False]), np.ones((2, 2)), self.cfg)
        n = make_trial(1, 2, 10, np.random.RandomState(3), training=True)
        h = trial_history(n, p, self.cfg, "cpu")
        self.assertTrue(torch.allclose(h["stimulus"][0], .1 * t["x"][t["response"][0]:].mean(0)))
        self.assertTrue(torch.all(h["feedback"][n["history_stop"]:] == 0))
        self.assertFalse(h["choice"].requires_grad)

    def test_cross_trial_state_has_gradients(self):
        model = self.model("baseline", network_noise=0)
        t = make_trial(0, 2, 10, np.random.RandomState(3), jitter_ms=0)
        h = trial_history(t, None, self.cfg, "cpu")
        _, _, state, _ = model.simulate(t["x"], h)
        state.retain_grad()
        _, rule, _, _ = model.simulate(t["x"], h, state)
        rule.sum().backward()
        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(model.rnn.w_rec.grad).all())

    def test_edge_block_is_example_and_window_specific(self):
        model = self.model("sr_vip")
        r = model.rnn
        with torch.no_grad():
            r.w_rec.fill_(.1)
            r.w_fix.zero_()
        t = make_trial(0, 2, 10, np.random.RandomState(2), jitter_ms=0)
        t["x"] = t["x"][:4]
        h = trial_history(t, None, self.cfg, "cpu")
        initial = torch.ones(2, r.total_n_neurons)
        noise = torch.zeros(4, *initial.shape)
        spec = dict(kind="edge_block", src=r.cg_idx["sr_vip"].tolist(), dst=r.cg_idx["sr_sst"].tolist(),
                    start=1, stop=2, examples=torch.tensor([True, False]))
        mask = r.mask.clone()
        with torch.no_grad():
            intact = model.simulate(t["x"], h, initial, noise)[3]
            blocked = model.simulate(t["x"], h, initial, noise, spec)[3]
        self.assertTrue(torch.equal(intact[0], blocked[0]))
        self.assertTrue(torch.equal(intact[:, 1], blocked[:, 1]))
        self.assertGreater(float(blocked[1, 0, r.cg_idx["sr_sst"]].sum()), float(intact[1, 0, r.cg_idx["sr_sst"]].sum()))
        self.assertTrue(torch.equal(mask, r.mask))

    def test_motif_training_has_matched_baseline(self):
        for route in ["baseline_motif_blocked", "sr_vip_motif_blocked"]:
            model = self.model(route)
            r = model.rnn
            self.assertTrue(torch.all(r.mask[np.ix_(r.cg_idx["sr_vip"], r.cg_idx["sr_sst"])] == 0))

    def test_replay_preserves_intact_events(self):
        model = self.model()
        with torch.no_grad():
            intact = run_sequence(model, self.cfg, 34, 10, [2, 4, .3])
            replay = run_sequence(model, self.cfg, 34, 10, [2, 4, .3], intervention="ablate_added", replay=True)
        for left, right in zip(intact["rows"], replay["rows"]):
            for key in ["choice", "rule_readout", "previous_error", "corruption", "observed_feedback"]:
                self.assertEqual(left[key], right[key])
        self.assertTrue(torch.equal(intact["hidden"], replay["hidden"]))

    def test_ablation_current_audit_is_actual_current(self):
        model = self.model()
        with torch.no_grad():
            ablated = run_sequence(model, self.cfg, 34, 10, [2, 4, .3], intervention="ablate_added")
            replay = run_sequence(model, self.cfg, 34, 10, [2, 4, .3], intervention="ablate_added", replay=True)
        self.assertTrue(all(r["added_current_l2"] == 0 for r in ablated["rows"]))
        self.assertTrue(all(r["intervention_added_current_l2"] == 0 for r in replay["rows"]))
        self.assertTrue(any(r["added_current_l2"] > 0 for r in replay["rows"]))

    def test_window_features_keep_units_and_examples(self):
        trial = make_trial(0, 2, 10, np.random.RandomState(4), jitter_ms=0)
        activity = torch.arange(210, dtype=torch.float32)[:, None, None].expand(210, 2, 90)
        f = window_features(activity, trial)
        self.assertEqual(f["early"].shape, (2, 90))
        self.assertAlmostEqual(float(f["early"][0, 0]), 14.5)

    def test_observer_noiseless_informative_feedback(self):
        observer = BayesianObserver([8, 22, .1], dict(self.cfg, feedback_noise=0), 2)
        observer.advance(np.array([[0, 1], [2, 1]]), np.array([0, 2]), np.array([[1, 0], [1, 0]]))
        np.testing.assert_allclose(observer.current(), [[1, 0], [1, 0]])

    def test_observer_neither_choice_is_uninformative(self):
        observer = BayesianObserver([8, 22, .1], self.cfg, 1)
        observer.advance(np.array([[0, 1]]), np.array([2]), np.array([[0, 1]]))
        np.testing.assert_allclose(observer.current(), [[.5, .5]])

    def test_grouped_decoder(self):
        from decode_publication import decode
        rng = np.random.RandomState(1)
        groups = np.repeat(np.arange(6), 24)
        y = np.tile(np.repeat([0, 1], 12), 6)
        f = rng.randn(len(y), 4, 10).astype(np.float32)
        f[:, :, 0] += 3 * y[:, None]
        pops = dict(pfc_esoma=[0], pfc_edend=[1], pfc_pv=[2], pfc_sst=[3], pfc_vip=[4],
                    sr_esoma=[5], sr_edend=[6], sr_pv=[7], sr_sst=[8], sr_vip=[9])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.npz"
            np.savez(path, features=f, labels=y, groups=groups, block_groups=groups*2+y,
                     populations=json.dumps(pops), windows=["feedback", "early", "delay", "response"])
            rows, audit = decode(path, folds=3, pca_components=2, permutations=2)
        self.assertEqual(len(rows), 20)
        for fold in audit:
            self.assertFalse(set(fold["train_sequences"]) & set(fold["test_sequences"]))


if __name__ == "__main__":
    unittest.main()
