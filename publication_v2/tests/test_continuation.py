import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core import DEFAULTS, VERSION, digest_file, trusted_load, write_json
from continuation import PILOT_CODE_SHA256, load_continuation_source, verify_source_code
from functions import get_default_hp
from model_working import Net_readoutSR_working
from prepare_continuation import prepare
import train_publication


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "pilot_reference"


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        hp = get_default_hp()[0]
        hp.update(task="wcst", batch_size=2, train_rule=True, mglur=False,
                  dendrite_type="additive", dend_nonlinearity="subtractive",
                  activation="relu", pos_wout=True, pos_wout_rule=True)
        for module in ["sr", "pfc"]:
            hp["n_" + module + "_esoma"] = 5
            hp["n_" + module + "_edend"] = 10
        torch.manual_seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            model = Net_readoutSR_working(hp)
        cls.parent = dict(hp=hp, model_state_dict=model.state_dict())

    def source(self, temp, route="sr_vip", adaptation="full", legacy=False, jitter_ms=0):
        root = Path(temp)
        parent = root / "parent.pt"
        torch.save(self.parent, parent)
        cfg = dict(DEFAULTS, updates=2, batch_size=2, block_trials=2, jitter_ms=jitter_ms,
                   validation_every=2, validation_trials=4, validation_sequences=2, checkpoint_every=1)
        identity = "parent_01/{}/{}/seed_17".format(adaptation, route)
        job = dict(id=identity, parent=str(parent), parent_id="parent_01", parent_sha256=digest_file(parent),
                   seed=17, route=route, adaptation=adaptation, config=cfg,
                   output=str(root / "pilot" / identity))
        plan_path = root / "pilot" / "plan.json"
        write_json(plan_path, dict(version=VERSION, parents=[dict(id="parent_01", source_name="test")], jobs=[job]))
        if legacy:
            subprocess.run([sys.executable, str(REFERENCE / "train_publication.py"),
                            "--plan", str(plan_path), "--job-index", "0", "--device", "cpu"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                train_publication.train(job, "cpu")
        return job, plan_path, REFERENCE if legacy else ROOT

    def continuation(self, temp, path, code_directory, updates=4):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(path, code_directory, Path(temp) / "continued", updates)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_original_pilot_snapshot_is_verified(self):
        self.assertEqual(verify_source_code(REFERENCE), PILOT_CODE_SHA256)

    def test_old_pilot_continuation_equals_uninterrupted_training(self):
        for route, adaptation in [("baseline", "full"), ("sr_vip", "full"),
                                  ("sr_dend10", "full"), ("sr_vip", "feedback_readout")]:
            with self.subTest(route=route, adaptation=adaptation), tempfile.TemporaryDirectory() as temp:
                source, path, directory = self.source(temp, route, adaptation, legacy=True, jitter_ms=200)
                source_hash = digest_file(Path(source["output"]) / "final.pt")
                plan = self.continuation(temp, path, directory)
                continued = plan["jobs"][0]
                reference = copy.deepcopy(source)
                reference["config"]["updates"] = 4
                reference["output"] = str(Path(temp) / "uninterrupted")
                with contextlib.redirect_stdout(io.StringIO()):
                    train_publication.train(reference, "cpu")
                    train_publication.train(continued, "cpu")
                actual = trusted_load(Path(continued["output"]) / "final.pt")
                expected = trusted_load(Path(reference["output"]) / "final.pt")
                for key in ["model_state_dict", "optimizer_state_dict", "training_records", "validation_records"]:
                    self.assert_tree_equal(actual[key], expected[key])
                self.assertEqual(actual["updates"], 4)
                self.assertEqual(actual["continuation"]["source_updates"], 2)
                self.assertEqual(digest_file(Path(source["output"]) / "final.pt"), source_hash)
                audit = json.loads((Path(continued["output"]) / "audit.json").read_text())
                self.assertEqual(audit["continuation_updates"], 2)
                self.assertGreater(audit["continuation_elapsed_seconds"], 0)

    def test_continuation_interruption_and_resume_preserve_adam_and_history(self):
        with tempfile.TemporaryDirectory() as temp:
            source, path, directory = self.source(temp)
            plan = self.continuation(temp, path, directory)
            job = plan["jobs"][0]
            original = train_publication.run_sequence
            def interrupt(*args, **kwargs):
                result = original(*args, **kwargs)
                signal.raise_signal(signal.SIGTERM)
                return result
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(train_publication, "run_sequence", interrupt):
                    train_publication.train(job, "cpu")
                checkpoint = trusted_load(Path(job["output"]) / "latest.pt")
                self.assertEqual(checkpoint["updates"], 3)
                train_publication.train(job, "cpu")
                reference = copy.deepcopy(source)
                reference["config"]["updates"] = 4
                reference["output"] = str(Path(temp) / "uninterrupted")
                train_publication.train(reference, "cpu")
            actual = trusted_load(Path(job["output"]) / "final.pt")
            expected = trusted_load(Path(reference["output"]) / "final.pt")
            for key in ["model_state_dict", "optimizer_state_dict", "training_records", "validation_records"]:
                self.assert_tree_equal(actual[key], expected[key])

    def test_preparation_is_idempotent_and_all_jobs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            source, path, directory = self.source(temp)
            first = self.continuation(temp, path, directory)
            second = self.continuation(temp, path, directory)
            self.assertEqual(first, second)
            self.assertEqual(len(first["jobs"]), 1)
            self.assertEqual(first["jobs"][0]["seed"], source["seed"])
            self.assertEqual(first["jobs"][0]["route"], source["route"])
            with self.assertRaisesRegex(ValueError, "already differs"):
                self.continuation(temp, path, directory, updates=6)

    def test_invalid_destination_and_nonextended_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, path, directory = self.source(temp)
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare(path, directory, path.parent / "continued", 4)
            with self.assertRaisesRegex(ValueError, "must exceed"):
                self.continuation(temp, path, directory, updates=2)

    def test_changed_source_checkpoint_and_plan_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, path, directory = self.source(temp)
            job = self.continuation(temp, path, directory)["jobs"][0]
            checkpoint = Path(source["output"]) / "final.pt"
            with open(checkpoint, "ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                load_continuation_source(job)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "plan changed"):
                load_continuation_source(job)

    def test_changed_learning_rate_or_route_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, path, directory = self.source(temp)
            job = self.continuation(temp, path, directory)["jobs"][0]
            changed = copy.deepcopy(job)
            changed["config"]["lr"] *= 2
            with self.assertRaisesRegex(ValueError, "only the total update budget"):
                load_continuation_source(changed)
            changed = copy.deepcopy(job)
            changed["route"] = "pfc_vip"
            with self.assertRaisesRegex(ValueError, "changed route"):
                load_continuation_source(changed)

    def test_incomplete_source_and_missing_optimizer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, path, directory = self.source(temp)
            checkpoint = Path(source["output"]) / "final.pt"
            saved = trusted_load(checkpoint)
            saved["optimizer_state_dict"] = {}
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "Adam state"):
                self.continuation(temp, path, directory)
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "has not completed"):
                self.continuation(temp, path, directory)
            self.assertFalse((Path(temp) / "continued" / "plan.json").exists())


if __name__ == "__main__":
    unittest.main()
