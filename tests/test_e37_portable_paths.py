"""E37: run files and the holdout lock record no absolute local path; the
cases file is identified by its name and sha256, outside paths come from the CLI."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from hfa import evalrun as E
from hfa import holdout

GATE_SHA = "9bc97c853fe50e1db154d6498e7db0d4751fcb1d72ac2299c5ea636db3be5813"


class E37PortablePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cases = os.path.join(self.tmp.name, "07_holdout_cases.jsonl")
        with open(self.cases, "w", encoding="utf-8") as f:
            f.write('{"id": "H01"}\n')
        self.snaps = os.path.join(self.tmp.name, "eval_snapshots")
        os.mkdir(self.snaps)

    def tearDown(self):
        self.tmp.cleanup()

    def _inputs(self, cases, snaps, live=False):
        return E.replay_inputs(cases, True, SimpleNamespace(live=live, snapshots=snaps))

    def test_a_outside_repo_names_and_hash_only(self):
        ri = self._inputs(self.cases, self.snaps)
        self.assertEqual(ri["cases"], "07_holdout_cases.jsonl")
        self.assertEqual(ri["snapshots"], "eval_snapshots")
        self.assertEqual(ri["cases_sha256"], holdout.file_sha256(self.cases))
        text = json.dumps(ri)
        self.assertNotIn(self.tmp.name, text)
        self.assertNotIn("/Users/", text)
        self.assertFalse(any(os.path.isabs(v) for v in ri.values() if isinstance(v, str)))

    def test_b_inside_repo_relative(self):
        ri = E.replay_inputs(E.DEFAULT_CASES, False, SimpleNamespace(live=True, snapshots=None))
        self.assertEqual(ri["cases"], "data/eval/cases_public.jsonl")
        self.assertIsNone(ri["snapshots"])

    def test_c_holdout_lock_has_no_local_path(self):
        lock = os.path.join(self.tmp.name, "HOLDOUT_RUN.json")
        holdout.start_run({"llm": "test", "n_cases": 1}, lock_path=lock)
        holdout.finish_run("COMPLETE", {"split": "holdout", "replay_inputs": self._inputs(self.cases, self.snaps)},
                           lock_path=lock)
        with open(lock, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("/Users/", text)
        self.assertNotIn(self.tmp.name, text)

    def _run(self, ri):
        return {"rows": [{"id": "H01", "raw_model_output": "{}"}], "summary": {"replay_inputs": ri}}

    def test_d_replay_needs_cli_paths_for_outside_files(self):
        with self.assertRaises(E.ReplayError) as cm:
            E.replay_rows(self._run(self._inputs(self.cases, self.snaps)))
        self.assertIn("--cases", str(cm.exception))

    def test_e_replay_refuses_other_cases_file(self):
        ri = self._inputs(self.cases, self.snaps)
        other = os.path.join(self.tmp.name, "other.jsonl")
        with open(other, "w", encoding="utf-8") as f:
            f.write('{"id": "H02"}\n')
        with self.assertRaises(E.ReplayError) as cm:
            E.replay_rows(self._run(ri), other, self.snaps)
        self.assertIn("cases_sha256", str(cm.exception))

    def test_e2_aborted_error_is_redacted(self):
        # E37b: the ABORTED lock stores the exception text through redact()
        lock = os.path.join(self.tmp.name, "HOLDOUT_RUN.json")
        holdout.start_run({"llm": "test", "n_cases": 1}, lock_path=lock)
        err = FileNotFoundError(2, "No such file or directory", "/Users/someone/ai-collab/002-cash/prep/x.txt")
        holdout.finish_run("ABORTED", error=f"{type(err).__name__}: {err}", lock_path=lock)
        with open(lock, encoding="utf-8") as f:
            rec = json.load(f)
        self.assertEqual(rec["state"], "ABORTED")
        self.assertIn("[PATH]", rec["error"])
        self.assertNotIn("/Users/", json.dumps(rec))

    def test_f_gate_hash_unchanged(self):
        self.assertEqual(holdout.decision_code_sha256(), GATE_SHA)


if __name__ == "__main__":
    unittest.main()
