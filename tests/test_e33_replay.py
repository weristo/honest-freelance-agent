"""E33: raw model output + parsed claims + kill class in the eval row and the
demo trace; `hfa eval --replay` re-runs them model-free. No model call."""
import contextlib
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from hfa import demo, evalrun
from hfa.proposer import base as pbase
from hfa.proposer import providers
from hfa.proposer.budget import Budget
from hfa.proposer.prompt import prompt_sha256

from .helpers import FILLER, RULE, StaticFetcher, page
from .test_proposer_budget_eval import FakeReal
from .test_round5 import ScriptedReal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://example.org/a"
OTHER = "Sellers must ship every order within three business days of payment."


def _boom(*_a, **_k):
    raise AssertionError("an LLM client was created or called during --replay")


@contextlib.contextmanager
def no_llm():
    """Every way to get or call a model raises."""
    with mock.patch.object(evalrun, "client_from_env", _boom), \
            mock.patch.object(providers, "client_from_env", _boom), \
            mock.patch.object(evalrun.Proposer, "__init__", _boom), \
            mock.patch.object(providers.MockClient, "__init__", _boom), \
            mock.patch.object(providers.AnthropicClient, "complete", _boom), \
            mock.patch.object(providers.OpenAICompatClient, "complete", _boom):
        yield


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        os.makedirs("snaps")
        with open("snaps/A.txt", "w") as f:
            f.write(f"# url={URL}\n# final_url={URL}\n# http=200 ctype=text/html\n"
                    f"# fetched_at=2026-10-04T15:00:00Z\n\n{FILLER} {RULE} {FILLER} {OTHER} {FILLER}\n")
        claim = lambda q, role: {"quote": q, "role": role, "url": URL, "snapshot": "A.txt"}
        self.cases = [
            {"id": "X1", "gold": "GO", "naive_false_go_trap": False, "input": "case one input", "split": "dev",
             "claims": [claim(RULE, "SUPPORTS")]},
            {"id": "X2", "gold": "KILL", "naive_false_go_trap": True, "input": "case two input", "split": "dev",
             "claims": [claim(RULE, "CONTRADICTS")]},
            {"id": "X3", "gold": "NEEDS_HUMAN", "naive_false_go_trap": False, "input": "case three input",
             "split": "dev", "claims": [claim(OTHER, "SUPPORTS")]},
            {"id": "X4", "gold": "UNVERIFIED", "naive_false_go_trap": True, "input": "case four input",
             "split": "dev", "claims": [claim("A rule sentence that is not on the page at all.", "SUPPORTS")]}]
        with open("cases.jsonl", "w") as f:
            f.write("\n".join(json.dumps(c) for c in self.cases) + "\n")

    def _mock_run(self) -> str:
        self.assertEqual(evalrun.main(["--mock", "--cases", "cases.jsonl", "--snapshots", "snaps",
                                       "--out", "runs"]), 0)
        return "runs/mock/eval_mock.json"

    def _real_run(self) -> str:
        with mock.patch.object(evalrun, "client_from_env", lambda: ScriptedReal(self.cases)):
            self.assertEqual(evalrun.main(["--cases", "cases.jsonl", "--snapshots", "snaps",
                                           "--out", "runs"]), 0)
        return "runs/real/eval_anthropic_claude-haiku-4-5.json"

    def _replay(self, path, *extra):
        # E37: the run records names only; files outside the repo come from the CLI
        extra = extra or ("--cases", "cases.jsonl", "--snapshots", "snaps")
        with no_llm():
            return evalrun.main(["--replay", path, *extra])

    def test_rows_store_raw_output_parsed_claims_and_kill_class(self):
        with open(self._mock_run()) as f:
            run = json.load(f)
        for r in run["rows"]:
            self.assertIn('"decision"', r["raw_model_output"])
            self.assertIn("kill_class", r)
            self.assertIn("proposed_kill_class", r)
            for c in r["claims_parsed"]:
                self.assertEqual(set(c), {"claim_id", "kind", "critical", "polarity", "quote", "source_url",
                                          "derived_from", "calc"})
        x2 = next(r for r in run["rows"] if r["id"] == "X2")
        self.assertEqual(x2["claims_parsed"][0]["quote"], RULE)          # full quote, not truncated
        self.assertEqual(x2["claims_parsed"][0]["polarity"], "CONTRADICTS")
        self.assertEqual(run["summary"]["replay_inputs"]["snapshots"], "snaps")           # E37: name only

    def test_replay_of_mock_run_gives_the_same_decision_for_every_case(self):
        path = self._mock_run()
        with open(path) as f:
            before = f.read()
        self.assertEqual(self._replay(path), 0)
        with open(path) as f:
            self.assertEqual(f.read(), before)                # replay writes nothing
        with open(path) as f:
            run = json.load(f)
        with no_llm():
            rows, diffs, _ = evalrun.replay_rows(run, "cases.jsonl", "snaps")
        self.assertEqual(diffs, [])
        self.assertEqual([(r["id"], r["verifier"], r["kill_class"]) for r in rows],
                         [(r["id"], r["verifier"], r["kill_class"]) for r in run["rows"]])
        self.assertEqual(len(rows), len(self.cases))

    def test_replay_of_real_labelled_run_uses_source_text_and_matches(self):
        path = self._real_run()
        self.assertEqual(self._replay(path), 0)

    def test_tampered_polarity_is_reported_with_nonzero_exit(self):
        path = self._mock_run()
        with open(path) as f:
            run = json.load(f)
        row = next(r for r in run["rows"] if r["id"] == "X1")
        prop = json.loads(row["raw_model_output"])
        prop["claims"][0]["polarity"] = "CONTRADICTS"            # was SUPPORTS
        row["raw_model_output"] = json.dumps(prop)
        with open("tampered.json", "w") as f:
            json.dump(run, f)
        with no_llm():
            _, diffs, _ = evalrun.replay_rows(run, "cases.jsonl", "snaps")
        self.assertTrue(diffs)
        self.assertTrue(all(d.startswith("X1:") for d in diffs))
        self.assertTrue(any("claims_parsed" in d and "CONTRADICTS" in d for d in diffs))
        self.assertTrue(any(d.startswith("X1: verifier:") for d in diffs))
        self.assertEqual(self._replay("tampered.json"), 1)

    def test_pre_e33_run_stops_with_a_clear_message(self):
        path = self._mock_run()
        with open(path) as f:
            run = json.load(f)
        for r in run["rows"]:
            for k in ("raw_model_output", "claims_parsed", "kill_class", "proposed_kill_class"):
                r.pop(k, None)
        run["summary"].pop("replay_inputs")
        with open("old.json", "w") as f:
            json.dump(run, f)
        with no_llm(), self.assertRaises(evalrun.ReplayError) as e:
            evalrun.replay_rows(run)
        self.assertEqual(str(e.exception), "no raw model output stored in this run (pre-E33)")
        self.assertEqual(self._replay("old.json"), 2)

    def test_real_pre_e33_files_if_present(self):
        for name in ("dev_1028", "dev_1028_p5"):
            p = os.path.join(HERE, "data", "eval", "runs", "real", name, "real",
                             "eval_anthropic_claude-haiku-4-5.json")
            if not os.path.exists(p):
                continue
            with open(p) as f, no_llm(), self.assertRaises(evalrun.ReplayError) as e:
                evalrun.replay_rows(json.load(f))
            self.assertIn("pre-E33", str(e.exception))

    def test_live_run_is_not_replayable(self):
        path = self._mock_run()
        with open(path) as f:
            run = json.load(f)
        run["summary"]["replay_inputs"]["live"] = True
        with no_llm(), self.assertRaises(evalrun.ReplayError):
            evalrun.replay_rows(run)

    def test_no_llm_guard_really_fires(self):
        with no_llm(), self.assertRaises(AssertionError):
            providers.client_from_env()
        with no_llm(), self.assertRaises(AssertionError):
            evalrun.main(["--mock", "--cases", "cases.jsonl", "--snapshots", "snaps", "--out", "runs"])


class DemoTraceTests(unittest.TestCase):
    def test_real_demo_trace_has_raw_output_claims_and_kill_class(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        with mock.patch.dict(os.environ, {"HFA_LLM_PROVIDER": "anthropic"}), \
                mock.patch.object(demo, "client_from_env", lambda: FakeReal()), \
                mock.patch.object(demo, "LiveFetcher", lambda: StaticFetcher(page(RULE))), \
                mock.patch.object(demo, "Budget", lambda: Budget(os.path.join(tmp.name, "b.json"))):
            demo.run_real(invoice=False)
        with open(os.path.join("data", "traces", "demo.jsonl")) as f:
            steps = [json.loads(line) for line in f]
        raw = [s for s in steps if s.get("actor") == "proposer_raw"]
        self.assertEqual(len(raw), 1)
        self.assertIn('"decision"', raw[0]["raw_model_output"])
        self.assertIn("kill_class", raw[0])
        self.assertTrue(raw[0]["claims_parsed"])
        self.assertEqual(raw[0]["claims_parsed"][0]["quote"], RULE)


class E33InvariantTests(unittest.TestCase):
    def test_prompt_hash_unchanged(self):
        # E34: p6 replaced p5 (920ca988b05c); E35: p7 replaced p6 (0445563f7cee); E36: p8 replaced p7 (1569d5ed153d).
        self.assertTrue(prompt_sha256().startswith("de5c1191515c"))

    def test_raw_output_paths_are_git_ignored(self):
        try:
            out = subprocess.run(["git", "-C", HERE, "check-ignore", "data/eval/runs/real/x.json",
                                  "data/eval/runs/mock/eval_mock.json", "data/traces/demo.jsonl"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest("git not available")
        if out.returncode == 128:
            self.skipTest("not a git checkout")
        self.assertEqual(out.returncode, 0)
        self.assertEqual(len(out.stdout.split()), 3)

    def test_propose_and_replay_share_the_model_free_code(self):
        self.assertIs(evalrun.proposal_from_text, pbase.proposal_from_text)


if __name__ == "__main__":
    unittest.main()
