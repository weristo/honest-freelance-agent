"""E29: user tick only for numbers from the user's own input; holdout
started-lock + budget pre-check."""
import json
import os
import tempfile
import unittest
from unittest import mock

from hfa import demo, evalrun, holdout
from hfa.pipeline import run
from hfa.proposer import MockClient, Proposer
from hfa.proposer.usernums import USER_NUMBER_NOT_IN_INPUT, missing_numbers

from .test_round5 import ScriptedReal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(HERE, "data", "eval", "holdout_public.jsonl")


def _job(user_text: str, cost_literal: str, request_cost: str = "26", user_confirm: bool = True):
    """Request says the user's cost is `request_cost`; the (scripted) model
    writes `user_text` as the user's statement and computes with `cost_literal`."""
    request = demo.job_request("30", request_cost, "example")
    proposal = {"decision": "GO", "rationale": "x", "claims": [
        demo._claim("c2", demo.FEE_QUOTE), demo._user("u1", user_text),
        demo._net_calc("k1", "30", cost_literal, "27.60")]}
    p = Proposer(MockClient(lambda s, u: json.dumps(proposal)), user_confirm=user_confirm)
    res = run(request, [demo.DEMO_URL], p, demo._DemoFixture())
    return res, p


class UserNumberTickTests(unittest.TestCase):
    def test_invented_user_number_is_not_ticked_and_no_go(self):
        res, p = _job("The job pays 30 USD and my own cost is 20 USD.", "20")
        self.assertNotEqual(res.decision.final.value, "GO")
        self.assertEqual(res.decision.final.value, "NEEDS_HUMAN")
        u1 = next(c for c in res.decision.claims if c.claim_id == "u1")
        self.assertFalse(u1.user_confirmed)
        self.assertIn(USER_NUMBER_NOT_IN_INPUT, p.last_user_unticked["u1"])
        self.assertIn("20 USD", p.last_user_unticked["u1"])
        self.assertTrue(any(g.startswith("G5 fail") for g in res.decision.gate_log))

    def test_user_number_from_input_is_ticked(self):
        res, p = _job("The job pays 30 USD and my own cost is 26 USD.", "26")
        u1 = next(c for c in res.decision.claims if c.claim_id == "u1")
        self.assertTrue(u1.user_confirmed)
        self.assertEqual(p.last_user_unticked, {})
        self.assertEqual(res.decision.final.value, "GO")             # 27.60 >= 26

    def test_no_user_tick_still_means_no_tick(self):
        res, p = _job("The job pays 30 USD and my own cost is 26 USD.", "26", user_confirm=False)
        self.assertEqual(res.decision.final.value, "NEEDS_HUMAN")
        self.assertEqual(p.last_user_unticked, {})                   # no tick asked, nothing withheld

    def test_card_and_ui_show_reason(self):
        res, p = _job("my own cost is 20 USD", "20")
        c = demo.card(res, p.last_user_unticked)
        r = next(x for x in c["receipts"] if x["claim"] == "u1")
        self.assertIn(USER_NUMBER_NOT_IN_INPUT, r["user_tick_reason"])
        from hfa import ui
        self.assertIn(USER_NUMBER_NOT_IN_INPUT, ui.render([c]).replace("&#x27;", "'"))

    def test_eval_row_carries_reason(self):
        case = {"id": "X9", "gold": "KILL", "input": "30 USD job, my cost is 26 USD.", "claims": []}
        prop = {"decision": "NEEDS_HUMAN", "claims": [
            {"claim_id": "u0", "text": "my cost is 20 USD", "kind": "USER_ATTESTED", "critical": True}]}
        p = Proposer(MockClient(lambda s, u: json.dumps(prop)), user_confirm=True,
                     user_input=evalrun.case_user_input(case))
        row = evalrun.run_case(case, p, demo._DemoFixture())
        self.assertIn(USER_NUMBER_NOT_IN_INPUT, row["user_unticked"]["u0"])
        self.assertIn(USER_NUMBER_NOT_IN_INPUT, row["reason_texts"]["u0"])

    def test_boundaries(self):
        inp = "A fixed price job that pays 30 USD. My own cost is 26 USD. Fee 3,5%."
        for claim in ("my cost is 26", "my cost is 26.00", "my cost is 26 USD", "my cost is $26",
                      "my cost is USD 26", "my cost is 26.00 dollars", "fee 3.5%", "fee 3,50 %",
                      "30 USD minus 3.5% leaves more than 26 USD", "no number here"):
            self.assertEqual(missing_numbers(claim, inp), [], claim)
        for claim, miss in (("my cost is 20 USD", ["20 USD"]), ("my cost is 26 EUR", ["26 EUR"]),
                            ("my cost is 26%", ["26%"]), ("fee 3.5", ["3.5"]),
                            ("fee 35%", ["35%"]), ("pays 30 USD, cost 20 USD", ["20 USD"]),
                            ("pays 31 USD, cost 26 USD, fee 3.5%", ["31 USD"])):
            self.assertEqual(missing_numbers(claim, inp), miss, claim)
        # a currency in the claim needs the currency in the input; a bare number never matches a %
        self.assertEqual(missing_numbers("cost 26 USD", "my cost is 26"), ["26 USD"])
        self.assertEqual(missing_numbers("cost 26", "my cost is 26"), [])
        self.assertEqual(missing_numbers("fee 20", "Fiverr 20%-ot von le"), ["20"])
        self.assertEqual(missing_numbers("fee 20%", "Fiverr 20%-ot von le"), [])
        # thousands, millions, ambiguous separators
        self.assertEqual(missing_numbers("2025: 23 000 €", "tavaly 23 000 € forgalom 2025-ben"), [])
        self.assertEqual(missing_numbers("2026-os bevételem < 20 M Ft",
                                         "2026-ban 20 M Ft alatt maradok"), [])
        self.assertEqual(missing_numbers("20 000 000 Ft", "20 M Ft"), [])
        self.assertEqual(missing_numbers("20 Ft", "20.000 Ft"), ["20 Ft"])     # never a looser match
        self.assertEqual(missing_numbers("cost 1642", "cost 1,642"), [])
        self.assertEqual(missing_numbers("cost 1642", "cost 1,643"), ["1642"])


class HoldoutLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        self.cases = holdout.load_holdout(HOLDOUT)
        os.makedirs("snaps")
        for c in self.cases:
            for p in evalrun._snapshot_map(c, "snaps").values():
                with open(p, "w") as f:
                    f.write("# url=x\n\nplaceholder\n")
        holdout.freeze("gate")
        self.sha = holdout.freeze("prompt")["prompt_sha256"]
        self.args = ["--holdout", HOLDOUT, "--frozen-prompt-sha", self.sha, "--snapshots", "snaps", "--out", "runs"]

    def _main(self, client, args=None):
        with mock.patch.object(evalrun, "client_from_env", lambda: client):
            return evalrun.main(args or self.args)

    def _lock(self):
        with open(holdout.RUN_LOCK_PATH) as f:
            return json.load(f)

    def test_exception_mid_run_keeps_lock_and_blocks_rerun(self):
        client = ScriptedReal(self.cases)
        real_run_case, n = evalrun.run_case, {"i": 0}

        def boom(case, proposer, fetcher):
            n["i"] += 1
            if n["i"] == 2:
                raise RuntimeError("simulated crash")
            return real_run_case(case, proposer, fetcher)

        with mock.patch.object(evalrun, "run_case", boom):
            with self.assertRaises(RuntimeError):
                self._main(client)
        self.assertEqual(client.calls, 1)
        lock = self._lock()
        self.assertEqual(lock["state"], "ABORTED")
        self.assertIn("simulated crash", lock["error"])
        with self.assertRaises(holdout.HoldoutLocked):
            self._main(client)
        self.assertEqual(client.calls, 1)

    def test_started_lock_left_by_a_kill_blocks_rerun_with_clear_message(self):
        holdout.start_run({"llm": "anthropic:claude-haiku-4-5"})   # process died right after this
        client = ScriptedReal(self.cases)
        with self.assertRaises(holdout.HoldoutLocked) as cm:
            self._main(client)
        msg = str(cm.exception)
        self.assertIn("STARTED", msg)
        self.assertIn("002", msg)
        self.assertIn("void", msg)
        self.assertEqual(client.calls, 0)
        self.assertEqual(self._lock()["state"], "STARTED")          # nothing unlocked automatically

    def test_manual_void_needs_a_reason_file(self):
        holdout.start_run({})
        void = "data/eval/HOLDOUT_RUN.void-20261028T120000Z.json"
        os.rename(holdout.RUN_LOCK_PATH, void)
        with self.assertRaises(holdout.HoldoutLocked):
            holdout.check_lock()
        with open(void[:-5] + ".reason.txt", "w") as f:
            f.write("")
        with self.assertRaises(holdout.HoldoutLocked):
            holdout.check_lock()
        with open(void[:-5] + ".reason.txt", "w") as f:
            f.write("002 decision 2026-10-28: provider outage before any result was seen.\n")
        holdout.check_lock()                                         # allowed again

    def test_budget_too_small_no_lock_no_call(self):
        from hfa.proposer.budget import Budget
        Budget().record("claude-haiku-4-5", 0, 160_000)              # $0.80 of the $1.00 day used
        client = ScriptedReal(self.cases)
        self.assertEqual(self._main(client), 4)
        self.assertFalse(os.path.exists(holdout.RUN_LOCK_PATH))
        self.assertEqual(client.calls, 0)

    def test_budget_precheck_factor_and_total_cap(self):
        from hfa.proposer.budget import BudgetStatus
        ok = BudgetStatus("d", 0.0, 0.0, 1.0, 15.0)
        holdout.budget_precheck(0.6, ok)                            # 0.9 < 1.0
        with self.assertRaises(holdout.BudgetTooSmall):
            holdout.budget_precheck(0.7, ok)                        # 1.05 >= 1.0
        with self.assertRaises(holdout.BudgetTooSmall):
            holdout.budget_precheck(0.3, BudgetStatus("d", 0.0, 14.7, 1.0, 15.0))   # total left 0.3
        est = evalrun.estimate_cost(self.cases, "claude-haiku-4-5")
        self.assertGreater(est, 11 * 2000 * 5 / 1_000_000)           # at least the output side of 11 calls

    def test_missing_snapshot_no_lock(self):
        os.remove(next(iter(evalrun._snapshot_map(self.cases[0], "snaps").values())))
        client = ScriptedReal(self.cases)
        self.assertEqual(self._main(client), 2)
        self.assertFalse(os.path.exists(holdout.RUN_LOCK_PATH))
        self.assertEqual(client.calls, 0)

    def test_full_run_fills_the_started_lock(self):
        client = ScriptedReal(self.cases)
        self.assertEqual(self._main(client), 0)
        lock = self._lock()
        self.assertEqual(lock["state"], "COMPLETE")
        self.assertIn("started_at", lock)
        self.assertEqual(lock["summary"]["n_cases"], 11)
        self.assertEqual(client.calls, 11)
        with self.assertRaises(holdout.HoldoutLocked):
            self._main(client)
        self.assertEqual(client.calls, 11)

    def test_gate_hash_unchanged(self):
        with open(os.path.join(HERE, "data", "eval", "FREEZE_GATE.json")) as f:
            self.assertEqual(holdout.decision_code_sha256(), json.load(f)["decision_code_sha256"])


if __name__ == "__main__":
    unittest.main()
