import json
import os
import tempfile
import unittest
from unittest import mock

from hfa import demo, evalrun, holdout
from hfa.gate import decide
from hfa.models import Claim, ClaimKind, Evidence, EvidenceState as S, Reason as R, Verdict as V
from hfa.paypal import SandboxClient
from hfa.paypal.client import InvoiceStatus, LiveHostRefused
from hfa.proposer import MockClient, Proposer
from hfa.proposer.base import LLMResponse, parse_proposal

from .helpers import FILLER, RULE, StaticFetcher, fact, page
from .test_paypal import FAKE_ID, FAKE_SECRET, FakePayPal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TICKED_BY_MODEL = {"decision": "GO", "rationale": "x", "claims": [
    {"claim_id": "c1", "text": "rule", "kind": "FACT", "critical": True, "quote": RULE,
     "source_url": "https://example.org/rules", "polarity": "SUPPORTS"},
    {"claim_id": "u1", "text": "I am a sole trader", "kind": "USER_ATTESTED", "critical": True,
     "polarity": "SUPPORTS", "user_confirmed": True}]}


class UserTickTests(unittest.TestCase):
    def _decide(self, user_confirm):
        r = Proposer(MockClient(lambda s, u: json.dumps(TICKED_BY_MODEL)), user_confirm=user_confirm).propose("x", [])
        ev = [Evidence("c1", S.VERIFIED, R.EXACT_MATCH), Evidence("u1", S.NOT_APPLICABLE, R.NOT_A_FACT)]
        return r, decide(r.decision, r.claims, ev, r.llm)

    def test_model_cannot_tick_user_claim(self):
        _, _, _, claims = parse_proposal(json.dumps(TICKED_BY_MODEL))
        self.assertFalse(next(c for c in claims if c.claim_id == "u1").user_confirmed)
        self.assertFalse(Claim.from_dict({"claim_id": "x", "text": "t", "kind": "USER_ATTESTED",
                                          "user_confirmed": "true"}).user_confirmed)
        _, d = self._decide(user_confirm=False)
        self.assertEqual(d.final, V.NEEDS_HUMAN)
        self.assertTrue(any(g.startswith("G5 fail") for g in d.gate_log))

    def test_user_flag_ticks(self):
        _, d = self._decide(user_confirm=True)
        self.assertEqual(d.final, V.GO)

    def test_gate_hash_unchanged(self):
        with open(os.path.join(HERE, "data", "eval", "FREEZE_GATE.json")) as f:
            frozen = json.load(f)["decision_code_sha256"]
        self.assertEqual(holdout.decision_code_sha256(), frozen)


class RawStatusTests(unittest.TestCase):
    def test_raw_status_not_merged(self):
        for raw in ("PAID", "MARKED_AS_PAID", "PAID_EXTERNAL", "PARTIALLY_PAID", "UNPAID", "SENT", "DRAFT",
                    "AUTO_CANCELLED", "SHARED"):
            st = InvoiceStatus("INV2-X", raw)
            self.assertEqual(st.display, raw)
        self.assertTrue(InvoiceStatus("i", "PAID").paid_by_payer)
        self.assertFalse(InvoiceStatus("i", "MARKED_AS_PAID").paid_by_payer)

    def test_record_payment_body_and_sandbox_only(self):
        t = FakePayPal()
        c = SandboxClient(FAKE_ID, FAKE_SECRET, transport=t)
        c.record_payment("INV2-TEST-0001", "BANK_TRANSFER", "2026-10-26")
        method, url, _, body = t.calls[-1]
        self.assertTrue(url.endswith("/v2/invoicing/invoices/INV2-TEST-0001/payments"))
        self.assertEqual(json.loads(body), {"method": "BANK_TRANSFER", "payment_date": "2026-10-26"})
        with self.assertRaises(ValueError):
            c.record_payment("INV2-TEST-0001", "CRYPTO")
        c.base_url = "https://api-m.paypal.com"
        with self.assertRaises(LiveHostRefused):
            c.record_payment("INV2-TEST-0001")


class ScriptedReal:
    """Fake metered provider: answers with the naive proposal for whatever case it is shown."""
    provider, model = "anthropic", "claude-haiku-4-5"

    def __init__(self, cases):
        self.cases, self.calls, self.prompts = cases, 0, []

    @property
    def label(self):
        return "anthropic:claude-haiku-4-5"

    def complete(self, system, user, max_tokens):
        self.calls += 1
        self.prompts.append(user)
        case = next(c for c in self.cases if c["input"] in user)
        prop = evalrun.naive_proposal(case)
        for cl in prop["claims"]:
            cl.pop("user_confirmed", None)
        return LLMResponse(json.dumps(prop), 3000, 400)


class RealEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        os.makedirs("snaps")
        url = "https://example.org/a"
        with open("snaps/A.txt", "w") as f:
            f.write(f"# url={url}\n# final_url={url}\n# http=200 ctype=text/html\n"
                    f"# fetched_at=2026-10-04T15:00:00Z\n\n{FILLER} {RULE} {FILLER}\n")
        self.cases = [
            {"id": "X1", "gold": "GO", "naive_false_go_trap": False, "input": "case one input", "split": "dev",
             "claims": [{"quote": RULE, "role": "SUPPORTS", "url": url, "snapshot": "A.txt"}]},
            {"id": "X2", "gold": "KILL", "naive_false_go_trap": True, "input": "case two input", "split": "dev",
             "claims": [{"quote": RULE, "role": "CONTRADICTS", "url": url, "snapshot": "A.txt"}]}]
        with open("cases.jsonl", "w") as f:
            f.write("\n".join(json.dumps(c) for c in self.cases) + "\n")

    def _main(self, args, client):
        with mock.patch.object(evalrun, "client_from_env", lambda: client):
            return evalrun.main(args)

    def test_real_branch_writes_real_table_with_usage_and_source_text(self):
        client = ScriptedReal(self.cases)
        rc = self._main(["--cases", "cases.jsonl", "--snapshots", "snaps", "--out", "runs"], client)
        self.assertEqual(rc, 0)
        with open("runs/real/eval_anthropic_claude-haiku-4-5.json") as f:
            out = json.load(f)
        self.assertIsNone(out["banner"])
        self.assertTrue(all(r["llm"] == "anthropic:claude-haiku-4-5" for r in out["rows"]))
        self.assertTrue(out["summary"]["complete"])
        self.assertEqual(out["rows"][0]["usage"]["input_tokens"], 3000)
        self.assertIn(RULE, client.prompts[0])            # the model saw the source text
        self.assertFalse(os.path.exists("runs/mock"))
        self.assertTrue(os.path.exists("runs/real/usage.jsonl"))

    def test_no_key_is_unavailable_exit(self):
        from hfa.proposer.providers import UnavailableClient
        self.assertEqual(self._main(["--cases", "cases.jsonl", "--snapshots", "snaps"],
                                    UnavailableClient("key file not found")), 3)

    def test_budget_exhaustion_stops_and_marks_incomplete(self):
        from hfa.proposer.budget import Budget
        Budget().record("claude-haiku-4-5", 0, 200_000)       # today's cap used up
        client = ScriptedReal(self.cases)
        self._main(["--cases", "cases.jsonl", "--snapshots", "snaps", "--out", "runs"], client)
        self.assertEqual(client.calls, 0)
        with open("runs/real/eval_anthropic_claude-haiku-4-5.json") as f:
            self.assertFalse(json.load(f)["summary"]["complete"])

    def test_holdout_real_needs_freezes_and_runs_once(self):
        src = os.path.join(HERE, "data", "eval", "holdout_public.jsonl")
        client = ScriptedReal([])
        with self.assertRaises(holdout.HoldoutLocked):      # nothing frozen in this tmp dir
            self._main(["--holdout", src, "--snapshots", "snaps"], client)
        with self.assertRaises(holdout.HoldoutLocked):      # mock never
            evalrun.main(["--holdout", src, "--mock", "--snapshots", "snaps"])
        holdout.freeze("gate")
        rec = holdout.freeze("prompt")
        with self.assertRaises(holdout.HoldoutLocked):      # wrong prompt sha
            self._main(["--holdout", src, "--frozen-prompt-sha", "0" * 64, "--snapshots", "snaps"], client)
        holdout.record_run({"n": 11})                        # pretend it already ran once
        with self.assertRaises(holdout.HoldoutLocked):
            self._main(["--holdout", src, "--frozen-prompt-sha", rec["prompt_sha256"], "--snapshots", "snaps"],
                       client)
        self.assertEqual(client.calls, 0)

    def test_mock_rows_never_in_real_table(self):
        rows = [{"id": "a", "llm": "mock", "gold": "GO", "status": "ok", "no_verifier": "GO", "verifier": "GO"}]
        with self.assertRaises(evalrun.MixedLLMError):
            evalrun.write_results(rows, evalrun.aggregate(rows), "runs", real=True)


class JobCliTests(unittest.TestCase):
    def test_job_tick_and_not_worth(self):
        c = demo.run_job("30", "26", "example", mock=True, user_confirm=True)
        self.assertEqual((c["llm"], c["final"]), ("mock", "GO"))      # 27.60 >= 26
        c = demo.run_job("30", "26", "example", mock=True, user_confirm=False)
        self.assertEqual(c["final"], "NEEDS_HUMAN")                    # no user tick
        c = demo.run_job("30", "29", "example", mock=True, user_confirm=True)
        self.assertEqual((c["final"], c["kill_class"]), ("KILL", "NOT_WORTH"))
        self.assertEqual(c["calcs"][0]["values"]["net"], "27.60")

    def test_job_rejects_real_platform_in_mock_and_bad_numbers(self):
        with self.assertRaises(SystemExit):
            demo.run_job("30", "26", "freelancer", mock=True, user_confirm=True)
        with self.assertRaises(SystemExit):
            demo.run_job("abc", "26", "example", mock=True, user_confirm=True)
        with self.assertRaises(SystemExit):
            demo.run_job("30", "26", "nowhere", mock=True, user_confirm=True)

    def test_card_shows_quote_url_and_source_hash(self):
        c = demo.run_mock()[0]
        r = next(x for x in c["receipts"] if x["kind"] == "FACT")
        self.assertTrue(r["quote"] and r["source_url"].startswith("https://"))
        self.assertEqual(len(r["source_sha256"]), 64)
        self.assertEqual(c["invoice"]["status"], "UNPAID")


if __name__ == "__main__":
    unittest.main()
