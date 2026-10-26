import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from hfa import evalrun
from hfa.fetcher import FixtureFetcher
from hfa.pipeline import run
from hfa.proposer import DEFAULT_MODEL, MockClient, Proposer, client_from_env, parse_proposal
from hfa.proposer.base import LLMResponse, ProposalInvalid
from hfa.proposer.budget import Budget, BudgetExhausted, UnknownPrice, cost_usd
from hfa.proposer.prompt import prompt_sha256
from hfa.redact import redact

from .helpers import FILLER, RULE

GOOD = {"decision": "GO", "rationale": "ok", "claims": [
    {"claim_id": "c1", "text": "rule", "kind": "FACT", "critical": True, "quote": RULE,
     "source_url": "https://example.org/rules", "rule_id": "R1", "polarity": "SUPPORTS"}]}


class FakeReal:
    """Pretends to be a metered provider; counts calls; never touches a network."""
    provider = "anthropic"

    def __init__(self, model="claude-haiku-4-5", out_tokens=500):
        self.model, self.calls, self.out_tokens = model, 0, out_tokens

    @property
    def label(self):
        return f"anthropic:{self.model}"

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return LLMResponse(json.dumps(GOOD), 1200, self.out_tokens)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = os.path.join(self.tmp.name, "data", "llm_budget.json")

    def tearDown(self):
        self.tmp.cleanup()


class ProposalParsingTests(unittest.TestCase):
    def test_parse_valid_and_fenced(self):
        v, k, why, claims = parse_proposal("```json\n" + json.dumps(GOOD) + "\n```")
        self.assertEqual((v.value, len(claims), claims[0].quote), ("GO", 1, RULE))

    def test_fact_without_quote_rejected(self):
        bad = {"decision": "GO", "rationale": "", "claims": [{"claim_id": "x", "text": "t", "kind": "FACT",
                                                              "critical": True, "polarity": "SUPPORTS"}]}
        with self.assertRaises(ProposalInvalid):
            parse_proposal(json.dumps(bad))
        with self.assertRaises(ProposalInvalid):
            parse_proposal("I think GO")

    def test_mock_label_and_prompt_hash(self):
        r = Proposer(MockClient(lambda s, u: json.dumps(GOOD))).propose("job", [])
        self.assertEqual((r.status, r.llm, r.prompt_sha256), ("ok", "mock", prompt_sha256()))


class BudgetTests(TmpDirCase):
    def test_default_model_is_haiku(self):
        self.assertEqual(DEFAULT_MODEL, "claude-haiku-4-5")
        with mock.patch.dict(os.environ, {"HFA_LLM_PROVIDER": "anthropic", "HFA_API_KEY": "x"}, clear=False):
            os.environ.pop("HFA_MODEL", None)
            self.assertEqual(client_from_env().label, "anthropic:claude-haiku-4-5")

    def test_cost_from_usage(self):
        self.assertAlmostEqual(cost_usd("claude-haiku-4-5", 1_000_000, 100_000), 1.5)

    def test_spend_recorded_from_usage(self):
        b = Budget(self.state, Clock())
        r = Proposer(FakeReal(), b).propose("job", [])
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.cost_usd, (1200 * 1 + 500 * 5) / 1e6)
        self.assertAlmostEqual(b.status().spent_total, r.cost_usd)

    def test_daily_cap_blocks_call_with_explicit_state(self):
        clock = Clock()
        b = Budget(self.state, clock)
        b.record("claude-haiku-4-5", 0, 199_900)          # $0.9995 today
        client = FakeReal()
        r = Proposer(client, b).propose("job", [])
        self.assertEqual(r.status, "budget_exhausted")
        self.assertIn("budget_exhausted", r.message)
        self.assertEqual(client.calls, 0)                  # no call was made
        clock.now += timedelta(days=1)                     # new UTC day
        self.assertEqual(Proposer(client, b).propose("job", []).status, "ok")

    def test_total_cap(self):
        clock = Clock()
        b = Budget(self.state, clock)
        for _ in range(16):                                # $0.99 per day over 16 days
            b.record("claude-haiku-4-5", 0, 198_000)
            clock.now += timedelta(days=1)
        self.assertGreaterEqual(b.status().spent_total, 15.0)
        self.assertEqual(Proposer(FakeReal(), b).propose("job", []).status, "budget_exhausted")

    def test_env_can_only_lower_caps(self):
        with mock.patch.dict(os.environ, {"HFA_BUDGET_DAILY_USD": "50", "HFA_BUDGET_TOTAL_USD": "5"}):
            b = Budget(self.state)
            self.assertEqual((b.daily_cap, b.total_cap), (1.0, 5.0))

    def test_unknown_price_and_corrupt_state_refuse(self):
        b = Budget(self.state, Clock())
        with self.assertRaises(UnknownPrice):
            b.check("some-unpriced-model", 10, 10)
        os.makedirs(os.path.dirname(self.state), exist_ok=True)
        with open(self.state, "w") as f:
            f.write("{broken")
        with self.assertRaises(BudgetExhausted):
            b.status()

    def test_real_provider_without_budget_refused(self):
        client = FakeReal()
        r = Proposer(client, None).propose("job", [])
        self.assertEqual((r.status, client.calls), ("error", 0))

    def test_pipeline_surfaces_budget_exhausted(self):
        b = Budget(self.state, Clock())
        b.record("claude-haiku-4-5", 0, 200_000)
        trace = os.path.join(self.tmp.name, "trace.jsonl")
        res = run("job", [], Proposer(FakeReal(), b), FixtureFetcher({}), trace_path=trace)
        self.assertEqual((res.status, res.decision), ("budget_exhausted", None))
        self.assertEqual(res.to_dict()["llm"], "anthropic:claude-haiku-4-5")
        with open(trace) as f:
            line = json.loads(f.readline())
        self.assertEqual((line["status"], line["llm"]), ("budget_exhausted", "anthropic:claude-haiku-4-5"))


class EvalHonestyTests(TmpDirCase):
    def _snap(self, name, url, text, http=200):
        p = os.path.join(self.tmp.name, name)
        with open(p, "w") as f:
            f.write(f"# url={url}\n# final_url={url}\n# http={http} ctype=text/html; charset=utf-8\n"
                    f"# fetched_at=2026-10-04T15:00:00Z\n\n{FILLER} {text} {FILLER}\n")
        return name

    def _cases(self):
        u1, u2 = "https://example.org/a", "https://example.org/b"
        self._snap("A.txt", u1, RULE)
        self._snap("B.txt", u2, "Nothing relevant on this page at all.")
        return [
            {"id": "X1", "gold": "GO", "naive_false_go_trap": False, "input": "a", "split": "dev",
             "claims": [{"quote": RULE, "role": "SUPPORTS", "url": u1, "snapshot": "A.txt"}]},
            {"id": "X2", "gold": "UNVERIFIED", "naive_false_go_trap": True, "input": "b", "split": "dev",
             "claims": [{"quote": "Platform charges a flat ten percent fee", "role": "SUPPORTS",
                         "url": u2, "snapshot": "B.txt"}]},
            {"id": "X3", "gold": "KILL", "naive_false_go_trap": True, "input": "c", "split": "dev",
             "claims": [{"quote": RULE, "role": "CONTRADICTS", "url": u1, "snapshot": "A.txt"}]},
        ]

    def _rows(self):
        rows = []
        for case in self._cases():
            client = MockClient(lambda s, u, c=case: json.dumps(evalrun.naive_proposal(c)))
            fetcher = FixtureFetcher(evalrun._snapshot_map(case, self.tmp.name))
            rows.append(evalrun.run_case(case, Proposer(client), fetcher))
        return rows

    def test_verifier_reduces_false_go_on_mock(self):
        s = evalrun.aggregate(self._rows())
        self.assertTrue(s["is_mock"])
        self.assertEqual(s["no_verifier"]["false_go"], 2)
        self.assertEqual(s["verifier"]["false_go"], 0)
        self.assertEqual(s["verifier"]["false_kill_count"], 0)

    def test_aggregate_refuses_mixed_llm_rows(self):
        rows = self._rows()
        rows[0] = dict(rows[0], llm="anthropic:claude-haiku-4-5")
        with self.assertRaises(evalrun.MixedLLMError):
            evalrun.aggregate(rows)

    def test_real_table_refuses_mock_rows_and_mock_banner(self):
        rows = self._rows()
        s = evalrun.aggregate(rows)
        with self.assertRaises(evalrun.MixedLLMError):
            evalrun.write_results(rows, s, self.tmp.name, real=True)
        path = evalrun.write_results(rows, s, self.tmp.name, real=False)
        with open(path) as f:
            self.assertEqual(json.load(f)["banner"], evalrun.MOCK_BANNER)
        self.assertIn(evalrun.MOCK_BANNER, evalrun.format_table(rows, s))
        self.assertTrue(all(r["llm"] == "mock" for r in rows))

    def test_holdout_locked_until_prompt_frozen(self):
        cases = [{"id": "T1", "split": "dev"}, {"id": "R1", "split": "holdout"}]
        self.assertEqual([c["id"] for c in evalrun.select_cases(cases, False, None)], ["T1"])
        with self.assertRaises(evalrun.HoldoutLocked):
            evalrun.select_cases(cases, True, None)
        with self.assertRaises(evalrun.HoldoutLocked):
            evalrun.select_cases(cases, True, "0" * 64)
        self.assertEqual([c["id"] for c in evalrun.select_cases(cases, True, prompt_sha256())], ["R1"])

    def test_public_case_file_has_no_long_quotes(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "data", "eval", "cases_public.jsonl"), encoding="utf-8") as f:
            cases = [json.loads(line) for line in f]
        self.assertEqual(len(cases), 40)
        for c in cases:
            for cl in c["claims"]:
                self.assertLessEqual(len(cl["quote"]), 200)
                self.assertEqual(len(cl["quote_sha256"]), 64)


class RedactTests(unittest.TestCase):
    def test_positive(self):
        self.assertNotIn("abc.def", redact("Authorization: Bearer abc.def"))
        self.assertIn("[EMAIL]", redact("mail me at someone@example.com"))
        self.assertIn("[IP]", redact("host 203.0.113.9"))
        self.assertIn("[PATH]", redact("file /Users/someone/secret.txt"))
        self.assertNotIn("s3cr3t", redact('{"client_secret": "s3cr3t"}'))
        self.assertEqual(redact("patient diagnosis follows"), "[HEALTH_REDACTED]")

    def test_negative_keeps_evidence_hashes(self):
        h = "a" * 64
        self.assertEqual(redact(f"snippet_sha256 {h} state VERIFIED"), f"snippet_sha256 {h} state VERIFIED")


if __name__ == "__main__":
    unittest.main()
