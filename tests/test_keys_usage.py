import json
import os
import tempfile
import unittest
from unittest import mock

from hfa import demo, evalrun
from hfa.proposer import Proposer
from hfa.proposer.base import usage_record
from hfa.proposer.budget import Budget
from hfa.proposer.providers import UnavailableClient, client_from_env

from .helpers import RULE, StaticFetcher, page
from .test_proposer_budget_eval import FakeReal

FAKE_KEY = "test-not-a-real-key-000"
SHELL_KEYS = {"ANTHROPIC_API_KEY": "shell-key-must-not-be-used", "OPENAI_API_KEY": "x",
              "HFA_API_KEY": "y", "LLMAPI_KEY": "z"}


class KeyFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)                    # no ./.env here
        self.addCleanup(os.chdir, self.cwd)
        self.keyfile = os.path.join(self.tmp.name, "anthropic.env")

    def _env(self, **extra):
        base = {k: v for k, v in os.environ.items()
                if k not in ("HFA_ENV_FILE", "HFA_LLM_PROVIDER", "HFA_MODEL")}
        base.update(extra)
        return mock.patch.dict(os.environ, base, clear=True)

    def test_key_from_explicit_key_file(self):
        with open(self.keyfile, "w") as f:
            f.write(f"# comment\nANTHROPIC_API_KEY={FAKE_KEY}\n")
        with self._env(HFA_ENV_FILE=self.keyfile):
            c = client_from_env()
        self.assertEqual(c.label, "anthropic:claude-haiku-4-5")
        self.assertNotIn(FAKE_KEY, repr(c.__dict__.keys()))

    def test_key_from_dot_env_in_cwd(self):
        with open(".env", "w") as f:
            f.write(f"ANTHROPIC_API_KEY={FAKE_KEY}\n")
        with self._env():
            self.assertEqual(client_from_env().label, "anthropic:claude-haiku-4-5")

    def test_shell_environment_keys_are_never_used(self):
        with self._env(**SHELL_KEYS):
            c = client_from_env()
        self.assertIsInstance(c, UnavailableClient)
        self.assertIn("key file not found", c.reason)
        with self._env(**SHELL_KEYS, HFA_LLM_PROVIDER="openai_compat",
                       HFA_OPENAI_BASE_URL="https://api.example.com/v1"):
            self.assertIsInstance(client_from_env(), UnavailableClient)

    def test_empty_key_or_missing_file_is_unavailable(self):
        with open(self.keyfile, "w") as f:
            f.write("ANTHROPIC_API_KEY=\n")
        with self._env(HFA_ENV_FILE=self.keyfile):
            self.assertIn("missing or empty", client_from_env().reason)
        with self._env(HFA_ENV_FILE=os.path.join(self.tmp.name, "nope.env")):
            self.assertIn("not found", client_from_env().reason)

    def test_mock_only_with_explicit_switch(self):
        with self._env(HFA_LLM_PROVIDER="mock"):
            self.assertIsInstance(client_from_env(), UnavailableClient)
        self.assertEqual(client_from_env(mock=True, mock_responder=lambda s, u: "{}").label, "mock")

    def test_unavailable_is_explicit_in_result_and_log(self):
        log = os.path.join(self.tmp.name, "usage.jsonl")
        b = Budget(os.path.join(self.tmp.name, "b.json"))
        r = Proposer(UnavailableClient("key file not found"), b, usage_log=log).propose("job", [])
        self.assertEqual((r.status, r.llm), ("unavailable", "unavailable"))
        self.assertIn("llm: unavailable", r.message)
        with open(log) as f:
            self.assertEqual(json.loads(f.readline())["status"], "unavailable")
        self.assertEqual(b.status().spent_total, 0.0)

    def test_demo_real_without_key_shows_unavailable(self):
        with self._env(**SHELL_KEYS), \
                mock.patch.object(demo, "LiveFetcher", lambda: StaticFetcher(page(RULE))):
            c = demo.run_real(invoice=True)
        self.assertEqual((c["llm"], c["status"]), ("unavailable", "unavailable"))
        self.assertIn("llm: unavailable", c["banner"])
        self.assertNotIn("invoice", c.get("invoice", "") if isinstance(c.get("invoice"), dict) else "")
        with open(os.path.join("data", "traces", "demo.jsonl")) as f:
            lines = [json.loads(x) for x in f]
        self.assertTrue(any(x.get("status") == "unavailable" for x in lines))

    def test_load_dotenv_never_exports_llm_key(self):
        with open(self.keyfile, "w") as f:
            f.write(f"ANTHROPIC_API_KEY={FAKE_KEY}\nPAYPAL_BASE_URL=https://api-m.sandbox.paypal.com\n")
        with self._env():
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("PAYPAL_BASE_URL", None)
            demo.load_dotenv(self.keyfile)
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
            self.assertEqual(os.environ.get("PAYPAL_BASE_URL"), "https://api-m.sandbox.paypal.com")

    def test_eval_without_mock_switch_is_unavailable(self):
        self.assertEqual(evalrun.main(["--cases", os.path.join(self.cwd, "data", "eval", "cases_public.jsonl")]), 3)


class UsageTests(unittest.TestCase):
    def test_usage_logged_and_budget_counts_same_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "usage.jsonl")
            b = Budget(os.path.join(tmp, "b.json"))
            p = Proposer(FakeReal(), b, usage_log=log)
            r = p.propose("job", [])
            self.assertEqual(r.usage["input_tokens"], 1200)
            self.assertEqual(r.usage["output_tokens"], 500)
            self.assertAlmostEqual(r.usage["cost_usd_estimate"], (1200 * 1 + 500 * 5) / 1e6)
            self.assertIn("estimate", r.usage["price_source"])
            self.assertAlmostEqual(b.status().spent_today, r.usage["cost_usd_estimate"])
            self.assertAlmostEqual(b.status().spent_total, r.usage["cost_usd_estimate"])
            with open(log) as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["llm"], "anthropic:claude-haiku-4-5")
            self.assertFalse(any("key" in k.lower() for k in rec))      # no key / key id fields

    def test_unknown_price_records_tokens_only(self):
        rec = usage_record("some-unpriced-model", 10, 20, "anthropic:some-unpriced-model")
        self.assertIsNone(rec["cost_usd_estimate"])
        self.assertEqual((rec["input_tokens"], rec["output_tokens"]), (10, 20))


if __name__ == "__main__":
    unittest.main()
