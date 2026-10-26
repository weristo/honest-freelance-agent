import os
import tempfile
import unittest
from unittest import mock

from hfa import demo
from hfa.paypal import SandboxClient
from hfa.proposer.budget import Budget

from .helpers import RULE, StaticFetcher, page
from .test_paypal import FAKE_ID, FAKE_SECRET, FakePayPal
from .test_proposer_budget_eval import FakeReal


class MockDemoTests(unittest.TestCase):
    def test_mock_demo_is_labelled_and_offline(self):
        cards = demo.run_mock()
        self.assertTrue(all(c["llm"] == "mock" and c["banner"] == demo.MOCK_BANNER for c in cards))
        self.assertEqual([c["final"] for c in cards], ["GO", "KILL", "NEEDS_HUMAN", "KILL"])
        self.assertEqual(cards[1]["kill_class"], "POLICY_REFUSAL")
        self.assertEqual(cards[0]["invoice"]["paypal"], "mock (no network)")
        self.assertNotIn("invoice", cards[1])
        self.assertNotIn("invoice", cards[2])
        self.assertEqual(cards[3]["kill_class"], "NOT_WORTH")       # 30 - 8% = 27.60 < 29
        self.assertEqual(cards[3]["calcs"][0]["values"]["net"], "27.60")


class RealDemoWiringTests(unittest.TestCase):
    """Real-mode wiring with fake LLM, fake fetch and fake PayPal: no network."""

    def test_real_path_invoices_only_on_go(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        transport = FakePayPal()
        settings = {"HFA_LLM_PROVIDER": "anthropic", "HFA_SANDBOX_MERCHANT_EMAIL": "m@example.com",
                    "HFA_SANDBOX_BUYER_EMAIL": "b@example.com"}
        with mock.patch.dict(os.environ, settings), \
                mock.patch.object(demo, "client_from_env", lambda: FakeReal()), \
                mock.patch.object(demo, "LiveFetcher", lambda: StaticFetcher(page(RULE))), \
                mock.patch.object(demo, "Budget", lambda: Budget(os.path.join(tmp.name, "b.json"))), \
                mock.patch.object(SandboxClient, "from_env",
                                  classmethod(lambda cls: cls(FAKE_ID, FAKE_SECRET, transport=transport))):
            c = demo.run_real(invoice=True)
        self.assertEqual((c["llm"], c["final"]), ("anthropic:claude-haiku-4-5", "GO"))
        self.assertEqual(c["invoice"]["paypal"], "sandbox")
        self.assertTrue(os.path.exists(os.path.join(tmp.name, "data", "traces", "demo.jsonl")))

    def test_real_mode_refuses_without_provider(self):
        with mock.patch.dict(os.environ, {"HFA_LLM_PROVIDER": "mock"}):
            with self.assertRaises(SystemExit):
                demo.run_real(invoice=False)


if __name__ == "__main__":
    unittest.main()
