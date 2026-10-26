import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from unittest import mock

from hfa import demo, ui
from hfa.__main__ import main as cli
from hfa.paypal import SandboxClient

from .test_paypal import FAKE_ID, FAKE_SECRET, FakePayPal


class UiTests(unittest.TestCase):
    def test_binds_to_loopback_only_and_renders_mock_card(self):
        srv = ui.make_server(demo.run_mock, port=0)
        try:
            self.assertEqual(srv.server_address[0], "127.0.0.1")
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            port = srv.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                body = r.read().decode()
                csp = r.headers.get("Content-Security-Policy")
            self.assertIn("MOCK - scripted demo, NOT a model measurement", body)
            self.assertIn("llm: mock", body)
            self.assertIn("Freelancers may invoice clients", body)          # quote text
            self.assertIn("https://marketplace.example/legal/terms", body)   # source URL
            self.assertIn("source SHA-256:", body)
            self.assertIn("UNPAID", body)                                    # raw invoice status
            self.assertIn("default-src 'none'", csp)
            self.assertNotIn("<script", body)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_no_way_to_bind_elsewhere(self):
        self.assertEqual(ui.BIND_HOST, "127.0.0.1")
        with self.assertRaises(TypeError):
            ui.make_server(demo.run_mock, port=0, host="0.0.0.0")       # no host parameter exists

    def test_html_is_escaped(self):
        page = ui.render([{"llm": "mock", "request": "<script>alert(1)</script>", "final": "GO", "receipts": []}])
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;", page)

    def test_unavailable_banner(self):
        page = ui.render([{"llm": "unavailable", "status": "unavailable", "message": "llm: unavailable - no key"}])
        self.assertIn("llm: unavailable", page)


class RemindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        self.t = FakePayPal()

    def _cli(self, args):
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(
                SandboxClient, "from_env", classmethod(lambda cls: cls(FAKE_ID, FAKE_SECRET, transport=self.t))):
            rc = cli(args)
        return rc, json.loads(buf.getvalue())

    def test_draft_never_sends_and_approve_sends_once(self):
        rc, out = self._cli(["remind", "INV2-TEST-0001"])
        self.assertEqual((rc, out["sent"]), (0, False))
        self.assertEqual(self.t.calls, [])                       # nothing went to PayPal
        rc, out = self._cli(["remind", "--approve", "INV2-TEST-0001"])
        self.assertEqual((rc, out["sent"]), (0, True))
        self.assertTrue(self.t.calls[-1][1].endswith("/remind"))
        n = len(self.t.calls)
        rc, out = self._cli(["remind", "--approve", "INV2-TEST-0001"])
        self.assertEqual((rc, out["sent"]), (1, False))           # already sent
        self.assertEqual(len(self.t.calls), n)

    def test_approve_without_draft_refused(self):
        rc, out = self._cli(["remind", "--approve", "INV2-NODRAFT"])
        self.assertEqual((rc, out["sent"]), (1, False))
        self.assertEqual(self.t.calls, [])


if __name__ == "__main__":
    unittest.main()
