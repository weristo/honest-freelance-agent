import json
import os
import unittest
from unittest import mock

from hfa.gate import GoNotAllowed, decide
from hfa.models import Evidence, EvidenceState as S, Reason as R, Verdict as V
from hfa.paypal import (ALLOWED_PAYPAL_BASE, LiveHostRefused, SandboxClient,
                        assert_sandbox, build_invoice, invoice_for_decision, smoke_token)

from .helpers import fact

FAKE_ID, FAKE_SECRET = "test-client-id", "test-client-secret"  # not real credentials


class FakePayPal:
    """In-memory transport that mimics the sandbox responses we rely on."""

    def __init__(self):
        self.calls = []
        self.status = "DRAFT"

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        path = url.replace(ALLOWED_PAYPAL_BASE, "")
        if path == "/v1/oauth2/token":
            assert headers["Authorization"].startswith("Basic ")
            return 200, json.dumps({"access_token": "A" * 97, "expires_in": 32400}).encode()
        assert headers["Authorization"] == "Bearer " + "A" * 97
        if method == "POST" and path == "/v2/invoicing/invoices":
            return 201, json.dumps({"id": "INV2-TEST-0001", "status": "DRAFT"}).encode()
        if path.endswith("/send"):
            self.status = "UNPAID"
            return 202, b""
        if path.endswith("/remind"):
            return 204, b""
        if method == "GET":
            return 200, json.dumps({"id": "INV2-TEST-0001", "status": self.status}).encode()
        return 404, b'{"message": "not found"}'


def go_decision():
    c = [fact(cid="a")]
    return decide(V.GO, c, [Evidence("a", S.VERIFIED, R.EXACT_MATCH)], "mock")


class SandboxAssertTests(unittest.TestCase):
    BLOCKLIST = [
        "https://api-m.paypal.com", "https://api.paypal.com", "https://www.paypal.com",
        "https://api-m.sandbox.paypal.com.evil.tld", "http://api-m.sandbox.paypal.com",
        "https://151.101.1.1", "https://API-M.SANDBOX.PAYPAL.COM.evil", "https://evil.tld/api-m.sandbox.paypal.com",
        "https://api-m.sandbox.paypal.com:8443", "https://user@api-m.sandbox.paypal.com",
        "https://xapi-m.sandbox.paypal.com", "api-m.sandbox.paypal.com",
    ]

    def test_live_and_lookalike_hosts_refused(self):
        for u in self.BLOCKLIST:
            with self.assertRaises(LiveHostRefused, msg=u):
                assert_sandbox(u + "/v1/oauth2/token")

    def test_sandbox_allowed(self):
        assert_sandbox("https://api-m.sandbox.paypal.com/v1/oauth2/token")

    def test_constructor_refuses_live_host(self):
        with self.assertRaises(LiveHostRefused):
            SandboxClient(FAKE_ID, FAKE_SECRET, "https://api-m.paypal.com")

    def test_env_base_url_checked_at_startup(self):
        env = {"PAYPAL_BASE_URL": "https://api-m.paypal.com", "PAYPAL_CLIENT_ID": FAKE_ID,
               "PAYPAL_CLIENT_SECRET": FAKE_SECRET}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(LiveHostRefused):
                SandboxClient.from_env(FakePayPal())

    def test_every_request_is_checked_even_after_tampering(self):
        t = FakePayPal()
        c = SandboxClient(FAKE_ID, FAKE_SECRET, transport=t)
        c.base_url = "https://api-m.paypal.com"      # simulate a bug / tampering
        with self.assertRaises(LiveHostRefused):
            c.fetch_token()
        self.assertEqual(t.calls, [])                # nothing left the process

    def test_repr_hides_credentials(self):
        self.assertNotIn(FAKE_SECRET, repr(SandboxClient(FAKE_ID, FAKE_SECRET, transport=FakePayPal())))


class InvoicingFlowTests(unittest.TestCase):
    def setUp(self):
        self.t = FakePayPal()
        self.c = SandboxClient(FAKE_ID, FAKE_SECRET, transport=self.t)
        self.inv = build_invoice(invoicer_email="merchant@example.com", recipient_email="buyer@example.com",
                                 currency="USD", item_name="Logo design", amount="120.00")

    def test_draft_send_status_remind(self):
        inv_id = invoice_for_decision(go_decision(), self.c, self.inv)
        self.assertEqual(inv_id, "INV2-TEST-0001")
        st = self.c.status(inv_id)
        self.assertEqual(st.simple, "UNPAID")
        self.t.status = "PAID"
        self.assertEqual(self.c.status(inv_id).simple, "PAID")
        self.c.remind(inv_id, "Reminder", "Friendly reminder")
        methods = [(m, u.split("/v")[1]) for m, u, _, _ in self.t.calls]
        self.assertEqual(methods[0], ("POST", "1/oauth2/token"))
        self.assertTrue(all(u.startswith(ALLOWED_PAYPAL_BASE) for _, u, _, _ in self.t.calls))

    def test_invoice_refused_without_verified_go(self):
        d = decide(V.GO, [fact(cid="a")], [Evidence("a", S.UNVERIFIED, R.NO_MATCH)], "mock")
        with self.assertRaises(GoNotAllowed):
            invoice_for_decision(d, self.c, self.inv)
        self.assertEqual(self.t.calls, [])

    def test_smoke_reports_only_status_and_length(self):
        out = smoke_token(self.c)
        self.assertEqual(out, {"http_status": 200, "token_length": 97})

    def test_invalid_invoice_id_rejected(self):
        with self.assertRaises(ValueError):
            self.c.status("../../v1/oauth2/token")


if __name__ == "__main__":
    unittest.main()
