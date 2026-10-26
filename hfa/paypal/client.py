"""PayPal REST client: SANDBOX ONLY, enforced in code.

`assert_sandbox(url)` requires scheme == "https" and host ==
"api-m.sandbox.paypal.com" by exact comparison (not endswith / contains), no
userinfo, default port. It runs (1) on the base URL read from the
environment at startup, (2) in the constructor and (3) before every single
request. There is no flag that disables it.

Invoicing v2 flow: create draft -> send -> get status (UNPAID/PAID) -> remind.
Every side-effecting call requires a gate decision that passes
`require_verified_go()`.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from ..gate import require_verified_go
from ..models import Decision

SANDBOX_HOST = "api-m.sandbox.paypal.com"
ALLOWED_PAYPAL_BASE = f"https://{SANDBOX_HOST}"
REFUSAL_MESSAGE = "I will not call the live PayPal API. Sandbox only."


class LiveHostRefused(Exception):
    """Raised for any PayPal URL that is not the sandbox API host."""


def assert_sandbox(url: str) -> str:
    try:
        p = urllib.parse.urlsplit(url)
        port = p.port
    except ValueError:
        raise LiveHostRefused(f"{REFUSAL_MESSAGE} (unparseable URL)") from None
    if p.scheme != "https" or p.hostname != SANDBOX_HOST or p.netloc != SANDBOX_HOST \
            or port not in (None,) or p.username or p.password:
        raise LiveHostRefused(f"{REFUSAL_MESSAGE} (refused host: {p.netloc or url!r})")
    return url


# transport(method, url, headers, body_bytes) -> (status, body_bytes)
Transport = Callable[[str, str, dict, Optional[bytes]], tuple[int, bytes]]


def urllib_transport(method: str, url: str, headers: dict, body: Optional[bytes]) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class PayPalError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"PayPal HTTP {status}: {message}")
        self.status = status


@dataclass
class InvoiceStatus:
    invoice_id: str
    raw_status: str

    @property
    def simple(self) -> str:
        if self.raw_status in ("PAID", "MARKED_AS_PAID"):
            return "PAID"
        if self.raw_status == "DRAFT":
            return "DRAFT"
        if self.raw_status in ("CANCELLED", "REFUNDED", "MARKED_AS_REFUNDED"):
            return "CLOSED"
        return "UNPAID"   # SENT, UNPAID, PARTIALLY_PAID, PAYMENT_PENDING, SCHEDULED


class SandboxClient:
    def __init__(self, client_id: str, client_secret: str, base_url: str = ALLOWED_PAYPAL_BASE,
                 transport: Transport = urllib_transport):
        self.base_url = assert_sandbox(base_url.rstrip("/"))
        self._client_id = client_id
        self._client_secret = client_secret
        self._transport = transport
        self._token: Optional[str] = None

    def __repr__(self) -> str:
        return f"SandboxClient(base_url={self.base_url!r}, credentials=[hidden])"

    @classmethod
    def from_env(cls, transport: Transport = urllib_transport) -> "SandboxClient":
        base = os.environ.get("PAYPAL_BASE_URL", ALLOWED_PAYPAL_BASE)
        assert_sandbox(base)            # startup check, before anything else
        cid = os.environ.get("PAYPAL_CLIENT_ID", "")
        sec = os.environ.get("PAYPAL_CLIENT_SECRET") or os.environ.get("PAYPAL_SECRET", "")
        if not cid or not sec:
            raise ValueError("set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET (sandbox app)")
        return cls(cid, sec, base, transport)

    # -- low level ----------------------------------------------------------
    def _request(self, method: str, path: str, *, body=None, form: Optional[dict] = None,
                 auth_basic: bool = False, extra_headers: Optional[dict] = None) -> tuple[int, dict]:
        url = assert_sandbox(self.base_url + path)   # before EVERY request
        headers = {"Accept": "application/json"}
        data: Optional[bytes] = None
        if auth_basic:
            cred = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {cred}"
        else:
            headers["Authorization"] = f"Bearer {self.token()}"
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = urllib.parse.urlencode(form).encode()
        elif body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        headers.update(extra_headers or {})
        status, raw = self._transport(method, url, headers, data)
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {"_raw": raw[:200].decode("utf-8", "replace")}
        if status >= 400:
            msg = parsed.get("message") or parsed.get("error_description") or parsed.get("name") or ""
            raise PayPalError(status, str(msg))
        return status, parsed

    def fetch_token(self) -> tuple[int, str]:
        status, data = self._request("POST", "/v1/oauth2/token",
                                     form={"grant_type": "client_credentials"}, auth_basic=True)
        tok = data.get("access_token", "")
        if not tok:
            raise PayPalError(status, "no access_token in response")
        self._token = tok
        return status, tok

    def token(self) -> str:
        if not self._token:
            self.fetch_token()
        return self._token  # type: ignore[return-value]

    # -- Invoicing v2 -------------------------------------------------------
    def create_draft(self, invoice: dict) -> str:
        _, data = self._request("POST", "/v2/invoicing/invoices", body=invoice,
                                extra_headers={"Prefer": "return=representation"})
        inv_id = data.get("id") or data.get("href", "").rstrip("/").split("/")[-1]
        if not inv_id:
            raise PayPalError(201, "draft created but no id returned")
        return inv_id

    def send(self, invoice_id: str, note: str = "") -> None:
        body = {"send_to_invoicer": False, "send_to_recipient": True}
        if note:
            body["note"] = note
        self._request("POST", f"/v2/invoicing/invoices/{_safe_id(invoice_id)}/send", body=body)

    def status(self, invoice_id: str) -> InvoiceStatus:
        _, data = self._request("GET", f"/v2/invoicing/invoices/{_safe_id(invoice_id)}")
        return InvoiceStatus(invoice_id, str(data.get("status", "UNKNOWN")))

    def remind(self, invoice_id: str, subject: str, note: str) -> None:
        self._request("POST", f"/v2/invoicing/invoices/{_safe_id(invoice_id)}/remind",
                      body={"subject": subject, "note": note,
                            "send_to_invoicer": False, "send_to_recipient": True})


def _safe_id(invoice_id: str) -> str:
    if not invoice_id or not all(c.isalnum() or c == "-" for c in invoice_id):
        raise ValueError("invalid invoice id")
    return invoice_id


def build_invoice(*, invoicer_email: str, recipient_email: str, currency: str,
                  item_name: str, amount: str, note: str = "", invoice_number: Optional[str] = None) -> dict:
    """Minimal Invoicing v2 payload (sandbox personas only)."""
    detail = {"currency_code": currency, "note": note}
    if invoice_number:
        detail["invoice_number"] = invoice_number
    return {
        "detail": detail,
        "invoicer": {"email_address": invoicer_email},
        "primary_recipients": [{"billing_info": {"email_address": recipient_email}}],
        "items": [{"name": item_name, "quantity": "1",
                   "unit_amount": {"currency_code": currency, "value": amount}}],
    }


def invoice_for_decision(decision: Decision, client: SandboxClient, invoice: dict) -> str:
    """The only path from a decision to money: GO is re-checked from evidence."""
    require_verified_go(decision)
    inv_id = client.create_draft(invoice)
    client.send(inv_id)
    return inv_id


def smoke_token(client: SandboxClient) -> dict:
    """Sandbox smoke test. Reports only the HTTP status and token length."""
    status, tok = client.fetch_token()
    return {"http_status": status, "token_length": len(tok)}
