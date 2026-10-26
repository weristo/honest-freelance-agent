"""One-command local demo.

  python -m hfa demo --mock      no keys, no network. "llm: mock": a scripted
                                 proposer and a fictional terms page bundled in
                                 hfa/demo_data/. PayPal is an in-memory fake.
                                 This shows the pipeline; it measures nothing.

  python -m hfa demo [--invoice] the judge's own keys from .env: a real LLM
                                 proposer (HFA_LLM_PROVIDER, default model
                                 claude-haiku-4-5, budget-capped), live fetch of
                                 the real source, verifier + gate, and with
                                 --invoice a PayPal SANDBOX invoice (draft ->
                                 send -> status) when, and only when, the gate
                                 says GO.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .fetcher import LiveFetcher, SourceDoc, doc_from_bytes
from .paypal.client import SandboxClient, build_invoice, invoice_for_decision
from .pipeline import RunResult, run
from .proposer.base import Proposer
from .proposer.budget import Budget
from .proposer.providers import MockClient, client_from_env

MOCK_BANNER = "llm: mock - scripted demo of the pipeline, NOT a model measurement"
DEMO_URL = "https://marketplace.example/legal/terms"          # fictional (.example)
_HERE = os.path.dirname(os.path.abspath(__file__))

REAL_REQUEST = ("I want to enter the PayPal AI Hackathon with an agent that only uses the "
                "PayPal sandbox for invoicing. Is that allowed?")
REAL_SOURCES = ["https://paypalaihackathon.devpost.com/rules"]


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides the real environment;
    never prints values."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


# --- mock mode ----------------------------------------------------------------

def _claim(cid, quote, polarity="SUPPORTS", kind="FACT"):
    return {"claim_id": cid, "text": quote, "kind": kind, "critical": True, "quote": quote,
            "source_url": DEMO_URL, "rule_id": "EXAMPLE-TERMS", "polarity": polarity}


def _user(cid, text):
    return {"claim_id": cid, "text": text, "kind": "USER_ATTESTED", "critical": True,
            "polarity": "SUPPORTS", "user_confirmed": True}


def _net_calc(cid, amount, minimum, claimed_net):
    return {"claim_id": cid, "text": "net after the platform fee vs. my minimum", "kind": "CALC",
            "critical": True, "polarity": "SUPPORTS", "calc": {
                "inputs": {"rate": {"literal": "8%", "source": "c2"},
                           "amount": {"literal": amount, "source": "u1"},
                           "minimum": {"literal": minimum, "source": "u1"}},
                "steps": [["fee", "rate * amount"], ["net", "amount - fee"]],
                "claimed": {"net": claimed_net}, "check": ["net", ">=", "minimum"], "on_fail": "NOT_WORTH"}}


FEE_QUOTE = "For fixed price projects the freelancer service fee is 8% of the project amount."


MOCK_SCENARIOS = [
    ("Invoice my client for the finished logo milestone (USD 120) through PayPal.",
     {"decision": "GO", "rationale": "Invoicing completed milestones is allowed.", "claims": [
         _claim("c1", "Freelancers may invoice clients for completed milestones through a supported payment provider."),
         _claim("c2", FEE_QUOTE),
         _user("u1", "The milestone is 120 USD and I need at least 100 USD net."),
         _net_calc("k1", "120", "100", "110.40")]}),
    ("Write a bot that bids on every new project for me.",
     {"decision": "GO", "rationale": "(the mock pretends to be over-eager)", "claims": [
         _claim("c1", "You may not use automated software to submit bids or proposals on behalf of other users.",
                "CONTRADICTS")]}),
    ("The marketplace only takes 5%, so price the quote on that.",
     {"decision": "GO", "rationale": "(the mock paraphrases a fee from memory)", "claims": [
         _claim("c1", "The freelancer service fee is only 5% for fixed price projects.")]}),
    ("A 30 USD fixed price job; my own cost is 29 USD. Take it?",
     {"decision": "GO", "rationale": "(the mock says GO; the code does the sum)", "claims": [
         _claim("c2", FEE_QUOTE),
         _user("u1", "The job pays 30 USD and my own cost is 29 USD."),
         _net_calc("k1", "30", "29", "27.60")]}),
]


class _DemoFixture:
    def __init__(self):
        with open(os.path.join(_HERE, "demo_data", "example_terms.html"), "rb") as f:
            self.raw = f.read()

    def fetch(self, url: str) -> SourceDoc:
        return doc_from_bytes(url, url, 200, "text/html; charset=utf-8", self.raw, "2026-10-26T00:00:00Z")


class _FakeSandbox:
    """In-memory PayPal stand-in for mock mode (no network)."""

    def __init__(self):
        self.state = "DRAFT"

    def __call__(self, method, url, headers, body):
        if url.endswith("/v1/oauth2/token"):
            return 200, b'{"access_token": "mock-token"}'
        if url.endswith("/v2/invoicing/invoices") and method == "POST":
            return 201, b'{"id": "INV2-MOCK-0001", "status": "DRAFT"}'
        if url.endswith("/send"):
            self.state = "UNPAID"
            return 202, b""
        return 200, json.dumps({"id": "INV2-MOCK-0001", "status": self.state}).encode()


def card(res: RunResult) -> dict:
    """Decision card as JSON (what the UI will render)."""
    d = res.decision
    out = {"llm": res.llm, "status": res.status, "message": res.message}
    if d:
        out.update(final=d.final.value, kill_class=d.kill_class.value if d.kill_class else None,
                   proposed=d.proposed.value, gate_log=d.gate_log,
                   calcs=[dict(c, derivation="INFERENCE - reading the formula from the quote is reasoning, "
                                             "not verified; code guarantees only the arithmetic")
                          for c in d.calcs],
                   receipts=[{"claim": e.claim_id, "state": e.state.value, "reason": e.reason_code.value,
                              "layer": e.match_layer, "snippet_sha256": (e.snippet_sha256 or "")[:12],
                              "fetched_at": e.fetched_at, "nearest_span": e.nearest_span,
                              "guard_hits": e.guard_hits} for e in d.evidence])
    return out


def run_mock() -> list[dict]:
    fixture = _DemoFixture()
    cards = []
    for request, proposal in MOCK_SCENARIOS:
        client = MockClient(lambda s, u, p=proposal: json.dumps(p))
        res = run(request, [DEMO_URL], Proposer(client), fixture)
        c = {"banner": MOCK_BANNER, "request": request, **card(res)}
        if res.decision and res.decision.final.value == "GO":
            pp = SandboxClient("mock-id", "mock-secret", transport=_FakeSandbox())
            inv = build_invoice(invoicer_email="merchant@example.com", recipient_email="buyer@example.com",
                                currency="USD", item_name="Logo milestone", amount="120.00")
            inv_id = invoice_for_decision(res.decision, pp, inv)
            c["invoice"] = {"paypal": "mock (no network)", "id": inv_id, "status": pp.status(inv_id).simple}
        cards.append(c)
    return cards


# --- real mode ----------------------------------------------------------------

def run_real(invoice: bool, request: Optional[str] = None, sources: Optional[list[str]] = None) -> dict:
    load_dotenv()
    if os.environ.get("HFA_LLM_PROVIDER", "mock").lower() == "mock":
        raise SystemExit("set HFA_LLM_PROVIDER=anthropic (and HFA_API_KEY) in .env, or use --mock")
    proposer = Proposer(client_from_env(), Budget())
    res = run(request or REAL_REQUEST, sources or REAL_SOURCES, proposer, LiveFetcher(),
              trace_path=os.path.join("data", "traces", "demo.jsonl"))
    c = card(res)
    if invoice:
        if not res.decision or res.decision.final.value != "GO":
            c["invoice"] = "not created: the gate did not say GO"
        else:
            pp = SandboxClient.from_env()
            inv = build_invoice(invoicer_email=os.environ["HFA_SANDBOX_MERCHANT_EMAIL"],
                                recipient_email=os.environ["HFA_SANDBOX_BUYER_EMAIL"],
                                currency="USD", item_name="Hackathon demo milestone", amount="10.00")
            inv_id = invoice_for_decision(res.decision, pp, inv)
            c["invoice"] = {"paypal": "sandbox", "id": inv_id, "status": pp.status(inv_id).simple}
    return c
