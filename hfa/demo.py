"""One-command local demo.

  python -m hfa demo --mock      no keys, no network. "llm: mock": a scripted
                                 proposer and a fictional terms page bundled in
                                 hfa/demo_data/. PayPal is an in-memory fake.
                                 This shows the pipeline; it measures nothing.

  python -m hfa demo [--invoice] the judge's own keys from the key file
                                 (HFA_ENV_FILE, or ./.env): a real LLM proposer
                                 (default model claude-haiku-4-5, budget-capped;
                                 no key -> "llm: unavailable", never a silent
                                 mock), live fetch of the real source, verifier +
                                 gate, and with --invoice a PayPal SANDBOX invoice
                                 (draft -> send -> status) when, and only when,
                                 the gate says GO.

  python -m hfa demo --price 30 --cost 26 --platform freelancer [--user-confirm]
                                 your own job: "is it worth it?". --user-confirm
                                 is YOUR tick on the numbers you typed (the model
                                 can never tick them for you).
"""
from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation
from typing import Optional

from .fetcher import LiveFetcher, SourceDoc, doc_from_bytes
from .normalize import norm
from .paypal.client import SandboxClient, build_invoice, invoice_for_decision
from .pipeline import RunResult, run
from .proposer.base import Proposer
from .proposer.budget import Budget
from .proposer.keys import KEY_NAMES, key_file_path
from .proposer.providers import MockClient, client_from_env

MOCK_BANNER = "llm: mock - scripted demo of the pipeline, NOT a model measurement"
DEMO_URL = "https://marketplace.example/legal/terms"          # fictional (.example)
_HERE = os.path.dirname(os.path.abspath(__file__))

REAL_REQUEST = ("I want to enter the PayPal AI Hackathon with an agent that only uses the "
                "PayPal sandbox for invoicing. Is that allowed?")
REAL_SOURCES = ["https://paypalaihackathon.devpost.com/rules"]

# Platforms for --platform. "example" is the fictional bundled page (works in
# mock mode); real platforms need real mode (the mock never quotes a real site).
PLATFORMS = {
    "example": DEMO_URL,
    "freelancer": "https://www.freelancer.com/feesandcharges",
    "peopleperhour": "https://www.peopleperhour.com/static/terms",
    "gumroad": "https://gumroad.com/pricing",
}
DERIVATION = ("INFERENCE - reading the formula from the quote is reasoning, not verified; "
              "code guarantees only the arithmetic")


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
            if k in KEY_NAMES.values():
                continue            # LLM keys stay in the file; they are never exported
            if k and v and k not in os.environ:
                os.environ[k] = v


# --- mock scenarios -----------------------------------------------------------

def _claim(cid, quote, polarity="SUPPORTS", kind="FACT"):
    return {"claim_id": cid, "text": quote, "kind": kind, "critical": True, "quote": quote,
            "source_url": DEMO_URL, "rule_id": "EXAMPLE-TERMS", "polarity": polarity}


def _user(cid, text):
    # No "user_confirmed" here: the tick comes from the user's input, not the proposal.
    return {"claim_id": cid, "text": text, "kind": "USER_ATTESTED", "critical": True, "polarity": "SUPPORTS"}


def _net_calc(cid, amount, minimum, claimed_net):
    return {"claim_id": cid, "text": "net after the platform fee vs. my minimum", "kind": "CALC",
            "critical": True, "polarity": "SUPPORTS", "calc": {
                "inputs": {"rate": {"literal": "8%", "source": "c2"},
                           "amount": {"literal": amount, "source": "u1"},
                           "minimum": {"literal": minimum, "source": "u1"}},
                "steps": [["fee", "rate * amount"], ["net", "amount - fee"]],
                "claimed": {"net": claimed_net}, "check": ["net", ">=", "minimum"], "on_fail": "NOT_WORTH"}}


FEE_QUOTE = "For fixed price projects the freelancer service fee is 8% of the project amount."

# (request, scripted proposal, scripted user tick)
MOCK_SCENARIOS = [
    ("Invoice my client for the finished logo milestone (USD 120) through PayPal.",
     {"decision": "GO", "rationale": "Invoicing completed milestones is allowed.", "claims": [
         _claim("c1", "Freelancers may invoice clients for completed milestones through a supported payment provider."),
         _claim("c2", FEE_QUOTE),
         _user("u1", "The milestone is 120 USD and I need at least 100 USD net."),
         _net_calc("k1", "120", "100", "110.40")]}, True),
    ("Write a bot that bids on every new project for me.",
     {"decision": "GO", "rationale": "(the mock pretends to be over-eager)", "claims": [
         _claim("c1", "You may not use automated software to submit bids or proposals on behalf of other users.",
                "CONTRADICTS")]}, False),
    ("The marketplace only takes 5%, so price the quote on that.",
     {"decision": "GO", "rationale": "(the mock paraphrases a fee from memory)", "claims": [
         _claim("c1", "The freelancer service fee is only 5% for fixed price projects.")]}, False),
    ("A 30 USD fixed price job; my own cost is 29 USD. Take it?",
     {"decision": "GO", "rationale": "(the mock says GO; the code does the sum)", "claims": [
         _claim("c2", FEE_QUOTE),
         _user("u1", "The job pays 30 USD and my own cost is 29 USD."),
         _net_calc("k1", "30", "29", "27.60")]}, True),
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


# --- cards --------------------------------------------------------------------

def card(res: RunResult) -> dict:
    """Decision card as JSON (rendered by `hfa ui`, printed by `hfa demo`)."""
    d = res.decision
    out = {"llm": res.llm, "status": res.status, "message": res.message}
    if d:
        claims = {c.claim_id: c for c in d.claims}
        receipts = []
        for e in d.evidence:
            c = claims.get(e.claim_id)
            receipts.append({
                "claim": e.claim_id, "kind": c.kind.value if c else None,
                "quote": c.quote if c else None, "text": c.text if c else None,
                "source_url": c.source_url if c else None,
                "user_confirmed": c.user_confirmed if c else None,
                "state": e.state.value, "reason": e.reason_code.value, "layer": e.match_layer,
                "source_sha256": e.raw_sha256, "snippet_sha256": e.snippet_sha256,
                "fetched_at": e.fetched_at, "nearest_span": e.nearest_span,
                "source_date": e.source_date, "date_flag": e.date_flag, "guard_hits": e.guard_hits})
        out.update(final=d.final.value, kill_class=d.kill_class.value if d.kill_class else None,
                   proposed=d.proposed.value, gate_log=d.gate_log,
                   calcs=[dict(c, derivation=DERIVATION) for c in d.calcs], receipts=receipts)
    return out


def _mock_invoice(res: RunResult, amount: str) -> dict:
    pp = SandboxClient("mock-id", "mock-secret", transport=_FakeSandbox())
    inv = build_invoice(recipient_email="buyer@example.com", currency="USD",
                        item_name="Milestone", amount=amount)
    inv_id = invoice_for_decision(res.decision, pp, inv)
    return {"paypal": "mock (no network)", "id": inv_id, "status": pp.status(inv_id).display}


def run_mock() -> list[dict]:
    fixture = _DemoFixture()
    cards = []
    for request, proposal, tick in MOCK_SCENARIOS:
        client = MockClient(lambda s, u, p=proposal: json.dumps(p))
        res = run(request, [DEMO_URL], Proposer(client, user_confirm=tick), fixture)
        c = {"banner": MOCK_BANNER, "request": request,
             "user_tick": "scripted user input (mock demo)" if tick else "none", **card(res)}
        if res.decision and res.decision.final.value == "GO":
            c["invoice"] = _mock_invoice(res, "120.00")
        cards.append(c)
    return cards


# --- your own job (--price/--cost/--platform) ---------------------------------

def _num(s: str) -> Decimal:
    try:
        v = Decimal(s)
    except InvalidOperation:
        raise SystemExit(f"not a number: {s!r}") from None
    if v <= 0 or v > Decimal("1000000"):
        raise SystemExit(f"out of range: {s!r}")
    return v


def job_request(price: str, cost: str, platform: str) -> str:
    _num(price), _num(cost)
    return (f"A fixed price job on {platform} that pays {price} USD. "
            f"My own cost is {cost} USD. Is it worth taking?")


def run_job(price: str, cost: str, platform: str, mock: bool, user_confirm: bool,
            invoice: bool = False) -> dict:
    if platform not in PLATFORMS:
        raise SystemExit(f"unknown platform {platform!r}; choose one of {sorted(PLATFORMS)}")
    request = job_request(price, cost, platform)
    url = PLATFORMS[platform]
    if mock:
        if platform != "example":
            raise SystemExit("mock mode only uses the fictional 'example' platform "
                             "(the mock never quotes a real website)")
        net = (_num(price) * Decimal("0.92")).quantize(Decimal("0.01"))
        proposal = {"decision": "GO", "rationale": "(the mock says GO; the code does the sum)", "claims": [
            _claim("c2", FEE_QUOTE), _user("u1", request), _net_calc("k1", price, cost, str(net))]}
        client = MockClient(lambda s, u: json.dumps(proposal))
        res = run(request, [url], Proposer(client, user_confirm=user_confirm), _DemoFixture())
        c = {"banner": MOCK_BANNER, "request": request,
             "user_tick": "given (--user-confirm)" if user_confirm else "missing (pass --user-confirm)",
             **card(res)}
        if invoice and res.decision and res.decision.final.value == "GO":
            c["invoice"] = _mock_invoice(res, str(_num(price)))
        return c
    return run_real(invoice, request=request, sources=[url], user_confirm=user_confirm, amount=price)


# --- real mode ----------------------------------------------------------------

def _live_text(fetcher: LiveFetcher):
    def get(url: str) -> str:
        doc = fetcher.fetch(url)
        return norm(doc.visible + " " + doc.embedded)
    return get


def run_real(invoice: bool, request: Optional[str] = None, sources: Optional[list[str]] = None,
             user_confirm: bool = False, amount: Optional[str] = None) -> dict:
    load_dotenv(key_file_path())   # PayPal settings (the LLM key is read separately)
    trace = os.path.join("data", "traces", "demo.jsonl")
    fetcher = LiveFetcher()
    proposer = Proposer(client_from_env(), Budget(), usage_log=trace, user_confirm=user_confirm,
                        source_text=_live_text(fetcher))
    res = run(request or REAL_REQUEST, sources or REAL_SOURCES, proposer, fetcher, trace_path=trace)
    c = card(res)
    c["request"] = request or REAL_REQUEST
    c["user_tick"] = "given (--user-confirm)" if user_confirm else "none"
    if res.status == "unavailable":
        c["banner"] = "llm: unavailable - no key file / no key; nothing was sent to any model"
    c["usage"] = getattr(proposer, "last_usage", None)
    c["source_context"] = proposer.last_context
    c["quotes_outside_context"] = proposer.last_outside
    if invoice:
        if not res.decision or res.decision.final.value != "GO":
            c["invoice"] = "not created: the gate did not say GO"
        else:
            pp = SandboxClient.from_env()
            inv = build_invoice(invoicer_email=os.environ.get("HFA_SANDBOX_MERCHANT_EMAIL") or None,
                                recipient_email=os.environ["HFA_SANDBOX_BUYER_EMAIL"],
                                currency="USD", item_name="Freelance milestone",
                                amount=str(_num(amount)) if amount else "10.00")
            inv_id = invoice_for_decision(res.decision, pp, inv)
            c["invoice"] = {"paypal": "sandbox", "id": inv_id, "status": pp.status(inv_id).display}
    return c
