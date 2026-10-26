"""CLI.

  python -m hfa verify --quote "..." --url https://...   # fresh fetch + verdict
  python -m hfa verify --quote "..." --snapshot FILE     # offline, from a snapshot
  python -m hfa eval [--snapshots DIR | --live]          # mini-eval (mock proposer)
  python -m hfa budget                                   # LLM spend vs. caps
  python -m hfa paypal-smoke                             # sandbox OAuth token only
  python -m hfa demo --mock                              # no keys, llm: mock
  python -m hfa demo [--invoice]                         # judge's own keys from .env
"""
from __future__ import annotations

import argparse
import json
import sys

from .fetcher import FixtureFetcher, LiveFetcher, load_snapshot
from .models import Claim, Polarity


def _verify(a) -> int:
    from .verifier import verify
    if a.snapshot:
        doc = load_snapshot(a.snapshot)
        url = a.url or doc.url
        fetcher = FixtureFetcher({url: a.snapshot})
    else:
        url, fetcher = a.url, LiveFetcher()
    claim = Claim("c_cli", a.quote, quote=a.quote, source_url=url,
                  polarity=Polarity(a.polarity), source_domain_type=a.domain_type)
    ev = verify(claim, fetcher).to_dict()
    ev["llm"] = "none (verifier only)"
    print(json.dumps(ev, indent=1, ensure_ascii=False))
    return 0


def _budget(_a) -> int:
    from .proposer.budget import Budget
    s = Budget().status()
    print(json.dumps({"day": s.day, "spent_today_usd": s.spent_today, "daily_cap_usd": s.daily_cap,
                      "spent_total_usd": s.spent_total, "total_cap_usd": s.total_cap,
                      "state": "budget_exhausted" if s.exhausted else "ok"}, indent=1))
    return 0


def _paypal_smoke(_a) -> int:
    from .demo import load_dotenv
    from .paypal import SandboxClient, smoke_token
    load_dotenv()
    from .paypal.client import PayPalError
    try:
        out = smoke_token(SandboxClient.from_env())
    except PayPalError as e:
        out = {"http_status": e.status, "token_length": 0}
    print(json.dumps(out))
    return 0 if out.get("token_length") else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        from .evalrun import main as eval_main
        return eval_main(argv[1:])
    ap = argparse.ArgumentParser(prog="hfa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--quote", required=True)
    v.add_argument("--url")
    v.add_argument("--snapshot")
    v.add_argument("--polarity", default="SUPPORTS", choices=["SUPPORTS", "CONTRADICTS"])
    v.add_argument("--domain-type", default="general", choices=["general", "legal"])
    sub.add_parser("eval", help="mini-eval; see python -m hfa eval --help")
    sub.add_parser("budget")
    sub.add_parser("paypal-smoke")
    dm = sub.add_parser("demo")
    dm.add_argument("--mock", action="store_true", help="no keys, scripted proposer (llm: mock)")
    dm.add_argument("--invoice", action="store_true", help="create a SANDBOX invoice if the gate says GO")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        from . import demo
        out = demo.run_mock() if a.mock else demo.run_real(a.invoice)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "verify":
        if not (a.url or a.snapshot):
            ap.error("verify needs --url or --snapshot")
        return _verify(a)
    if a.cmd == "budget":
        return _budget(a)
    return _paypal_smoke(a)


if __name__ == "__main__":
    raise SystemExit(main())
