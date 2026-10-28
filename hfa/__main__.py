"""CLI.

  python -m hfa verify --quote "..." --url https://...   # fresh fetch + verdict
  python -m hfa verify --quote "..." --snapshot FILE     # offline, from a snapshot
  python -m hfa eval [--snapshots DIR | --live]          # mini-eval (mock proposer)
  python -m hfa eval --replay RUN.json                   # model-free replay of a stored run (E33)
  python -m hfa budget                                   # LLM spend vs. caps
  python -m hfa paypal-smoke                             # sandbox OAuth token only
  python -m hfa demo --mock                              # no keys, llm: mock
  python -m hfa demo [--invoice]                         # judge's own keys from .env
"""
from __future__ import annotations

import argparse
import json
import os
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
    from .proposer.keys import key_file_path
    load_dotenv(key_file_path())
    from .paypal.client import PayPalError
    try:
        out = smoke_token(SandboxClient.from_env())
    except PayPalError as e:
        out = {"http_status": e.status, "token_length": 0}
    print(json.dumps(out))
    return 0 if out.get("token_length") else 1


REMINDER_DIR = os.path.join("data", "reminders")


def _remind(a, ap) -> int:
    """Reminders are never sent without approval: `hfa remind <id>` only saves
    a draft; `hfa remind --approve <id>` sends that saved draft (sandbox)."""
    import re
    inv = a.approve or a.invoice_id
    if not inv or not re.fullmatch(r"[A-Za-z0-9-]{4,64}", inv):
        ap.error("give a valid invoice id")
    path = os.path.join(REMINDER_DIR, f"{inv}.json")
    if not a.approve:
        os.makedirs(REMINDER_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"invoice_id": inv, "subject": a.subject, "note": a.note, "sent": False}, f, indent=1)
        print(json.dumps({"draft_saved": path, "sent": False,
                          "next": f"python3.11 -m hfa remind --approve {inv}"}))
        return 0
    if not os.path.exists(path):
        print(json.dumps({"sent": False, "error": "no saved draft for this invoice; create one first"}))
        return 1
    with open(path, encoding="utf-8") as f:
        draft = json.load(f)
    if draft.get("sent"):
        print(json.dumps({"sent": False, "error": "this draft was already sent"}))
        return 1
    from .demo import load_dotenv
    from .paypal import SandboxClient
    from .proposer.keys import key_file_path
    load_dotenv(key_file_path())
    client = SandboxClient.from_env()
    client.remind(inv, draft["subject"], draft["note"])
    draft["sent"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=1)
    print(json.dumps({"sent": True, "http_status": client.last_status}))
    return 0


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
    for name in ("demo", "ui"):
        dm = sub.add_parser(name)
        dm.add_argument("--mock", action="store_true", help="no keys, scripted proposer (llm: mock)")
        dm.add_argument("--invoice", action="store_true", help="create a SANDBOX invoice if the gate says GO")
        dm.add_argument("--price", help="your job: what it pays (USD)")
        dm.add_argument("--cost", help="your job: your own cost / minimum (USD)")
        dm.add_argument("--platform", default="example",
                        help="your job: example (fictional, mock ok) | freelancer | peopleperhour | gumroad")
        dm.add_argument("--user-confirm", action="store_true",
                        help="YOUR tick: you confirm the numbers you typed (the model can never tick them)")
        if name == "ui":
            dm.add_argument("--port", type=int, default=8765)
    rm = sub.add_parser("remind", help="reminder: save a draft; send only with --approve")
    rm.add_argument("invoice_id", nargs="?")
    rm.add_argument("--subject", default="Friendly reminder")
    rm.add_argument("--note", default="This is a friendly reminder about the open invoice.")
    rm.add_argument("--approve", metavar="INVOICE_ID", help="send the saved draft for this invoice")
    a = ap.parse_args(argv)
    if a.cmd in ("demo", "ui"):
        from . import demo

        def cards():
            if a.price or a.cost:
                if not (a.price and a.cost):
                    ap.error("--price and --cost go together")
                return [demo.run_job(a.price, a.cost, a.platform, a.mock, a.user_confirm, a.invoice)]
            return demo.run_mock() if a.mock else [demo.run_real(a.invoice, user_confirm=a.user_confirm)]

        if a.cmd == "ui":
            from .ui import serve
            serve(cards, a.port)
            return 0
        print(json.dumps(cards(), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "remind":
        return _remind(a, ap)
    if a.cmd == "verify":
        if not (a.url or a.snapshot):
            ap.error("verify needs --url or --snapshot")
        return _verify(a)
    if a.cmd == "budget":
        return _budget(a)
    return _paypal_smoke(a)


if __name__ == "__main__":
    raise SystemExit(main())
