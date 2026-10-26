"""Minimal local UI: `python3.11 -m hfa ui [--mock]`.

Standard library only (http.server), no external JS/CSS, bound to 127.0.0.1
only - never to another interface, and there is no option to change that.
It renders the decision cards produced by hfa.demo: the quoted text, the
source URL, the source SHA-256, the verifier state, the "llm: ..." label and
the raw PayPal invoice status. Mock mode shows a large MOCK banner.
"""
from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

BIND_HOST = "127.0.0.1"

_CSS = """
body{font:15px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}
main{max-width:980px;margin:0 auto;padding:16px}
.banner{font-size:28px;font-weight:800;padding:18px;text-align:center;letter-spacing:.04em}
.mock{background:#ffcc00;color:#000}.unavail{background:#b00020;color:#fff}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;margin:14px 0;padding:14px}
.verdict{font-size:20px;font-weight:700}.GO{color:#116329}.KILL{color:#b00020}.NEEDS_HUMAN{color:#9a6700}
.llm{font-family:ui-monospace,Menlo,monospace;background:#eef;border-radius:4px;padding:2px 6px}
.r{border-top:1px solid #eee;padding:8px 0}.st{font-weight:700}
.VERIFIED{color:#116329}.UNVERIFIED{color:#57606a}.NEEDS_HUMAN_s{color:#9a6700}
blockquote{margin:6px 0;padding:6px 10px;border-left:4px solid #8c959f;background:#f6f8fa}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;word-break:break-all}
.undated{background:#fff3b0;padding:1px 6px;border-radius:4px}
.inv{margin-top:8px;padding:8px;background:#eef6ff;border-radius:6px}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def render(cards: list[dict]) -> str:
    out = ["<!doctype html><html><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Honest Freelance Agent</title><style>{_CSS}</style></head><body>"]
    if any(c.get("llm") == "mock" for c in cards):
        out.append("<div class='banner mock'>MOCK - scripted demo, NOT a model measurement</div>")
    if any(c.get("llm") == "unavailable" for c in cards):
        out.append("<div class='banner unavail'>llm: unavailable - no key; nothing was sent to any model</div>")
    out.append("<main><h1>Honest Freelance Agent</h1>")
    for c in cards:
        final = c.get("final") or c.get("status")
        out.append("<div class='card'>")
        out.append(f"<div><span class='llm'>llm: {_e(c.get('llm'))}</span></div>")
        out.append(f"<p>{_e(c.get('request'))}</p>")
        kc = f" ({_e(c.get('kill_class'))})" if c.get("kill_class") else ""
        out.append(f"<div class='verdict {_e(final)}'>{_e(final)}{kc}</div>")
        if c.get("message"):
            out.append(f"<p>{_e(c['message'])}</p>")
        for r in c.get("receipts", []):
            st = r.get("state")
            cls = "NEEDS_HUMAN_s" if st == "NEEDS_HUMAN" else st
            out.append(f"<div class='r'><span class='st {_e(cls)}'>{_e(st)}</span> "
                       f"<span class='mono'>{_e(r.get('reason'))}</span> - {_e(r.get('kind'))}")
            if r.get("quote"):
                out.append(f"<blockquote>{_e(r['quote'])}</blockquote>")
            elif r.get("text"):
                tick = "user confirmed" if r.get("user_confirmed") else "NOT confirmed by the user"
                out.append(f"<blockquote>{_e(r['text'])} <i>({tick})</i></blockquote>")
            if r.get("source_url"):
                out.append(f"<div>source: <a href='{_e(r['source_url'])}'>{_e(r['source_url'])}</a></div>")
            if r.get("source_sha256"):
                out.append(f"<div class='mono'>source SHA-256: {_e(r['source_sha256'])}"
                           f" &middot; fetched {_e(r.get('fetched_at'))}</div>")
            if r.get("date_flag") == "UNDATED":
                out.append("<span class='undated'>UNDATED source - check it is current</span>")
            if r.get("nearest_span"):
                out.append(f"<div>closest text on page (not a quote): <s>{_e(r['nearest_span'])}</s></div>")
            out.append("</div>")
        for k in c.get("calcs", []):
            out.append(f"<div class='r'>CALC {_e(k.get('claim_id'))}: {_e(k.get('status'))} "
                       f"{_e(k.get('detail'))} <span class='mono'>{_e(json.dumps(k.get('values')))}</span>"
                       f"<br><i>{_e(k.get('derivation'))}</i></div>")
        inv = c.get("invoice")
        if isinstance(inv, dict):
            out.append(f"<div class='inv'>PayPal ({_e(inv.get('paypal'))}) invoice {_e(inv.get('id'))}: "
                       f"<b>{_e(inv.get('status'))}</b> (raw PayPal status)</div>")
        elif inv:
            out.append(f"<div class='inv'>{_e(inv)}</div>")
        out.append("</div>")
    out.append("</main></body></html>")
    return "".join(out)


def make_server(cards_fn: Callable[[], list[dict]], port: int = 8765) -> ThreadingHTTPServer:
    page = render(cards_fn()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((BIND_HOST, port), Handler)


def serve(cards_fn: Callable[[], list[dict]], port: int = 8765) -> None:
    srv = make_server(cards_fn, port)
    print(f"HFA UI on http://{BIND_HOST}:{srv.server_address[1]}/ (local only; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
