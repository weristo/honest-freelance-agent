"""HTML -> text layers.

visible  : rendered-ish text. <script>, <style>, <noscript>, <svg>, <template>
           are dropped. A separator (newline) is inserted only at block-level
           elements, never between inline elements -- inserting it everywhere
           breaks real quotes such as "Evectv.[3] szerinti".
embedded : string payloads that pages ship but do not render (collapsed
           sections): Next.js `self.__next_f.push` chunks, `__NEXT_DATA__`
           and `application/ld+json`. A match here is VERIFIED but flagged as
           "found in page data" in the UI.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser

_SKIP = {"script", "style", "noscript", "svg", "template"}
_BLOCK = {
    "address", "article", "aside", "blockquote", "br", "dd", "details", "div",
    "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1",
    "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
    "p", "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "ul", "title", "body", "html", "head",
}
_NEXT_F = re.compile(r"self\.__next_f\.push\(\s*(\[.*?\])\s*\)\s*;?\s*$", re.S)


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.embedded: list[str] = []
        self._skip_depth = 0
        self._script_attrs: dict[str, str] | None = None
        self._script_buf: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip_depth += 1
            if tag == "script":
                self._script_attrs = {k: (v or "") for k, v in attrs}
                self._script_buf = []
            return
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK and not self._skip_depth:
            self.visible.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK and not self._skip_depth:
            self.visible.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            if tag == "script" and self._script_attrs is not None:
                self._handle_script("".join(self._script_buf), self._script_attrs)
                self._script_attrs = None
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK and not self._skip_depth:
            self.visible.append("\n")

    def handle_data(self, data):
        if self._script_attrs is not None:
            self._script_buf.append(data)
            return
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        self.visible.append(data)

    def _handle_script(self, body: str, attrs: dict[str, str]) -> None:
        stype = attrs.get("type", "").lower()
        if attrs.get("id") == "__NEXT_DATA__" or stype == "application/ld+json":
            try:
                self.embedded.extend(_json_strings(json.loads(body)))
            except ValueError:
                pass
            return
        m = _NEXT_F.search(body.strip())
        if m:
            try:
                arr = json.loads(m.group(1))
            except ValueError:
                return
            self.embedded.extend(s for s in arr if isinstance(s, str))


def _json_strings(obj) -> list[str]:
    out: list[str] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            stack.extend(reversed(list(o.values())))
        elif isinstance(o, list):
            stack.extend(reversed(o))
    return out


def extract_html(html_text: str) -> tuple[str, str, str]:
    """Return (visible, embedded, title). Not yet normalized."""
    p = _Parser()
    p.feed(html_text)
    p.close()
    return "".join(p.visible), "\n".join(p.embedded), p.title.strip()


# --- page-type heuristics (pattern lists; heuristics, not proofs) ----------

SOFT_404_PATTERNS = [
    "a keresett oldal nem található",
    "az ön által kért oldal vagy dokumentum nem található",
    "page not found",
    "404 not found",
    "the page you requested could not be found",
    "seite nicht gefunden",
    "die angeforderte seite wurde nicht gefunden",
]

BLOCK_PATTERNS = [
    "just a moment...",
    "checking your browser before accessing",
    "sorry, you have been blocked",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
    "verify you are human",
    "please complete the security check",
]

LOGIN_PATH_MARKERS = ("/sign-in", "/signin", "/login", "/log-in", "/auth/")
