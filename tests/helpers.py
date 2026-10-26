"""Synthetic fixtures. All text here is written for the tests (no third-party text)."""
from hfa.fetcher import SourceDoc
from hfa.models import Claim, ClaimKind, Polarity

FILLER = " ".join(f"Section {i} describes general marketplace behaviour in plain words." for i in range(60))
RULE = "Every submission must include a working build that judges can run themselves."
FEE = "The freelancer fee for fixed price projects is 10% or $5.00 USD, whichever is greater."


def page(text: str, **kw) -> SourceDoc:
    d = dict(url="https://example.org/rules", final_url="https://example.org/rules", http_status=200,
             content_type="text/html", raw_sha256="ab" * 32, fetched_at="2026-10-26T10:00:00Z",
             visible=FILLER + " " + text + " " + FILLER, embedded="")
    d.update(kw)
    return SourceDoc(**d)


class StaticFetcher:
    def __init__(self, doc: SourceDoc):
        self.doc = doc
        self.calls = 0

    def fetch(self, url):
        self.calls += 1
        return self.doc


def fact(quote=RULE, polarity=Polarity.SUPPORTS, cid="c1", **kw) -> Claim:
    return Claim(claim_id=cid, text=quote, kind=kw.pop("kind", ClaimKind.FACT), quote=quote,
                 source_url=kw.pop("source_url", "https://example.org/rules"), polarity=polarity, **kw)
