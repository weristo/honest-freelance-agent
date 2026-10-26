# Honest Freelance Agent (HFA)

An agent that helps a freelancer decide whether to take a job, and then bills
it through PayPal Invoicing (sandbox). The LLM **proposes** a GO / KILL
decision and cites the platform rules it relies on. Deterministic code
**verifies** every cited quote against a freshly fetched copy of the source,
and only a decision whose critical facts are all `VERIFIED` can create an
invoice.

> Built by AI agents (Claude) under the supervision of Otto Lőkös.

Status: early build (hackathon), local runs only (no hosted demo). The
verifier, gate, budget guard, PayPal sandbox client and mini-eval run and
are tested. The real-LLM path is implemented and wired, but no model has
been measured yet: every number in this README comes from the **mock**
proposer. The UI is not built yet; the demo prints decision cards as JSON.

## Why

Freelancers lose money by applying a platform rule from memory: a fee that
changed, a minimum payout, a clause that forbids working off-platform. LLMs
do the same thing, confidently. HFA does not ask you to trust the model: it
shows the exact sentence, where it was found, its SHA-256 and the fetch
time, or it says plainly that it could not check.

## Architecture

```
 request (job / quote / invoice step) + source URLs
   |
   v
 [1] Proposer (LLM, or mock)  -> proposed GO/KILL/NEEDS_HUMAN + claims[]
   |                              (claim = verbatim quote + source_url + rule_id)
   v
 [2] Fetcher (SSRF-guarded, fresh fetch, robots.txt honoured)
   |
   v
 [3] Extractor -> visible text + embedded page data -> Normalizer
   |
   v
 [4] Verifier (deterministic) -> Evidence per claim:
   |     VERIFIED / UNVERIFIED / NEEDS_HUMAN + reason code + hashes
   v
 [5] Contradiction pass (same topic, opposite polarity -> NEEDS_HUMAN)
   |
   v
 [6] Gate G1-G5 (deterministic) -> final GO / KILL / NEEDS_HUMAN
   |
   v
 [7] PayPal Invoicing v2, SANDBOX ONLY: draft -> send -> status -> remind
       (require_verified_go() re-checks the evidence before any call)
```

Gate rules: **G1** a verified critical fact that contradicts the plan -> KILL.
**G2** GO only if every critical fact is VERIFIED and there is at least one.
**G3** any critical NEEDS_HUMAN, or a decision resting on an INFERENCE ->
NEEDS_HUMAN. **G4** any critical UNVERIFIED -> no GO. **G5** facts only the
user knows (USER_ATTESTED) are never VERIFIED and need an explicit tick.

Reason codes: `EXACT_MATCH`, `NO_MATCH`, `QUOTE_TOO_SHORT`, `URL_REFUSED`,
`FETCH_FAILED`, `FETCH_BLOCKED` (403/429, captcha, login wall, robots.txt -
never bypassed), `SOFT_404`, `THIN_CONTENT` (JS-only page), `MULTI_MATCH`
(several places, and one of them has a guard word or the places show
different amounts), `MULTI_CONSISTENT` (several places that agree: VERIFIED,
all snippets kept), `REDIRECTED` (other domain), `STALE_SOURCE` (legal page
older than 365 days), `AMBIGUOUS_CONTEXT` (a guard word such as "except",
"not", "employer" in the sentence of a supporting quote or the sentence right
after it; words inside the quote do not count), `CONTRADICTED`,
`SOURCE_CHANGED`.

REFUSE vs KILL is decided by a small source registry in code
(`hfa/registry.py`), never by the LLM: a verified contradicting fact from a
policy document (terms / acceptable-use / legal / site-policy pages, tax
authority and legislation sites) gives REFUSE (`POLICY_REFUSAL`); any other
source gives KILL (`RULE_VIOLATION`). An unknown source is never a REFUSE.

### Normalization (applied identically to quote and source)
HTML entity decoding; Unicode **NFKC**; curly quotes -> straight, dashes ->
`-`; soft hyphen and zero-width characters removed; whitespace runs ->
one space; `casefold()` for comparison only. Digits, currency signs,
negations and footnote markers such as `[3]` are **not** normalized: a quote
has to match them exactly. Block-level HTML elements become separators,
inline elements do not (otherwise `Evectv.<sup>[3]</sup>` would break).

## Setup

Python 3.11+ and nothing else for the core (standard library only).

```bash
git clone <this repo> && cd honest-freelance-agent
python3 -m unittest discover -s tests -t .      # 87 tests, offline, no keys
```

## Run

**Mock mode - no keys, no network, `"llm": "mock"`.** A scripted proposer
against a fictional terms page bundled in `hfa/demo_data/`, and an in-memory
PayPal stand-in. It shows the pipeline (GO with invoice, REFUSE, UNVERIFIED
paraphrase). It is not a measurement of anything.

```bash
python3 -m hfa demo --mock
```

**Real mode - your own keys, one command.** Fill in `.env` (copy
`.env.example`): a PayPal **sandbox** REST app (developer.paypal.com -> Apps
& Credentials -> Sandbox) plus two sandbox personas, and your Anthropic key.

```bash
cp .env.example .env          # set PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET,
                              # HFA_SANDBOX_MERCHANT_EMAIL, HFA_SANDBOX_BUYER_EMAIL,
                              # HFA_LLM_PROVIDER=anthropic, HFA_API_KEY
pip install anthropic         # only needed for the Anthropic provider
python3 -m hfa demo --invoice
```

This asks the model (default `claude-haiku-4-5`, budget-capped) whether the
example request is allowed under the hackathon rules, fetches the rules page
live, verifies every quote, runs the gate, and only on GO creates a sandbox
invoice (draft -> send -> status). Without `--invoice` it stops after the
decision. The trace goes to `data/traces/demo.jsonl` (redacted).

Other commands:

```bash
python3 -m hfa paypal-smoke   # sandbox OAuth token only: prints HTTP status + token length
python3 -m hfa budget         # LLM spend vs. caps

# Verify one quote against a live page (fresh fetch). On 2026-10-26 this
# returned VERIFIED / MULTI_CONSISTENT: the URL appears in 4 code samples on
# that page, none with a guard word or a differing amount.
python3 -m hfa verify --url https://developer.paypal.com/api/rest/ \
  --quote "https://api-m.sandbox.paypal.com/v1/oauth2/token"

python3 -m hfa eval --live    # mini-eval (mock proposer), sources fetched live
                              # (implemented, not yet part of the test suite)
```

The PayPal client refuses any host other than `api-m.sandbox.paypal.com`
(exact match, https, default port), at startup, in the constructor and
before every request. There is no switch to turn this off.

LLM settings: `HFA_LLM_PROVIDER` = `mock` (default) | `anthropic` |
`openai_compat`; `HFA_MODEL` (default `claude-haiku-4-5`); `HFA_API_KEY`.

## Cost guard

Every real LLM call is metered from the provider's `usage` field. Hard caps:
**USD 1 per UTC day, USD 15 in total**, stored in `data/llm_budget.json`
(git-ignored). Env variables can only lower the caps. Before a call the
worst case (input estimate + `max_tokens`) is reserved; if it does not fit,
no call is made and the result is the explicit state `budget_exhausted`,
in the CLI output and in the trace.

## Mini-eval

`data/eval/cases_public.jsonl`: 40 cases (hackathon rules, platform fees,
platform terms, tax thresholds, payments) with gold labels GO / KILL /
REFUSE / NEEDS_HUMAN / UNVERIFIED. Third-party text is not redistributed:
each quote is cut to at most 200 characters and stored with the SHA-256 of
the full quote and the hashes recorded at fetch time (2026-10-04). The full
snapshots stay outside the repository (`data/private/`); `--live` re-fetches.

Metric: **false-GO rate** = GO given when gold is not GO, divided by the
number of non-GO cases. Reported next to it: false-KILL count and deferral
rate (NEEDS_HUMAN/UNVERIFIED where a firm answer was possible) - the price
of honesty, so that "always ask a human" cannot look like a win.

Honesty rules in code: every row carries `llm` (`mock` or
`<provider>:<model>`); aggregation refuses a run that mixes mock and real
rows; mock results go only to a separate table with the banner *"MOCK -
pipeline test, not a model measurement"*. KILL<->REFUSE mix-ups are
reported on their own line.

Holdout: `data/eval/holdout_public.jsonl` (11 cases), pinned by SHA-256. It
runs only after `python3 -m hfa eval --freeze` has recorded the prompt hash
and the decision-code hash, only if both still match, never with the mock,
and only once. Every gold or coding change before the freeze is logged in
[EVAL_CHANGELOG.md](EVAL_CHANGELOG.md), including whether it was made after
seeing mock results.

Current numbers are **mock only** (a stand-in that says GO on every trap
case), dev set: without verifier 32/35 false GO; with verifier + gate 1/35
(T16: the fee quote is correct, the loss is arithmetic the gate cannot see),
false KILL 0/5, deferral 3/24, KILL<->REFUSE mix-ups 2. This shows that the
pipeline wiring works; it says nothing about any real model.

## Limitations (honest list)

- A string match proves the sentence is on the page, not that the page is
  right or that the conclusion drawn from it is right.
- Guard words, soft-404 / captcha patterns and the "legal page older than
  365 days" rule are heuristics, and the date heuristic takes the latest
  date on the page.
- No headless browser: JS-only pages end as `THIN_CONTENT`. No OCR, no PDF
  layer yet.
- The domain check uses the last two host labels (no public suffix list).
- DNS is resolved for the SSRF check and again for the connection
  (rebinding window).
- REFUSE vs KILL comes from a URL-pattern registry; a fee or threshold page
  that happens to live under `/terms` or on a tax-authority domain is
  treated as a policy document.
- "NOT_WORTH" decisions rest on arithmetic; unless the proposer states that
  arithmetic as an INFERENCE claim, the gate cannot catch a wrong sum.

## License

Apache-2.0, see [LICENSE](LICENSE).
