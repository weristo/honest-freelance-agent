# Honest Freelance Agent (HFA)

An agent that helps a freelancer decide whether to take a job, and then bills
it through PayPal Invoicing (sandbox). The LLM **proposes** a GO / KILL
decision and cites the platform rules it relies on. Deterministic code
**verifies** every cited quote against a freshly fetched copy of the source,
and only a decision whose critical facts are all `VERIFIED` can create an
invoice.

> Built by AI agents (Claude) under the supervision of Otto Lőkös.

Status: early build (hackathon). The verifier, gate, budget guard, PayPal
sandbox client and mini-eval run and are tested. The proposer runs on a
**mock** in this version; the real-LLM path is implemented but has not been
measured yet. The UI is not built yet.

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
never bypassed), `SOFT_404`, `THIN_CONTENT` (JS-only page), `MULTI_MATCH`,
`REDIRECTED` (other domain), `STALE_SOURCE` (legal page older than 365 days),
`AMBIGUOUS_CONTEXT` (guard words such as "except", "not", "employer" next to
a supporting quote), `CONTRADICTED`, `SOURCE_CHANGED`.

### Normalization (applied identically to quote and source)
HTML entity decoding; Unicode **NFKC**; curly quotes -> straight, dashes ->
`-`; soft hyphen and zero-width characters removed; whitespace runs ->
one space; `casefold()` for comparison only. Digits, currency signs,
negations and footnote markers such as `[3]` are **not** normalized: a quote
has to match them exactly. Block-level HTML elements become separators,
inline elements do not (otherwise `Evectv.<sup>[3]</sup>` would break).

## Setup

Python 3.11+ and nothing else: the core uses only the standard library.

```bash
git clone <this repo> && cd honest-freelance-agent
python3 -m unittest discover -s tests -t .      # 67 tests, offline
cp .env.example .env                              # then fill in YOUR sandbox keys
```

## Run

```bash
# Verify one quote against a live page (fresh fetch). On 2026-10-26 this
# returned NEEDS_HUMAN / MULTI_MATCH: the URL appears in 4 different code
# samples on that page, so a human has to look.
python3 -m hfa verify --url https://developer.paypal.com/api/rest/ \
  --quote "https://api-m.sandbox.paypal.com/v1/oauth2/token"

# PayPal sandbox smoke test (prints only HTTP status and token length)
set -a; . ./.env; set +a
python3 -m hfa paypal-smoke

# LLM spend vs. caps
python3 -m hfa budget

# Mini-eval (mock proposer), fetching every source live
# (live mode is implemented but not yet part of the test suite)
python3 -m hfa eval --live
# ...or offline, if you have the snapshot files in data/private/snapshots
python3 -m hfa eval
```

PayPal: create a **sandbox** REST app at developer.paypal.com (Apps &
Credentials -> Sandbox) and use its client id / secret. The client refuses
any host other than `api-m.sandbox.paypal.com` (exact match, https, default
port), at startup, in the constructor and before every request. There is no
switch to turn this off.

LLM: `HFA_LLM_PROVIDER` = `mock` (default) | `anthropic` | `openai_compat`;
`HFA_MODEL` defaults to `claude-haiku-4-5`; the key is the judge's own
(`HFA_API_KEY`). `pip install anthropic` is needed only for the Anthropic
provider.

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
pipeline test, not a model measurement"*. Holdout cases (`split: holdout`)
run only with `--holdout --frozen-prompt-sha <current prompt hash>`; no
holdout cases ship yet.

Current numbers are **mock only** (a stand-in that says GO on every trap
case): without verifier 32/35 false GO, with verifier + gate 0/35, deferral
5/23. This shows that the pipeline wiring works; it says nothing about any
real model.

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
- The gate cannot tell a terms-of-service refusal from a rule violation by
  itself; that comes from the proposer's `kill_class`.

## License

Apache-2.0, see [LICENSE](LICENSE).
