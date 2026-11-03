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
proposer. A minimal local UI (`hfa ui`, 127.0.0.1 only) shows the decision cards.

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

Gate rules: **C0** arithmetic (CALC claims, below) is checked first: bad
inputs or a sum that differs from what the proposer claimed -> NEEDS_HUMAN;
a failed check (e.g. net pay below your minimum) -> KILL (NOT_WORTH) even if
the proposer said GO. **G1** a verified critical fact that contradicts the plan -> KILL.
**G2** GO only if every critical fact is VERIFIED and there is at least one.
**G3** any critical NEEDS_HUMAN, or a decision resting on an INFERENCE ->
NEEDS_HUMAN. **G4** any critical UNVERIFIED -> no GO. **G5** facts only the
user knows (USER_ATTESTED) are never VERIFIED and need an explicit tick.
**G6** a GO that rests on the user's own amounts or thresholds (a money
amount in the request, or a USER_ATTESTED number) needs a passing CALC.

### CALC: the code does the sums
A CALC claim names a formula and its inputs, e.g. `fee = max(rate * price,
minfee)`, `net = price - fee`, check `net >= cost`. Each input must point at
a VERIFIED quote that contains the number verbatim (`"10%"`, `"$5.00"`) or
at a USER_ATTESTED statement (`"the job pays 30 USD"`). The code parses the
numbers itself and evaluates the formula with its own small parser (no
`eval()`; only `+ - * /`, parentheses, `max`, `min`; `decimal.Decimal`).

**What this guarantees and what it does not:** the code guarantees the
arithmetic for the stated formula and inputs. Reading the formula out of the
quoted rule (that "10% or $5.00, whichever is greater" means
`max(0.10 * price, 5.00)`) is an INFERENCE by the model, it is not verified,
and the decision card labels it that way.

Reason codes: `EXACT_MATCH`, `NO_MATCH`, `QUOTE_TOO_SHORT`, `URL_REFUSED`,
`FETCH_FAILED`, `FETCH_BLOCKED` (403/429, captcha, login wall, robots.txt -
never bypassed), `SOFT_404`, `THIN_CONTENT` (JS-only page), `MULTI_MATCH`
(several places, and one of them has a guard word or the places show
different amounts), `MULTI_CONSISTENT` (several places that agree: VERIFIED,
all snippets kept), `REDIRECTED` (other domain), `STALE_SOURCE` (legal / tax
source whose latest date - Last-Modified, URL year or a date in the text - is
older than 365 days; no date at all gives VERIFIED with `date_flag: UNDATED`), `AMBIGUOUS_CONTEXT` (a guard word such as "except",
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

> **Requires Python 3.11 or newer (`python3.11+`).** On macOS the system
> `python3` can be an old or broken build; call `python3.11` (or newer)
> explicitly, as all commands below do.

Nothing else is needed for the core (standard library only).

```bash
git clone <this repo> && cd honest-freelance-agent
python3.11 -m unittest discover -s tests -t .      # 190 tests, offline, no keys (1 skipped without the private snapshots)
```

## Run

**Mock mode - no keys, no network, `"llm": "mock"`.** A scripted proposer
against a fictional terms page bundled in `hfa/demo_data/`, and an in-memory
PayPal stand-in. It shows the pipeline (GO with invoice, REFUSE, UNVERIFIED
paraphrase). It is not a measurement of anything.

```bash
python3.11 -m hfa demo --mock
```

**Real mode - your own keys, one command.** All keys live in ONE key file:
`.env` in the repo folder (git-ignored), or any file you name with
`HFA_ENV_FILE=/path/to/file`. Copy `.env.example` and fill in a PayPal
**sandbox** REST app (developer.paypal.com -> Apps & Credentials -> Sandbox),
two sandbox personas, and your Anthropic key as `ANTHROPIC_API_KEY=...`.

```bash
cp .env.example .env          # PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET,
                              # HFA_SANDBOX_MERCHANT_EMAIL, HFA_SANDBOX_BUYER_EMAIL,
                              # ANTHROPIC_API_KEY
python3.11 -m pip install anthropic   # only needed for the Anthropic provider
python3.11 -m hfa demo --invoice
# or, with the key file elsewhere:
HFA_ENV_FILE=~/my-keys/hfa.env python3.11 -m hfa demo --invoice
```

**What the model sees vs. what is checked.** The model gets each source's
fetched text, cut to the first 12 000 characters (the cut is logged per
source: original length, passed length, truncated or not). The verifier
never uses that text: it compares every quote with its **own** fresh fetch.
A quote the model takes from the part that was cut off is marked in the
proposer layer, its quote field is emptied, and it ends UNVERIFIED (the
frozen verifier reports this as `QUOTE_TOO_SHORT`, the closest existing
reason code; the claim text carries the mark "quote outside the text given
to the model").

The LLM key is read **only** from that file. A key that happens to be set in
your shell (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) is never used, and
the key is never copied into the process environment. No file or no key ->
every output says `"llm": "unavailable"` (CLI card, trace, usage log) and no
model is called; the mock runs only when you pass `--mock`.

This asks the model (default `claude-haiku-4-5`, budget-capped) whether the
example request is allowed under the hackathon rules, fetches the rules page
live, verifies every quote, runs the gate, and only on GO creates a sandbox
invoice (draft -> send -> status). Without `--invoice` it stops after the
decision. The trace goes to `data/traces/demo.jsonl` (redacted).

**Your own job, as live input.** "Is it worth it?" with your numbers:

```bash
python3.11 -m hfa demo --mock --price 30 --cost 29 --user-confirm   # fictional platform, llm: mock
python3.11 -m hfa demo --price 30 --cost 26 --platform freelancer --user-confirm   # real model
```

`--user-confirm` is **your** tick on the numbers you typed (G5). The model can
never set it: any `user_confirmed` in the model's output is discarded when it
is parsed. Without the tick the result is NEEDS_HUMAN.
The tick also covers only **your** numbers: it lands on a USER_ATTESTED claim
only if every number in that claim (read as a Decimal, with its currency or
`%`) appears in what you typed (the CLI fields; in the eval, the case's input).
If the model writes its own "user" number (`my cost is 20 USD` while you
typed 26), that claim stays unticked with the reason *"user number not in the
user's input"*, and the result cannot be GO. `26`, `26.00`, `26 USD`,
`USD 26` and `$26` are the same number; `26 EUR` and `26%` are not; a number
with a currency needs that currency in your input.

**Local UI.** The same cards in a browser, served only on `127.0.0.1`
(standard library, no external scripts or styles, nothing hosted):

```bash
python3.11 -m hfa ui --mock            # big MOCK banner
python3.11 -m hfa ui --price 30 --cost 26 --platform freelancer --user-confirm
# then open http://127.0.0.1:8765/
```

Each card shows the quoted text, the source URL, the source SHA-256 and
fetch time, the verifier state and reason, the `llm: ...` label, and the
raw PayPal invoice status.

**Reminders need your approval.** `python3.11 -m hfa remind <invoice-id>`
only saves a draft (`data/reminders/`); nothing is sent until you run
`python3.11 -m hfa remind --approve <invoice-id>`. (PayPal also limits
reminders per day.)

The invoice status is always PayPal's raw value: `MARKED_AS_PAID` (the
invoicer recorded a payment) is shown as such, never as `PAID` (the payer
paid).

Other commands:

```bash
python3.11 -m hfa paypal-smoke   # sandbox OAuth token only: prints HTTP status + token length
python3.11 -m hfa budget         # LLM spend vs. caps

# Verify one quote against a live page (fresh fetch). On 2026-10-26 this
# returned VERIFIED / MULTI_CONSISTENT: the URL appears in 4 code samples on
# that page, none with a guard word or a differing amount.
python3.11 -m hfa verify --url https://developer.paypal.com/api/rest/ \
  --quote "https://api-m.sandbox.paypal.com/v1/oauth2/token"

python3.11 -m hfa eval --mock --live   # mini-eval (mock proposer, labelled MOCK),
                                        # sources fetched live (not yet in the test suite)

# Model-free replay of a stored run (E33): the raw model answers saved in the
# run file go again through parse -> verifier -> gate on the same snapshots.
# No LLM client is created. Exit 0 = identical, 1 = differences (listed),
# 2 = cannot replay (e.g. "no raw model output stored in this run (pre-E33)").
python3.11 -m hfa eval --replay data/eval/runs/real/eval_<llm>.json [--snapshots DIR] [--cases FILE]
```

The PayPal client refuses any host other than `api-m.sandbox.paypal.com`
(exact match, https, default port), at startup, in the constructor and
before every request. There is no switch to turn this off.

LLM settings: `HFA_LLM_PROVIDER` = `anthropic` (default) | `openai_compat`
(key `HFA_OPENAI_COMPAT_KEY` in the key file, plus `HFA_OPENAI_BASE_URL`);
`HFA_MODEL` (default `claude-haiku-4-5`). There is no "mock" provider
setting; the mock is only the explicit `--mock` switch.

## Cost guard

Every real LLM call is metered from the provider's `usage` field. Each call
writes one line to the run log (`data/traces/demo.jsonl`): model, input and
output tokens, and an **estimated** USD cost from Anthropic's list prices
(Claude Haiku 4.5: USD 1 / 5 per million input / output tokens; the figure is
labelled as an estimate, and an unpriced model logs tokens only). No key and
no key identifier is ever logged. The daily and total counters below are
computed from the same usage numbers. Hard caps:
**USD 1 per UTC day, USD 15 in total**, stored in `data/llm_budget.json`
(git-ignored). Env variables can only lower the caps. Before a call the
worst case (input estimate + `max_tokens`) is reserved; if it does not fit,
no call is made and the result is the explicit state `budget_exhausted`,
in the CLI output and in the trace.

## Mini-eval

`data/eval/cases_public.jsonl`: 40 dev cases (hackathon rules, platform fees,
platform terms, tax thresholds, payments) with gold labels GO / KILL /
REFUSE / NEEDS_HUMAN / UNVERIFIED. Third-party text is not redistributed:
each quote is cut to at most 200 characters and stored with the SHA-256 of
the full quote and the hashes recorded at fetch time (2026-10-04). The full
snapshots stay outside the repository (`data/private/`, `eval_snapshots/`
are git-ignored).

**Reproducibility, plainly:** without those snapshots a judge cannot
reproduce the eval byte for byte, because the pages have changed since
2026-10-04 and their full text is not ours to redistribute. What a judge can
check: the verifier fetches every source fresh (`hfa verify --url`, `hfa eval
--live`), and the stored quote hashes and fetch-time hashes show what was
matched then.

Metric: **false-GO rate** = GO given when gold is not GO, divided by the
number of non-GO cases. Reported next to it: false-KILL count and deferral
rate (NEEDS_HUMAN/UNVERIFIED where a firm answer was possible) - the price
of honesty, so that "always ask a human" cannot look like a win.

Honesty rules in code: every row carries `llm` (`mock` or
`<provider>:<model>`); aggregation refuses a run that mixes mock and real
rows; mock results go only to a separate table with the banner *"MOCK -
pipeline test, not a model measurement"*. KILL<->REFUSE mix-ups are
reported on their own line.

Holdout: `data/eval/holdout_public.jsonl` (11 cases), pinned by SHA-256. The
freeze has two final steps, issued by the eval owner: `python3.11 -m hfa eval
--freeze gate` (hash of the decision code) and `--freeze prompt` (hash of the
proposer prompt). Each step only writes a file (`data/eval/FREEZE_GATE.json`
or `data/eval/FREEZE_PROMPT.json`, created once, never overwritten); it
makes no git commit. The holdout runs only when both exist and still match,
never with the mock, and only once.

Running it: the snapshots are not in the repository and the default
`--snapshots data/private/snapshots` does not exist in a fresh clone (the run
stops with a message saying so). The eval owner's copy is
`~/ai-collab/002-cash/prep/eval_snapshots/` (all files the 11 holdout cases
reference; R07 uses `T08.txt`):

```bash
python3.11 -m hfa eval --holdout data/eval/holdout_public.jsonl \
  --frozen-prompt-sha <hash from FREEZE_PROMPT.json> \
  --snapshots ~/ai-collab/002-cash/prep/eval_snapshots
```

Before the first model call the run checks that every referenced snapshot
exists and that the **budget** fits: the estimated cost of all cases (every
source at the full 12 000 characters given to the model, ~3 chars per token,
the whole `max_tokens` as output, same price table as the cost guard) x 1.5
must be below both the remaining daily and the remaining total budget;
otherwise nothing starts (exit 4, no lock, no call). Then it writes
`data/eval/HOLDOUT_RUN.json` atomically with state `STARTED`, and fills it in
at the end (`COMPLETE`, `INCOMPLETE`, or `ABORTED` with the error if the run
raised). A lock in any state - also a `STARTED` lock left behind by a crash
or a kill - refuses every later run.

**Voiding a holdout lock** (eval owner 002 only, by hand; nothing unlocks
automatically): rename the lock and record the decision and the reason next
to it, then add an EVAL_CHANGELOG entry.

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
mv data/eval/HOLDOUT_RUN.json data/eval/HOLDOUT_RUN.void-$ts.json
echo "002 decision <date>: <why the run is void, what was seen>" > data/eval/HOLDOUT_RUN.void-$ts.reason.txt
```

A voided lock without a (non-empty) reason file blocks the run as well.

### Results (real model, frozen gate + frozen prompt p8)

Model `anthropic:claude-haiku-4-5`; decision code frozen (gate `9bc97c85...5813`),
proposer prompt frozen (p8, `de5c1191...c7fc`). Each run was replayed from its stored
raw model output with `hfa eval --replay` on commit `53755ad` (holdout v2 lock committed in `daadad2`) and gave identical
decisions (exit 0).

**Dev set, three runs (T1-T3 p8, 2026-11-01/02), 40 cases each:**

| | without verifier | with verifier + gate + CALC |
|---|---|---|
| false GO | 5 / 5 / 5 of 35 non-GO cases | **0 / 0 / 0** |
| wrong KILL (of 5 gold-GO cases) | 0 / 0 / 0 | 1 / 0 / 0 |
| deferral rate | 41.7 / 37.5 / 39.1 % | 50 / 50 / 47.8 % (T3: 50 % incl. errors) |
| cases with no decision (model output rejected) | – | 0 / 3 / 1 |
| GO decisions | – | **0 in every run** |

Honest reading: with the real model the verifier has never produced a GO. It removes
every false GO, but it also defers or rejects the good opportunities. The one wrong
KILL (T1, case T03) came from the model marking a neutral rules sentence as
contradicting.

CALC claims: T1 0/7, T2 2/8 (+1 not evaluated), T3 0/7 valid. The model keeps writing
thresholds as numbers inside `check` and makes its own arithmetic errors; the code
rejects both.

**Holdout:** v1 (prompt p5, 2026-10-30, replayed exit 0 on the code it ran on, 2026-10-30, before E34b) and v2
(prompt p8, 2026-11-02, 11 cases, 1 rejected model output): false GO 0/10 with the
verifier vs 2/10 without; exact label 8/10 vs 2/10; no GO decision. v1 (p5): 0/11 vs
1/11, exact 4/11 vs 2/11. Both are reported. v1 was voided after the E34-E36 prompt/parse
fixes. Replaying holdout v1 on later code changes claim-level parsing for R01/R04
(E34b alias handling); the decisions are unchanged.

**Cost:** all real-model runs including voided ones: 4.02 $ (sum of the per-call cost estimates stored with the 15 real-model runs);
one p8 dev run ≈ 0.35 $.

**Read this as a small sample, not a benchmark:** 40 + 11 hand-labelled
cases, one model. It shows that the verifier and the gate stop false GOs on
these cases, at the price of deferring every good opportunity so far; it does
not show how the agent performs on other questions or other models. Every
gold, code and prompt change is logged in [EVAL_CHANGELOG.md](EVAL_CHANGELOG.md),
including the ones made after looking at results (prompt p5 -> p8, E34-E36).
The earlier mock-only numbers (a stand-in that says GO on every trap case)
only showed that the pipeline wiring works.

## Known limitations

- A string match proves the sentence is on the page, not that the page is
  right or that the conclusion drawn from it is right.
- Guard words and soft-404 / captcha patterns are heuristics.
- No headless browser: JS-only pages end as `THIN_CONTENT`. No OCR, no PDF
  layer yet.
- The domain check uses the last two host labels (no public suffix list).
- DNS is resolved for the SSRF check and again for the connection
  (rebinding window).
- REFUSE vs KILL comes from a URL-pattern registry; a contradicting quote
  from a page under `/terms` or on a tax-authority domain counts as a policy
  document. When a CALC decides the case (e.g. a fee table on a terms page),
  the registry is not consulted and the result is KILL (NOT_WORTH).
- CALC checks the sums, not the formula: a wrong reading of a fee rule gives
  a correctly computed wrong answer. Literal parsing handles common formats
  (`10%`, `$5.00`, `3,40%`, `25 000`) but not words ("twenty").
- amounts written in words are not detected by G6
- G6 fires when the request states a money amount or a USER_ATTESTED claim
  carries a number; a percentage that appears only in a source quote does
  not trigger it, and a bare year (1900-2100 with no currency or % next to
  it) is not counted as a number.
- Source dates for legal / tax sources (the latest of HTTP Last-Modified, a
  year in the URL path, and dates found in the text) are a heuristic: an old
  date quoted on a current page can look current only if a newer date is
  also present, and a page with no date at all is shown as UNDATED (still
  VERIFIED, with a yellow label), not as stale.

## License

Apache-2.0, see [LICENSE](LICENSE).
