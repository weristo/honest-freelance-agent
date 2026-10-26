# Eval changelog

Every change to gold labels, case coding or decision code made **before the
freeze** is listed here: what changed, why, when, and whether it was made
after looking at mock-run results. Changes made after seeing results are a
form of tuning on the dev set; the holdout set (`data/eval/holdout_public.jsonl`,
pinned SHA-256, run once after the freeze, never with the mock) is the
counterweight. Status: **not frozen yet** (no `data/eval/FREEZE.json`).

All numbers below come from the **mock** proposer (a stand-in that says GO on
every case flagged as a false-GO trap). They test the wiring, not a model.

| # | Date | Kind | Change | Why | After seeing mock results? |
|---|------|------|--------|-----|----|
| E1 | 2026-10-26 | code (verifier) | MULTI_MATCH de-duplication: a hit in the visible layer and one in the embedded layer count as one passage; within a layer two hits are the same passage if the 40 chars on the left OR on the right are identical (was: identical +-80 chars, across layers). | The first version reported the same sentence shipped as rendered text + page data (and repeated table blocks) as ambiguous. | Yes (first mock run: false MULTI_MATCH on T01, T15, T31). |
| E2 | 2026-10-26 | code (eval mock) | The mock proposer gives CONTEXT claims no `rule_topic`, so they do not enter the contradiction pass. | Context quotes are not claims about the decision; pairing them with the critical claim produced spurious CONTRADICTED. | Yes (T17, T21, T26, T33, T34, T40). |
| E3 | 2026-10-26 | code (verifier) | Guard-word window = the sentence that contains the quote + the sentence right after it (was: +-150 chars, spec said +-300). Guard words inside the quote itself do not count. Word list unchanged. T01 gold stays GO. | Coordinator decision (002). | **Yes. Changed after seeing the mock run; the holdout counterweighs this.** Effect on the mock run: T16 (gold KILL) is now a false GO, because the fee quote is verified and the loss is arithmetic the gate cannot see; T04 (gold GO) is now deferred by a guard word in the next sentence. |
| E4 | 2026-10-26 | code (gate) | Source registry (`hfa/registry.py`): a G1 KILL is REFUSE (POLICY_REFUSAL) only if the contradicting verified fact comes from a policy document (path contains terms / acceptable-use / legal / site-policy, or a tax-authority / legislation domain: irs.gov, nav.gov.hu, eur-lex.europa.eu, bzst.de, gesetze-im-internet.de, njt.hu); otherwise RULE_VIOLATION. The proposer can no longer grant POLICY_REFUSAL anywhere: an LLM-proposed refusal without a registry-backed G1 becomes RULE_VIOLATION. KILL<->REFUSE mix-ups are reported on a separate line. | Coordinator decision (002): REFUSE vs KILL must come from code, not from the LLM. | Yes (round-1 mock showed 6 REFUSE->KILL mix-ups). Effect: mix-ups 6 -> 2 (T21: PeoplePerHour terms page; T30: nav.gov.hu threshold page; both gold KILL, now REFUSE). |
| E5 | 2026-10-26 | code (verifier) | Several distinct matches are VERIFIED (`MULTI_CONSISTENT`, all snippets kept) if no match context has a guard word and the contexts do not differ in amounts (prices, percentages, currency amounts within +-300 chars, quote excluded). Otherwise NEEDS_HUMAN (`MULTI_MATCH`). The "different amounts" test is a heuristic. | Coordinator decision (002). R07 (same sentence twice, once next to a different price) must stay NEEDS_HUMAN: proven with a synthetic fixture test, the holdout was not run. | Yes (T03, T31 deferred by MULTI_MATCH in the mock run). |
| E6 | 2026-10-26 | data + code | Holdout loader with SHA-256 lock (private file `0b7ae930...94994`, public file `886c9636...989d3`), freeze (prompt hash + decision-code hash), run-once lock, never with the mock. Public holdout file: quotes cut to <=200 chars + SHA-256 of the full quote. Loader rule: a case whose claimed quote is known not to be in the source must have gold UNVERIFIED (R03-R06, R09-R11; already true in the source file, no gold changed). | Coordinator decision (002). | No (holdout not run; no results seen). Note: the public holdout file is readable in the repo, so its integrity is procedural plus the hash lock. |
| E7 | 2026-10-26 | gold | T05: gold NEEDS_HUMAN -> **KILL (RULE_VIOLATION)**. The source is the hackathon's rules page (devpost /rules), not a terms-of-service document. | Coordinator decision (002). The earlier NEEDS_HUMAN rested on an internal team rule that is not part of the case. | Yes (round-1 mock gave KILL). |
| E8 | 2026-10-26 | case coding | T06: the "solely owned" quote's role CONTRADICTS -> CONTEXT; added a critical INFERENCE claim ("the OAH promotional licence conflicts with 'solely owned'", derived from both quotes) so G3 applies. Gold stays NEEDS_HUMAN. | Coordinator decision (002): the conflict is a conclusion, not a quotable fact. | Yes (round-1 mock gave KILL). |

## Mock results by stage (dev set, 40 cases, MOCK - not a model measurement)

| Stage | false GO (verifier) | false KILL | deferral | KILL<->REFUSE | exact |
|---|---|---|---|---|---|
| Round 1 first run (before E1, E2) | 0/35 | 0/5 | 13/23 | 6 | 21/40 |
| Round 1 final (E1, E2) | 0/35 | 0/5 | 5/23 | 6 | 24/40 |
| Round 2 (E3-E8) | **1/35** (T16) | 0/5 | 3/24 | 2 | 31/40 |

Without the verifier the same mock gives 32/35 false GO in every stage.
The round-2 rise from 0 to 1 false GO is kept on purpose and reported: the
earlier 0 depended on a guard word that happened to be near the T16 quote.

Fingerprint at the time of writing (not a freeze): prompt
`19d1e6930602...`, decision code `d10648785255...`.
