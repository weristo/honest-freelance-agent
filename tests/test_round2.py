import json
import os
import tempfile
import unittest

from hfa import evalrun, holdout
from hfa.gate import decide
from hfa.models import ClaimKind, EvidenceState as S, Polarity, Reason as R, Verdict as V
from hfa.registry import is_policy_source
from hfa.verifier import find_contradictions, guard_window, verify

from .helpers import FILLER, RULE, StaticFetcher, fact, page

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(claim, doc):
    return verify(claim, StaticFetcher(doc))


class GuardWindowTests(unittest.TestCase):
    def test_window_is_quote_sentence_plus_next(self):
        t = "One. Two before " + RULE + " Three after. Four far away."
        s = t.index(RULE)
        before, after = guard_window(t, s, s + len(RULE))
        self.assertEqual(before.strip(), "Two before")
        self.assertEqual(after, " Three after.")

    def test_guard_word_inside_quote_does_not_count(self):
        q = "GitHub Sponsors does not charge any fees for personal sponsorships."
        e = run(fact(q), page(q))
        self.assertEqual(e.state, S.VERIFIED)

    def test_guard_word_in_next_sentence_counts(self):
        e = run(fact(), page(f"{RULE} This does not apply to teams."))
        self.assertEqual(e.reason_code, R.AMBIGUOUS_CONTEXT)

    def test_guard_word_two_sentences_later_does_not_count(self):
        e = run(fact(), page(f"{RULE} Judges read the README. Late entries are not accepted."))
        self.assertEqual(e.state, S.VERIFIED)

    def test_guard_word_earlier_in_same_sentence_counts(self):
        e = run(fact("judges can run themselves", source_url="https://example.org/rules"),
                page("Unless waived, every submission must include a working build that judges can run themselves."))
        self.assertEqual(e.reason_code, R.AMBIGUOUS_CONTEXT)


class MultiMatchTests(unittest.TestCase):
    def test_consistent_multi_match_is_verified_with_all_snippets(self):
        txt = f"Overview. {RULE} Read the rules. " + FILLER + f" Checklist. {RULE} Good luck."
        e = run(fact(), page(txt))
        self.assertEqual((e.state, e.reason_code), (S.VERIFIED, R.MULTI_CONSISTENT))
        self.assertEqual(len(e.other_snippets), 1)
        self.assertIn("2 consistent matches", e.detail)

    def test_guard_word_in_any_context_needs_human(self):
        txt = f"Overview. {RULE} Read the rules. " + FILLER + f" Checklist. {RULE} Except for demos."
        e = run(fact(), page(txt))
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.MULTI_MATCH))

    def test_r07_like_fixture_conflicting_prices_needs_human(self):
        # Synthetic stand-in for holdout R07 (holdout itself NOT run): the same
        # sentence twice, once next to a different price in the fee table.
        q = "Featured ($9.99) highlights your project in search results."
        txt = (f"Upgrades. Featured $10.99 USD. Urgent $10.99 USD. {q} " + FILLER +
               f" Frequently asked questions. {q} Upgrades are optional.")
        doc = page(txt)
        e = run(fact(q), doc)
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.MULTI_MATCH))
        ctx = fact("Featured $10.99 USD. Urgent $10.99 USD.", cid="c0", critical=False)
        claims = [ctx, fact(q, cid="c1")]
        evs = find_contradictions(claims, [run(c, doc) for c in claims])
        self.assertEqual(decide(V.GO, claims, evs, "mock").final, V.NEEDS_HUMAN)


class RegistryTests(unittest.TestCase):
    def test_policy_sources(self):
        yes = ["https://www.freelancer.com/about/terms", "https://info.devpost.com/legal/terms-of-service",
               "https://docs.github.com/en/site-policy/github-terms/github-terms-of-service",
               "https://docs.github.com/en/site-policy/acceptable-use-policies/x",
               "https://www.irs.gov/businesses/understanding-your-form-1099-k",
               "https://nav.gov.hu/ado/uj-kata", "https://eur-lex.europa.eu/legal-content/EN/TXT/",
               "https://www.bzst.de/DE/x.html"]
        no = ["https://paypalaihackathon.devpost.com/rules", "https://www.freelancer.com/feesandcharges",
              "https://example.org/", "", "https://irs.gov.evil.example/terms-x".replace("terms-x", "x"),
              "https://notnav.gov.hu.example.com/"]
        for u in yes:
            self.assertTrue(is_policy_source(u), u)
        for u in no:
            self.assertFalse(is_policy_source(u), u)


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.freeze = os.path.join(self.tmp.name, "FREEZE.json")
        self.lock = os.path.join(self.tmp.name, "HOLDOUT_RUN.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_public_holdout_loads_with_pinned_hash(self):
        cases = holdout.load_holdout(os.path.join(HERE, "data", "eval", "holdout_public.jsonl"))
        self.assertEqual(len(cases), 11)
        for c in cases:
            self.assertEqual(c["split"], "holdout")
            if any(cl["claimed_quote_not_in_source"] for cl in c["claims"]):
                self.assertEqual(c["gold"], "UNVERIFIED")
            for cl in c["claims"]:
                self.assertLessEqual(len(cl["quote"]), 200)

    def test_tampered_holdout_refused(self):
        src = os.path.join(HERE, "data", "eval", "holdout_public.jsonl")
        p = os.path.join(self.tmp.name, "h.jsonl")
        with open(src, encoding="utf-8") as f, open(p, "w", encoding="utf-8") as g:
            g.write(f.read().replace('"gold": "NEEDS_HUMAN"', '"gold": "GO"', 1))
        with self.assertRaises(holdout.HoldoutHashMismatch):
            holdout.load_holdout(p)

    def test_run_requires_freeze_real_model_matching_hash_and_runs_once(self):
        with self.assertRaises(holdout.HoldoutLocked):              # not frozen
            holdout.authorize_run("anthropic:claude-haiku-4-5", None, self.freeze, self.lock)
        rec = holdout.freeze(self.freeze)
        with self.assertRaises(holdout.HoldoutLocked):              # freeze is final
            holdout.freeze(self.freeze)
        with self.assertRaises(holdout.HoldoutLocked):              # never mock
            holdout.authorize_run("mock", rec["prompt_sha256"], self.freeze, self.lock)
        with self.assertRaises(holdout.HoldoutLocked):              # wrong prompt hash
            holdout.authorize_run("anthropic:claude-haiku-4-5", "0" * 64, self.freeze, self.lock)
        holdout.authorize_run("anthropic:claude-haiku-4-5", rec["prompt_sha256"], self.freeze, self.lock)
        holdout.record_run({"n": 11}, self.lock)
        with self.assertRaises(holdout.HoldoutLocked):              # only once
            holdout.authorize_run("anthropic:claude-haiku-4-5", rec["prompt_sha256"], self.freeze, self.lock)

    def test_code_change_after_freeze_refused(self):
        rec = holdout.freeze(self.freeze)
        with open(self.freeze) as f:
            d = json.load(f)
        d["decision_code_sha256"] = "1" * 64
        with open(self.freeze, "w") as f:
            json.dump(d, f)
        with self.assertRaises(holdout.HoldoutLocked):
            holdout.authorize_run("anthropic:claude-haiku-4-5", rec["prompt_sha256"], self.freeze, self.lock)

    def test_cli_holdout_with_mock_is_refused(self):
        with self.assertRaises(holdout.HoldoutLocked):
            evalrun.main(["--holdout", os.path.join(HERE, "data", "eval", "holdout_public.jsonl")])


class GoldChangeTests(unittest.TestCase):
    def _case(self, cid):
        with open(os.path.join(HERE, "data", "eval", "cases_public.jsonl"), encoding="utf-8") as f:
            return next(c for c in map(json.loads, f) if c["id"] == cid)

    def test_t05_gold_kill_and_t06_inference(self):
        self.assertEqual(self._case("T05")["gold"], "KILL")
        t06 = self._case("T06")
        self.assertEqual(t06["gold"], "NEEDS_HUMAN")
        self.assertEqual(t06["claims"][0]["role"], "CONTEXT")
        prop = evalrun.naive_proposal(t06)
        kinds = [(c["kind"], c["critical"]) for c in prop["claims"]]
        self.assertIn(("INFERENCE", True), kinds)

    def test_mixups_reported_separately(self):
        rows = [{"id": "a", "llm": "mock", "gold": "REFUSE", "status": "ok", "no_verifier": "GO", "verifier": "KILL"},
                {"id": "b", "llm": "mock", "gold": "GO", "status": "ok", "no_verifier": "GO", "verifier": "GO"}]
        s = evalrun.aggregate(rows)
        self.assertEqual(s["verifier"]["kill_refuse_mixups"], 1)
        self.assertEqual(s["verifier"]["false_go"], 0)
        self.assertIn("KILL<->REFUSE", evalrun.format_table(rows, s))


if __name__ == "__main__":
    unittest.main()
