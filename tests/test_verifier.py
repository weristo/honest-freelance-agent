import unittest

from hfa.extract import extract_html
from hfa.models import ClaimKind, EvidenceState as S, Polarity, Reason as R
from hfa.normalize import fold_with_map, norm
from hfa.verifier import find_contradictions, verify

from .helpers import FEE, FILLER, RULE, StaticFetcher, fact, page


def run(claim, doc):
    return verify(claim, StaticFetcher(doc))


class NormalizeTests(unittest.TestCase):
    def test_punctuation_whitespace_softhyphen(self):
        self.assertEqual(norm("  “free” – it’s  in­side &amp; ok\n\n"),
                         "\"free\" - it's inside & ok")

    def test_nfkc_ligature_and_fullwidth(self):
        self.assertEqual(norm("ﬁle １０%"), "file 10%")

    def test_footnote_marker_and_digits_kept(self):
        self.assertEqual(norm("Evectv.[3] szerinti 3,40%"), "Evectv.[3] szerinti 3,40%")

    def test_fold_map_handles_length_change(self):
        folded, idx = fold_with_map("Straße X")
        self.assertEqual(folded, "strasse x")
        self.assertEqual(idx[folded.index("x")], 7)


class ExtractTests(unittest.TestCase):
    def test_inline_elements_get_no_separator(self):
        vis, _, _ = extract_html("<p>Evectv.<sup>[3]</sup> szerinti <b>egyéni</b>.</p>")
        self.assertIn("Evectv.[3] szerinti egyéni.", norm(vis))

    def test_scripts_dropped_and_embedded_layer(self):
        html = ('<html><head><title>T</title><script>var x="hidden secret text";</script>'
                '<script type="application/ld+json">{"a": {"b": "LD sentence here"}}</script></head>'
                '<body><div>Shown</div><script>self.__next_f.push([1,"Collapsed rule text"])</script></body></html>')
        vis, emb, title = extract_html(html)
        self.assertNotIn("hidden", vis)
        self.assertIn("Shown", vis)
        self.assertIn("Collapsed rule text", emb)
        self.assertIn("LD sentence here", emb)
        self.assertEqual(title, "T")


class VerifierBranchTests(unittest.TestCase):
    def test_verified_exact_match_with_hashes(self):
        e = run(fact(), page(RULE))
        self.assertEqual((e.state, e.reason_code), (S.VERIFIED, R.EXACT_MATCH))
        self.assertEqual(len(e.snippet_sha256), 64)
        self.assertEqual(e.raw_sha256, "ab" * 32)
        self.assertTrue(e.fetched_at)
        self.assertIn(RULE, e.context_snippet)

    def test_match_is_case_and_whitespace_insensitive(self):
        e = run(fact(RULE.upper()), page(RULE.replace(" ", "\n  ")))
        self.assertEqual(e.state, S.VERIFIED)

    def test_embedded_layer_match_is_flagged(self):
        e = run(fact(), page("", embedded=RULE))
        self.assertEqual((e.state, e.match_layer), (S.VERIFIED, "embedded"))
        self.assertIn("page data", e.detail)

    def test_non_fact_is_not_applicable(self):
        e = run(fact(kind=ClaimKind.INFERENCE), page(RULE))
        self.assertEqual((e.state, e.reason_code), (S.NOT_APPLICABLE, R.NOT_A_FACT))

    def test_quote_too_short_does_not_fetch(self):
        f = StaticFetcher(page(RULE))
        e = verify(fact("30 days"), f)
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.QUOTE_TOO_SHORT))
        self.assertEqual(f.calls, 0)

    def test_url_refused(self):
        e = run(fact(), page(RULE, refused=True, error="non-public address"))
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.URL_REFUSED))

    def test_fetch_failed_network_and_404(self):
        self.assertEqual(run(fact(), page(RULE, http_status=None, error="timeout")).reason_code, R.FETCH_FAILED)
        self.assertEqual(run(fact(), page(RULE, http_status=404)).reason_code, R.FETCH_FAILED)

    def test_blocked_variants(self):
        cases = [
            page(RULE, http_status=403),
            page(RULE, http_status=429),
            page("", visible="Just a moment... Checking your browser before accessing example.org"),
            page(RULE, final_url="https://example.org/sign-in?next=%2Frules"),
            page("", visible="", http_status=202),
            page(RULE, robots_disallowed=True),
        ]
        for doc in cases:
            e = run(fact(), doc)
            self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.FETCH_BLOCKED), doc)

    def test_soft_404_even_when_long(self):
        e = run(fact(), page("Az Ön által kért oldal vagy dokumentum nem található."))
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.SOFT_404))

    def test_thin_content(self):
        e = run(fact(), page("", visible="Menu Home About " + RULE))
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.THIN_CONTENT))

    def test_no_match_paraphrase_shows_nearest_span_only(self):
        e = run(fact("The freelancer fee is 10% or $5, whichever is greater."), page(FEE))
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.NO_MATCH))
        self.assertIsNotNone(e.nearest_span)
        self.assertIsNone(e.context_snippet)

    def test_source_changed_when_seen_before(self):
        e = run(fact(first_seen_snippet_sha256="00" * 32), page("Something else entirely here."))
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.SOURCE_CHANGED))

    def test_multi_match_in_different_contexts(self):
        txt = f"Promotion A: {RULE} Ends in May. " + FILLER + f" Promotion B: {RULE} Applies to new users."
        e = run(fact(), page(txt))
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.MULTI_MATCH))
        self.assertEqual(len(e.other_snippets), 1)

    def test_same_passage_in_both_layers_is_one_match(self):
        e = run(fact(), page(RULE, embedded=RULE))
        self.assertEqual(e.state, S.VERIFIED)

    def test_identical_duplicate_context_is_merged(self):
        block = f"Pricing table for all plans and regions. {RULE} Paid monthly, cancel any time, taxes extra."
        e = run(fact(), page(f"{block} {block}"))
        self.assertEqual(e.state, S.VERIFIED)

    def test_cross_domain_redirect(self):
        e = run(fact(), page(RULE, final_url="https://example.de/rules"))
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.REDIRECTED))

    def test_stale_legal_source(self):
        e = run(fact(source_domain_type="legal"), page(f"Updated 2022.07.29. {RULE}"))
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.STALE_SOURCE))
        self.assertEqual(e.source_date, "2022-07-29")

    def test_fresh_legal_source_is_verified(self):
        e = run(fact(source_domain_type="legal"), page(f"Updated 2026-09-01. {RULE}"))
        self.assertEqual(e.state, S.VERIFIED)

    def test_guard_word_makes_supporting_quote_ambiguous(self):
        e = run(fact(), page(f"{RULE} This does not apply to employers."))
        self.assertEqual((e.state, e.reason_code), (S.NEEDS_HUMAN, R.AMBIGUOUS_CONTEXT))
        self.assertIn("employers", e.guard_hits)

    def test_guard_word_does_not_weaken_contradicting_quote(self):
        e = run(fact(polarity=Polarity.CONTRADICTS), page(f"{RULE} Except for declared components."))
        self.assertEqual(e.state, S.VERIFIED)
        self.assertIn("except", e.guard_hits)

    def test_contradiction_pass(self):
        a = fact(cid="a", rule_topic="fee")
        b = fact(FEE, Polarity.CONTRADICTS, cid="b", rule_topic="fee")
        doc = page(RULE + " " + FILLER + " " + FEE)
        ev = find_contradictions([a, b], [run(a, doc), run(b, doc)])
        self.assertEqual({(e.state, e.reason_code) for e in ev}, {(S.NEEDS_HUMAN, R.CONTRADICTED)})


if __name__ == "__main__":
    unittest.main()
