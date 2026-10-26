import unittest

from hfa.fetcher import UrlRefused, check_url
from hfa.gate import GoNotAllowed, decide, require_verified_go
from hfa.models import (ClaimKind, Evidence, EvidenceState as S, KillClass,
                        Polarity, Reason as R, Verdict as V)

from .helpers import fact


def ev(cid, state, reason=R.EXACT_MATCH):
    return Evidence(cid, state, reason)


class GateTests(unittest.TestCase):
    def test_go_requires_all_critical_facts_verified(self):
        c = [fact(cid="a"), fact(cid="b")]
        d = decide(V.GO, c, [ev("a", S.VERIFIED), ev("b", S.VERIFIED)], "mock")
        self.assertEqual(d.final, V.GO)
        require_verified_go(d)  # does not raise

    def test_g1_verified_contradiction_kills_even_if_proposer_says_go(self):
        c = [fact(cid="a", polarity=Polarity.CONTRADICTS)]
        d = decide(V.GO, c, [ev("a", S.VERIFIED)], "mock", KillClass.POLICY_REFUSAL)
        self.assertEqual((d.final, d.kill_class), (V.KILL, KillClass.POLICY_REFUSAL))

    def test_g2_go_with_zero_facts_is_refused(self):
        d = decide(V.GO, [], [], "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)

    def test_g3_needs_human_and_inference(self):
        d = decide(V.GO, [fact(cid="a")], [ev("a", S.NEEDS_HUMAN, R.MULTI_MATCH)], "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)
        c = [fact(cid="a"), fact(cid="i", kind=ClaimKind.INFERENCE)]
        d = decide(V.GO, c, [ev("a", S.VERIFIED), ev("i", S.NOT_APPLICABLE, R.NOT_A_FACT)], "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)

    def test_g4_unverified_blocks_go_but_keeps_kill(self):
        c = [fact(cid="a")]
        self.assertEqual(decide(V.GO, c, [ev("a", S.UNVERIFIED, R.NO_MATCH)], "mock").final, V.NEEDS_HUMAN)
        self.assertEqual(decide(V.KILL, c, [ev("a", S.UNVERIFIED, R.NO_MATCH)], "mock").final, V.KILL)
        self.assertEqual(decide(V.GO, c, [], "mock").final, V.NEEDS_HUMAN)  # missing evidence

    def test_g5_user_attested_needs_tick(self):
        u = fact(cid="u", kind=ClaimKind.USER_ATTESTED)
        c = [fact(cid="a"), u]
        evs = [ev("a", S.VERIFIED), ev("u", S.NOT_APPLICABLE, R.NOT_A_FACT)]
        self.assertEqual(decide(V.GO, c, evs, "mock").final, V.NEEDS_HUMAN)
        u.user_confirmed = True
        self.assertEqual(decide(V.GO, c, evs, "mock").final, V.GO)

    def test_decision_requires_llm_label(self):
        with self.assertRaises(ValueError):
            decide(V.GO, [], [], "")

    def test_go_enforcement_recomputes_from_evidence(self):
        c = [fact(cid="a")]
        d = decide(V.GO, c, [ev("a", S.UNVERIFIED, R.NO_MATCH)], "mock")
        with self.assertRaises(GoNotAllowed):
            require_verified_go(d)
        d.final = V.GO          # tampered: final says GO, evidence does not
        with self.assertRaises(GoNotAllowed):
            require_verified_go(d)


def public(host):
    return ["93.184.216.34"]


class SsrfTests(unittest.TestCase):
    def test_refused_urls(self):
        bad = ["http://example.org/", "ftp://example.org/", "file:///etc/passwd", "https://localhost/",
               "https://127.0.0.1/", "https://10.0.0.5/", "https://192.168.1.1/", "https://172.16.0.1/",
               "https://169.254.169.254/latest", "https://[::1]/", "https://[fc00::1]/",
               "https://user:pw@example.org/", "https://example.org:8443/", "https://printer.local/"]
        for u in bad:
            with self.assertRaises(UrlRefused, msg=u):
                check_url(u, public)

    def test_dns_to_private_address_refused(self):
        with self.assertRaises(UrlRefused):
            check_url("https://rebind.example/", lambda h: ["93.184.216.34", "10.1.2.3"])

    def test_public_https_allowed(self):
        check_url("https://example.org/rules", public)


if __name__ == "__main__":
    unittest.main()
