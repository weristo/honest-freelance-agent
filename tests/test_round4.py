import os
import unittest
from datetime import datetime, timezone

from hfa import evalrun
from hfa.calc import has_money
from hfa.dates import dates_in_text, date_from_url, source_date
from hfa.fetcher import FixtureFetcher, load_snapshot
from hfa.models import EvidenceState as S, Reason as R
from hfa.verifier import verify

from .helpers import FILLER, RULE, StaticFetcher, fact, page

NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)
NAV = "https://nav.gov.hu/ado/example-page"
PRIVATE_SNAPSHOTS = os.environ.get("HFA_PRIVATE_SNAPSHOTS", os.path.join("data", "private", "snapshots"))


def policy(text, url=NAV, **kw):
    return verify(fact(source_url=url), StaticFetcher(page(text, url=url, final_url=url, **kw)), now=NOW)


class DateDetectionTests(unittest.TestCase):
    def test_formats(self):
        txt = ("Adó 2022.07.29. Hatályos: 2025. január 1. Stand: 03/2024 gültig ab 1. Juli 2023 "
               "Effective date: March 5, 2024. Last updated 12-Aug-2025. Published 4 May 2021. am 29.07.2022")
        found = set(dates_in_text(txt))
        for d in ("2022-07-29", "2025-01-01", "2024-03-31", "2023-07-01", "2024-03-05", "2025-08-12",
                  "2021-05-04"):
            self.assertIn(d, found)

    def test_url_year_and_latest_wins(self):
        self.assertEqual(date_from_url("https://nav.gov.hu/pfile/file?path=/inf_fuz/2022/99-booklet"), "2022-12-31")
        self.assertIsNone(date_from_url("https://nav.gov.hu/ado/page-20221"))
        self.assertEqual(date_from_url("https://nav.gov.hu/docs/2022/booklet.pdf"), "2022-12-31")
        d, origins = source_date("Updated 2021-01-05. Revised 2025-09-01.", ["https://x.gov/2019/a"],
                                 "Wed, 01 Oct 2025 08:00:00 GMT")
        self.assertEqual(d, "2025-10-01")
        self.assertEqual(len(origins), 4)


class StalenessTests(unittest.TestCase):
    def test_old_policy_source_is_stale_and_blocks_g1(self):
        e = policy(f"Adó 2022.07.29. {RULE}")
        self.assertEqual((e.state, e.reason_code, e.source_date), (S.NEEDS_HUMAN, R.STALE_SOURCE, "2022-07-29"))

    def test_fresh_policy_source_is_not_stale(self):
        e = policy(f"Hatályos: 2026. január 1. Korábbi szabály: 2019.03.01. {RULE}")
        self.assertEqual((e.state, e.date_flag, e.source_date), (S.VERIFIED, "DATED", "2026-01-01"))

    def test_undated_policy_source_is_verified_but_flagged(self):
        e = policy(RULE)
        self.assertEqual((e.state, e.date_flag), (S.VERIFIED, "UNDATED"))
        self.assertIn("UNDATED", e.detail)

    def test_url_year_makes_it_stale(self):
        url = "https://nav.gov.hu/docs/2022/booklet"
        e = policy(RULE, url=url)
        self.assertEqual((e.reason_code, e.source_date), (R.STALE_SOURCE, "2022-12-31"))

    def test_last_modified_header(self):
        self.assertEqual(policy(RULE, last_modified="Mon, 03 Jan 2022 10:00:00 GMT").reason_code, R.STALE_SOURCE)
        e = policy(f"Adó 2022.07.29. {RULE}", last_modified="Thu, 01 Oct 2026 10:00:00 GMT")
        self.assertEqual((e.state, e.source_date), (S.VERIFIED, "2026-10-01"))   # latest wins

    def test_non_policy_source_is_not_dated(self):
        e = verify(fact(), StaticFetcher(page(f"Adó 2022.07.29. {RULE}")), now=NOW)
        self.assertEqual((e.state, e.date_flag), (S.VERIFIED, None))

    @unittest.skipUnless(os.path.exists(os.path.join(PRIVATE_SNAPSHOTS, "T36.txt")),
                         "private snapshot T36.txt not available (third-party text, not in the repo)")
    def test_t36_real_snapshot_is_stale(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import json
        with open(os.path.join(here, "data", "eval", "cases_public.jsonl"), encoding="utf-8") as f:
            case = next(c for c in map(json.loads, f) if c["id"] == "T36")
        self.assertEqual(case["gold"], "NEEDS_HUMAN")
        cl = case["claims"][0]
        fetcher = FixtureFetcher({cl["url"]: os.path.join(PRIVATE_SNAPSHOTS, cl["snapshot"])})
        e = verify(fact(cl["quote"], source_url=cl["url"]), fetcher)
        self.assertEqual(e.reason_code, R.STALE_SOURCE, e.date_origins)


class G6YearTests(unittest.TestCase):
    def _decide(self, user_text):
        from hfa.gate import decide
        from hfa.models import Claim, ClaimKind, Evidence, Verdict as V
        u = Claim("u1", user_text, kind=ClaimKind.USER_ATTESTED, user_confirmed=True)
        return decide(V.GO, [fact(cid="c1"), u], [Evidence("c1", S.VERIFIED, R.EXACT_MATCH),
                                                  Evidence("u1", S.NOT_APPLICABLE, R.NOT_A_FACT)],
                      "mock", request_text="Can I register?")

    def test_year_only_does_not_trigger_g6(self):
        d = self._decide("2026-os bevételem alapján jogosult vagyok.")
        self.assertEqual(d.final.value, "GO")
        self.assertFalse(any(g.startswith("G6 fail") for g in d.gate_log))

    def test_year_plus_amount_triggers_g6(self):
        d = self._decide("2026-os bevételem 6 M Ft.")
        self.assertEqual(d.final.value, "NEEDS_HUMAN")
        self.assertTrue(any(g.startswith("G6 fail") for g in d.gate_log))

    def test_year_like_number_with_currency_counts(self):
        from hfa.calc import has_number
        self.assertTrue(has_number("I earned 2026 EUR"))
        self.assertTrue(has_number("I earned €2026"))
        self.assertTrue(has_number("a fee of 2000%"))
        self.assertFalse(has_number("since 1999 and in 2100"))
        self.assertTrue(has_number("6 months"))


class G6NarrowingTests(unittest.TestCase):
    def test_money_detection_in_request(self):
        self.assertTrue(has_money("tavaly 23 000 € forgalom"))
        self.assertTrue(has_money("a 30 USD job"))
        self.assertTrue(has_money("20 M Ft alatt"))
        self.assertFalse(has_money("GitHub Sponsors fees are 100% passed on"))
        self.assertFalse(has_money("open-source developer in Hungary"))

    def test_t31_input_has_no_money_and_t35_has(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import json
        with open(os.path.join(here, "data", "eval", "cases_public.jsonl"), encoding="utf-8") as f:
            cases = {c["id"]: c for c in map(json.loads, f)}
        self.assertFalse(has_money(cases["T31"]["input"]))
        self.assertTrue(has_money(cases["T35"]["input"]))
        self.assertNotIn("calc", cases["T31"])
        self.assertNotIn("calc", cases["T35"])


if __name__ == "__main__":
    unittest.main()
