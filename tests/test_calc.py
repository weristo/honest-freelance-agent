import random
import unittest
from decimal import Decimal
from fractions import Fraction

from hfa.calc import CalcError, evaluate, has_amount, parse_literal, run_calc
from hfa.gate import GoNotAllowed, decide, require_verified_go
from hfa.models import Claim, ClaimKind, Evidence, EvidenceState as S, KillClass, Polarity, Reason as R, Verdict as V

from .helpers import fact

D = Decimal


class EvaluatorTests(unittest.TestCase):
    def test_basic_arithmetic_and_precedence(self):
        self.assertEqual(evaluate("30 - max(0.10*30, 5)"), D("25"))
        self.assertEqual(evaluate("2 + 3 * 4"), D("14"))
        self.assertEqual(evaluate("(2 + 3) * 4"), D("20"))
        self.assertEqual(evaluate("-3 + 10 / 4"), D("-0.5"))
        self.assertEqual(evaluate("min(1, 2, -7)"), D("-7"))
        self.assertEqual(evaluate("30 × 0.1 − 1 ÷ 2"), D("2.5"))

    def test_decimal_not_float(self):
        self.assertEqual(evaluate("0.1 + 0.2"), D("0.3"))

    def test_names(self):
        self.assertEqual(evaluate("price - fee", {"price": D(30), "fee": D(5)}), D(25))
        with self.assertRaises(CalcError):
            evaluate("price - unknown", {"price": D(30)})

    def test_attacks_are_rejected(self):
        attacks = [
            "__import__('os').system('id')", "().__class__", "price.__class__", "a.b", "open('x')",
            "eval('1')", "exec", "lambda: 1", "1 if 1 else 2", "[1, 2]", "{'a': 1}", "1; 2", "1 ** 2",
            "2 ** 999999", "1 // 2", "1 % 2", "'abc'", "max", "max(", "max()", "max(1,)", "(1", "1)", "",
            "   ", "1e999", "0x10", "1_000", "PRICE", "pow(2, 3)", "abs(-1)", "\x00", "1 2", "--",
            "1/0", "5 / (2 - 2)", "max(1, 2) / min(0, 0)",
        ]
        for a in attacks:
            with self.assertRaises(CalcError, msg=repr(a)):
                evaluate(a, {"price": D(1)})

    def test_division_by_zero(self):
        with self.assertRaises(CalcError):
            evaluate("10 / (5 - 5)")

    def test_limits(self):
        with self.assertRaises(CalcError):
            evaluate("1+" * 200 + "1")                      # too long
        with self.assertRaises(CalcError):
            evaluate("(" * 25 + "1" + ")" * 25)              # too deep
        with self.assertRaises(CalcError):
            evaluate("999999999 * 999999999 * 999999999")  # out of range
        with self.assertRaises(CalcError):
            evaluate("-" * 100 + "1")                       # unary chain too deep

    def test_fuzz_random_strings_only_raise_calcerror(self):
        rng = random.Random(1026)
        alphabet = "0123456789.+-*/(),_ abcmaxin×÷−'\"[]{}:;=<>!@#$%^&|\\\x00é"
        for _ in range(3000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            try:
                v = evaluate(s, {"a": D(2), "b": D(3)})
                self.assertIsInstance(v, Decimal)
            except CalcError:
                pass

    def test_fuzz_random_expressions_match_fraction_reference(self):
        rng = random.Random(42)

        def gen(depth):
            if depth == 0 or rng.random() < 0.3:
                n = rng.randint(0, 999)
                return str(n), Fraction(n)
            op = rng.choice("+-*/mM")
            (ls, lv), (rs, rv) = gen(depth - 1), gen(depth - 1)
            if op in "mM":
                f = "max" if op == "m" else "min"
                return f"{f}({ls}, {rs})", (max if op == "m" else min)(lv, rv)
            if op == "/" and rv == 0:
                rs, rv = "1", Fraction(1)
            val = {"+": lv + rv, "-": lv - rv, "*": lv * rv, "/": lv / rv if op == "/" else None}[op]
            return f"({ls} {op} {rs})", val

        checked = 0
        for _ in range(500):
            s, ref = gen(4)
            try:
                v = evaluate(s)
            except CalcError:
                continue                                    # length / range limits
            self.assertAlmostEqual(float(v), float(ref), places=6, msg=s)
            checked += 1
        self.assertGreater(checked, 300)

    def test_parse_literal(self):
        self.assertEqual(parse_literal("10%"), D("0.10"))
        self.assertEqual(parse_literal("$5.00 USD"), D("5.00"))
        self.assertEqual(parse_literal("3,40%"), D("0.034"))
        self.assertEqual(parse_literal("25 000"), D("25000"))
        self.assertEqual(parse_literal("1,642"), D("1642"))
        self.assertEqual(parse_literal("£250"), D("250"))
        with self.assertRaises(CalcError):
            parse_literal("no number")

    def test_has_amount(self):
        self.assertTrue(has_amount("10% or $5.00 USD"))
        self.assertTrue(has_amount("25 000 Euro"))
        self.assertFalse(has_amount("within 30 days of the announcement"))


FEE = "The fee for fixed price projects is 10% or $5.00 USD, whichever is greater."


def t16_claims(claimed_net="25", user_text="The job pays 30 USD; my own cost is 26 USD.", fee_literal="10%"):
    c1 = fact(FEE, cid="c1")
    u1 = Claim("u1", user_text, kind=ClaimKind.USER_ATTESTED, user_confirmed=True)
    k1 = Claim("k1", "net vs cost", kind=ClaimKind.CALC, calc={
        "inputs": {"rate": {"literal": fee_literal, "source": "c1"}, "minfee": {"literal": "$5.00", "source": "c1"},
                   "price": {"literal": "30", "source": "u1"}, "cost": {"literal": "26", "source": "u1"}},
        "steps": [["fee", "max(rate*price, minfee)"], ["net", "price - fee"]],
        "claimed": {"net": claimed_net}, "check": ["net", ">=", "cost"], "on_fail": "NOT_WORTH"})
    return [c1, u1, k1]


def evs(state=S.VERIFIED):
    return [Evidence("c1", state, R.EXACT_MATCH if state == S.VERIFIED else R.NO_MATCH),
            Evidence("u1", S.NOT_APPLICABLE, R.NOT_A_FACT), Evidence("k1", S.NOT_APPLICABLE, R.NOT_A_FACT)]


class CalcGateTests(unittest.TestCase):
    def test_t16_net_below_cost_kills_even_if_proposer_says_go(self):
        d = decide(V.GO, t16_claims(), evs(), "mock")
        self.assertEqual((d.final, d.kill_class), (V.KILL, KillClass.NOT_WORTH))
        self.assertEqual(d.calcs[0]["values"]["net"], "25.00")

    def test_calc_kill_ignores_registry(self):
        claims = t16_claims()
        claims[0].source_url = "https://example.org/legal/terms"   # a policy document
        claims[0].polarity = Polarity.CONTRADICTS
        d = decide(V.GO, claims, evs(), "mock")
        self.assertEqual(d.kill_class, KillClass.NOT_WORTH)

    def test_mismatch_needs_human(self):
        d = decide(V.GO, t16_claims(claimed_net="27"), evs(), "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)
        self.assertTrue(any("CALC_MISMATCH" in g for g in d.gate_log))

    def test_input_from_unverified_fact_is_invalid(self):
        d = decide(V.GO, t16_claims(), evs(S.UNVERIFIED), "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)
        self.assertTrue(any("CALC_INVALID" in g for g in d.gate_log))

    def test_literal_must_appear_in_source(self):
        d = decide(V.GO, t16_claims(fee_literal="8%"), evs(), "mock")
        self.assertTrue(any("CALC_INVALID" in g for g in d.gate_log))

    def test_passing_calc_allows_go_and_enforcement(self):
        claims = t16_claims(claimed_net="25", user_text="The job pays 30 USD; my own cost is 20 USD.")
        claims[2].calc["inputs"]["cost"]["literal"] = "20"
        d = decide(V.GO, claims, evs(), "mock")
        self.assertEqual(d.final, V.GO)
        require_verified_go(d)

    def test_g6_quote_percentage_alone_does_not_trigger(self):
        d = decide(V.GO, [fact(FEE, cid="c1")], [Evidence("c1", S.VERIFIED, R.EXACT_MATCH)], "mock",
                   request_text="Can I use this platform?")
        self.assertEqual(d.final, V.GO)

    def test_g6_money_in_request_without_calc_needs_human(self):
        d = decide(V.GO, [fact(FEE, cid="c1")], [Evidence("c1", S.VERIFIED, R.EXACT_MATCH)], "mock",
                   request_text="A 30 USD logo job. Take it?")
        self.assertEqual(d.final, V.NEEDS_HUMAN)
        self.assertTrue(any(g.startswith("G6 fail") for g in d.gate_log))
        d.final = V.GO                                   # tampering does not help
        with self.assertRaises(GoNotAllowed):
            require_verified_go(d)

    def test_g6_user_number_without_calc_needs_human(self):
        u = Claim("u1", "Last year's turnover was 23 000 EUR.", kind=ClaimKind.USER_ATTESTED, user_confirmed=True)
        d = decide(V.GO, [fact(cid="c1"), u], [Evidence("c1", S.VERIFIED, R.EXACT_MATCH),
                                               Evidence("u1", S.NOT_APPLICABLE, R.NOT_A_FACT)], "mock")
        self.assertEqual(d.final, V.NEEDS_HUMAN)

    def test_go_without_amount_needs_no_calc(self):
        d = decide(V.GO, [fact(cid="c1")], [Evidence("c1", S.VERIFIED, R.EXACT_MATCH)], "mock")
        self.assertEqual(d.final, V.GO)

    def test_bad_on_fail_and_bad_check(self):
        claims = t16_claims()
        claims[2].calc["on_fail"] = "POLICY_REFUSAL"
        self.assertTrue(any("CALC_INVALID" in g for g in decide(V.GO, claims, evs(), "mock").gate_log))
        claims = t16_claims()
        claims[2].calc["check"] = ["net", "==", "cost"]
        self.assertTrue(any("CALC_INVALID" in g for g in decide(V.GO, claims, evs(), "mock").gate_log))
        claims = t16_claims()
        claims[2].calc["steps"] = [["fee", "__import__('os')"]]
        self.assertTrue(any("CALC_INVALID" in g for g in decide(V.GO, claims, evs(), "mock").gate_log))

    def test_run_calc_rejects_inference_source(self):
        claims = t16_claims()
        claims[1].kind = ClaimKind.INFERENCE
        r = run_calc(claims[2], {c.claim_id: c for c in claims}, {e.claim_id: e for e in evs()})
        self.assertEqual(r.status, "INVALID")


if __name__ == "__main__":
    unittest.main()
