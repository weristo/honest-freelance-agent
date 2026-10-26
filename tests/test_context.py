"""p3 conditions: the verifier compares with its OWN fetch, never with the text
given to the model; quotes from the cut-off part of a long source are marked
in the proposer layer and end UNVERIFIED; the truncation is logged."""
import json
import os
import tempfile
import unittest

from hfa import holdout
from hfa.fetcher import FixtureFetcher
from hfa.models import EvidenceState as S, Reason as R, Verdict as V
from hfa.pipeline import run
from hfa.proposer import Proposer
from hfa.proposer.base import OUTSIDE_CONTEXT_MARK, LLMResponse
from hfa.proposer.budget import Budget
from hfa.proposer.prompt import SOURCE_TEXT_LIMIT

from .helpers import FILLER, RULE, StaticFetcher, page

URL = "https://example.org/rules"
FAKE = "Every submission may skip the working build if the team is in a hurry today."


class ScriptedProvider:
    provider, model = "anthropic", "claude-haiku-4-5"

    def __init__(self, quotes):
        self.quotes, self.prompts = quotes, []

    @property
    def label(self):
        return "anthropic:claude-haiku-4-5"

    def complete(self, system, user, max_tokens):
        self.prompts.append(user)
        claims = [{"claim_id": f"c{i}", "text": "rule", "kind": "FACT", "critical": True, "quote": q,
                   "source_url": URL, "polarity": "SUPPORTS"} for i, q in enumerate(self.quotes)]
        return LLMResponse(json.dumps({"decision": "GO", "rationale": "x", "claims": claims}), 100, 50)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "trace.jsonl")
        self.budget = Budget(os.path.join(self.tmp.name, "b.json"))

    def _run(self, quotes, model_text, verifier_doc):
        client = ScriptedProvider(quotes)
        p = Proposer(client, self.budget, usage_log=self.log, source_text=lambda u: model_text)
        return client, p, run("job", [URL], p, StaticFetcher(verifier_doc), trace_path=self.log)

    def test_verifier_uses_its_own_fetch_not_the_model_text(self):
        # The model is shown a MODIFIED text and quotes from it.
        modified = FILLER + " " + FAKE + " " + FILLER
        client, _, res = self._run([FAKE], modified, page(RULE))
        self.assertIn(FAKE, client.prompts[0])                 # the model really saw the fake text
        e = res.decision.evidence[0]
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.NO_MATCH))
        self.assertNotEqual(res.decision.final, V.GO)

    def test_same_text_for_model_and_verifier_is_fine(self):
        doc = page(RULE)
        _, _, res = self._run([RULE], doc.visible, doc)
        self.assertEqual(res.decision.evidence[0].state, S.VERIFIED)
        self.assertEqual(res.decision.final, V.GO)

    def test_quote_from_cut_off_part_is_marked_and_unverified(self):
        head = ("Section text without the rule. " * 500)[:SOURCE_TEXT_LIMIT + 500]
        long_text = head + " " + RULE + " " + FILLER        # RULE lies beyond the limit
        doc = page("", visible=long_text)
        client, p, res = self._run([RULE], long_text, doc)
        self.assertNotIn(RULE, client.prompts[0])               # really cut off
        self.assertEqual(p.last_outside, ["c0"])
        claim = res.decision.claims[0]
        self.assertTrue(claim.text.startswith(OUTSIDE_CONTEXT_MARK))
        e = res.decision.evidence[0]
        self.assertEqual((e.state, e.reason_code), (S.UNVERIFIED, R.QUOTE_TOO_SHORT))
        self.assertNotEqual(res.decision.final, V.GO)
        with open(self.log) as f:
            events = [json.loads(x) for x in f]
        ctx = next(x for x in events if x.get("event") == "source_context")
        self.assertEqual(ctx["sources"][0]["passed_chars"], SOURCE_TEXT_LIMIT)
        self.assertGreater(ctx["sources"][0]["original_chars"], SOURCE_TEXT_LIMIT)
        self.assertTrue(ctx["sources"][0]["truncated"])
        self.assertTrue(any(x.get("event") == "quote_outside_context" and x["claims"] == ["c0"] for x in events))

    def test_quote_inside_passed_part_and_paraphrase_are_not_marked(self):
        long_text = RULE + " " + ("Filler sentence for length. " * 600)
        doc = page("", visible=long_text)
        _, p, res = self._run([RULE, "A paraphrase that appears nowhere in the source text."], long_text, doc)
        self.assertEqual(p.last_outside, [])
        self.assertEqual(res.decision.evidence[0].state, S.VERIFIED)
        self.assertEqual(res.decision.evidence[1].reason_code, R.NO_MATCH)
        ctx = p.last_context[0]
        self.assertEqual((ctx["truncated"], ctx["original_chars"] > SOURCE_TEXT_LIMIT), (True, True))

    def test_frozen_code_untouched(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "data", "eval", "FREEZE_GATE.json")) as f:
            self.assertEqual(holdout.decision_code_sha256(), json.load(f)["decision_code_sha256"])


if __name__ == "__main__":
    unittest.main()
