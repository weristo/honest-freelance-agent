"""Public version of the holdout file (same rules as make_public_cases.py:
quotes cut to <=200 chars + SHA-256 of the full quote; no source text).

usage: python scripts/make_public_holdout.py PRIVATE_HOLDOUT.jsonl data/eval/holdout_public.jsonl
"""
import hashlib
import json
import sys

from make_public_cases import MAX_QUOTE

CLAIM_KEYS = ("role", "snapshot", "url", "final_url", "http", "fetched_at", "norm_sha256",
              "match_hits", "claimed_quote_not_in_source")
CASE_KEYS = ("id", "domain", "input", "gold", "gold_detail", "trap", "naive_false_go_trap")


def _bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() == "true"


def main(src: str, dst: str) -> None:
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            out = {k: case[k] for k in CASE_KEYS if k in case}
            out["naive_false_go_trap"] = _bool(case.get("naive_false_go_trap", False))
            out["split"] = "holdout"
            claims = []
            for c in case.get("claims") or []:
                q = c["quote"]
                pc = {"quote": q[:MAX_QUOTE], "quote_truncated": len(q) > MAX_QUOTE,
                      "quote_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest()}
                pc.update({k: c[k] for k in CLAIM_KEYS if k in c})
                pc["claimed_quote_not_in_source"] = _bool(c.get("claimed_quote_not_in_source", False))
                claims.append(pc)
            out["claims"] = claims
            g.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} holdout cases to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
