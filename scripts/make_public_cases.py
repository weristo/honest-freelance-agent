"""Build the public eval file from the private one.

The private file and the source snapshots contain third-party text and stay
out of the repository (data/private/, git-ignored). The public file keeps,
per quote, at most 200 characters plus the SHA-256 of the full quote, and the
hashes recorded at fetch time.

usage: python scripts/make_public_cases.py PRIVATE.jsonl data/eval/cases_public.jsonl
"""
import hashlib
import json
import sys

MAX_QUOTE = 200
CLAIM_KEYS = ("role", "snapshot", "url", "http", "fetched_at", "norm_sha256",
              "match_hits", "match_layers", "snippet_sha256")
CASE_KEYS = ("id", "domain", "input", "gold", "gold_detail", "trap",
             "naive_false_go_trap", "user_attested")


def public_claim(c: dict) -> dict:
    q = c["quote"]
    out = {"quote": q[:MAX_QUOTE], "quote_truncated": len(q) > MAX_QUOTE,
           "quote_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest()}
    out.update({k: c[k] for k in CLAIM_KEYS if k in c})
    return out


def public_source(s: dict) -> dict:
    return {k: s[k] for k in ("snapshot", "url", "http", "fetched_at", "norm_sha256") if k in s}


def main(src: str, dst: str) -> None:
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            out = {k: case[k] for k in CASE_KEYS if k in case}
            out["split"] = "holdout" if case["id"].startswith("R") else "dev"
            out["claims"] = [public_claim(c) for c in case.get("claims") or []]
            for k in ("source", "source2"):
                if case.get(k):
                    out[k] = public_source(case[k])
            g.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} cases to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
