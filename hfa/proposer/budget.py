"""Application-level LLM spending cap.

Hard caps: USD 1.00 per UTC day and USD 15.00 in total. Env variables can only
LOWER them (HFA_BUDGET_DAILY_USD, HFA_BUDGET_TOTAL_USD). Spend is computed
from the provider's `usage` field (input/output tokens) times the price table,
and persisted in a JSON state file under data/ (git-ignored).

Before a call we reserve the worst case (estimated input + max_tokens of
output). If that does not fit, no call is made and the caller gets the
explicit state "budget_exhausted" (never an empty answer).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

HARD_DAILY_USD = 1.00
HARD_TOTAL_USD = 15.00
DEFAULT_STATE_PATH = os.path.join("data", "llm_budget.json")

# USD per 1M tokens (input, output). Unknown models are refused, because we
# cannot budget what we cannot price (override with HFA_PRICE_IN/HFA_PRICE_OUT).
PRICES_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class BudgetExhausted(Exception):
    pass


class UnknownPrice(Exception):
    pass


def _env_cap(name: str, hard: float) -> float:
    try:
        v = float(os.environ.get(name, hard))
    except ValueError:
        v = hard
    return max(0.0, min(v, hard))


def price_for(model: str) -> tuple[float, float]:
    pin, pout = os.environ.get("HFA_PRICE_IN"), os.environ.get("HFA_PRICE_OUT")
    if pin and pout:
        return float(pin), float(pout)
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    raise UnknownPrice(f"no price for model '{model}'; set HFA_PRICE_IN/HFA_PRICE_OUT")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class BudgetStatus:
    day: str
    spent_today: float
    spent_total: float
    daily_cap: float
    total_cap: float

    @property
    def exhausted(self) -> bool:
        return self.spent_today >= self.daily_cap or self.spent_total >= self.total_cap


class Budget:
    def __init__(self, path: str = DEFAULT_STATE_PATH, clock=None):
        self.path = path
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.daily_cap = _env_cap("HFA_BUDGET_DAILY_USD", HARD_DAILY_USD)
        self.total_cap = _env_cap("HFA_BUDGET_TOTAL_USD", HARD_TOTAL_USD)

    def _today(self) -> str:
        return self.clock().strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"total_usd": 0.0, "days": {}, "calls": 0}
        except ValueError:
            # A corrupt state file must not silently reset spend to zero.
            raise BudgetExhausted("budget state file unreadable; refusing to call")

    def _save(self, st: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def status(self) -> BudgetStatus:
        st = self._load()
        day = self._today()
        return BudgetStatus(day, float(st["days"].get(day, 0.0)), float(st["total_usd"]),
                            self.daily_cap, self.total_cap)

    def check(self, model: str, est_input_tokens: int, max_output_tokens: int) -> None:
        worst = cost_usd(model, est_input_tokens, max_output_tokens)
        s = self.status()
        if s.spent_today + worst > s.daily_cap or s.spent_total + worst > s.total_cap:
            raise BudgetExhausted(
                f"budget_exhausted: today ${s.spent_today:.4f}/{s.daily_cap:.2f}, "
                f"total ${s.spent_total:.4f}/{s.total_cap:.2f}, next call up to ${worst:.4f}")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        c = cost_usd(model, input_tokens, output_tokens)
        st = self._load()
        day = self._today()
        st["days"][day] = round(float(st["days"].get(day, 0.0)) + c, 6)
        st["total_usd"] = round(float(st["total_usd"]) + c, 6)
        st["calls"] = int(st.get("calls", 0)) + 1
        self._save(st)
        return c


def estimate_tokens(text: str) -> int:
    # Conservative: ~3 chars per token (real tokenizers average ~4 for English).
    return len(text) // 3 + 1


def remaining(b: Optional[Budget] = None) -> BudgetStatus:
    return (b or Budget()).status()
