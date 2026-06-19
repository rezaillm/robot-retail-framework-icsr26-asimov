"""Real LLM-only retail assistant using the Anthropic API.

Requires ANTHROPIC_API_KEY to be set in the environment.

Use LLMOnlyRetailAssistantReal as a drop-in replacement for LLMOnlyRetailAssistant. 
The neuro-symbolic condition is unchanged.
"""
from __future__ import annotations

import json
from random import Random
from typing import Sequence

from .models import InteractionState, Product, RecommendationResult

# Unicode quote characters that appear when an API key is copy-pasted from a
# PDF or rich-text editor.  httpx cannot encode them as ASCII header values.
_UNICODE_QUOTES = str.maketrans("", "", "‘’“”′″")


def _clean_api_key() -> str:
    import os

    raw = os.environ.get("ANTHROPIC_API_KEY", "")
    cleaned = raw.translate(_UNICODE_QUOTES).strip()
    if not cleaned:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Export it before running with --real-llm."
        )
    try:
        cleaned.encode("ascii")
    except UnicodeEncodeError:
        raise EnvironmentError(
            f"ANTHROPIC_API_KEY contains non-ASCII characters even after stripping "
            f"Unicode quotes. Re-export the key using plain straight quotes.\n"
            f"First 20 chars (repr): {repr(cleaned[:20])}"
        )
    return cleaned


_SYSTEM_PROMPT = """\
You are a retail assistant helping a customer find trousers.
Given the customer's request and a product catalogue, pick ONE product to \
recommend and give a friendly, natural response.

Reply with valid JSON only — no prose outside the JSON block:
{
  "product_id": "<id from catalogue>",
  "response": "<natural-language response to the customer>",
  "explanation": "<one-sentence reason for the choice>"
}"""


def _catalog_text(catalog: Sequence[Product]) -> str:
    lines = []
    for p in catalog:
        if not p.available:
            continue
        tags = []
        if p.premium:
            tags.append("premium")
        if p.promotion:
            tags.append("on promotion")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"  {p.id}: {p.name} | €{p.price:.2f} | style={p.style} | "
            f"comfort={p.comfort}/5{tag_str}"
        )
    return "\n".join(lines)


class LLMOnlyRetailAssistantReal:
    """LLM-only baseline using a live Claude API call.

    This replaces the stochastic heuristic in LLMOnlyRetailAssistant with an
    actual language model so the comparison in Table 1 is honest: the
    neuro-symbolic condition adds a PDDL planning layer on top of the same
    quality of language understanding, rather than comparing against a
    deliberately-impaired random baseline.
    """

    def __init__(
        self,
        catalog: Sequence[Product],
        rng: Random | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.catalog = list(catalog)
        self.rng = rng or Random(0)
        self.model = model
        self._catalog_index = {p.id: p for p in self.catalog}

    def recommend(self, state: InteractionState) -> RecommendationResult:
        """Call Claude with the original utterance and the catalogue."""
        import anthropic  # imported lazily so the module loads without the package

        utterance = state.utterance or "(no utterance stored — using mental model summary)"
        user_message = (
            f"Customer request: {utterance}\n\n"
            f"Available products:\n{_catalog_text(self.catalog)}"
        )

        client = anthropic.Anthropic(api_key=_clean_api_key())
        message = client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = message.content[0].text.strip()

        product, response, explanation = self._parse_response(raw)
        return RecommendationResult(
            condition="llm_only",
            product=product,
            response=response,
            explanation=explanation,
            plan=[],
            metadata={"llm_model": self.model, "raw_response": raw},
        )

    def _parse_response(self, raw: str) -> tuple[Product, str, str]:
        """Parse the JSON response; fall back to cheapest available on failure."""
        try:
            # Strip markdown code fences if the model emits them
            text = raw
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
            product_id = data["product_id"]
            product = self._catalog_index[product_id]
            return product, data["response"], data["explanation"]
        except Exception:
            fallback = min(
                [p for p in self.catalog if p.available],
                key=lambda p: p.price,
            )
            return fallback, f"I suggest {fallback.name}.", "Chosen as cheapest available."
