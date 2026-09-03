"""Builds the three prompt-length buckets Phase 2's sweep needs: short
(~128 tok), medium (~1k tok), long (~4k tok). Per the plan: "Prefill cost
scales with this; it's how you make the compute-bound claim visible."

Built from REAL schema content — the capstone's own per-database training
system prompts (fine_tuning/data_23db/split_manifest.json) — not synthetic
filler. A "long prompt" arm whose content is repeated garbage text would
measure prefill cost on something nothing resembling production traffic.
Token counts are measured with the actual model tokenizer, not estimated
from character counts: whitespace-heavy, punctuation-dense schema text does
not tokenize at a fixed chars-per-token ratio.

None of the 6 real training batches are naturally sized for these targets —
measured range is 1066-2799 tokens per batch (split_manifest.json's own
approx_tokens_range) — so each bucket slices or combines real batch text
rather than fabricating new content:

  - short:  one collection's own field list, sliced out of the case's
            database block. Restricted to held-out cases with exactly one
            gold_collection — a join question isn't answerable from a
            single collection's schema, so those cases are excluded from
            this bucket rather than given a schema that can't answer them.
  - medium: one whole database's schema block, sliced out of its batch —
            close to the ~1k/database share several batches land in.
  - long:   the case's own full batch prompt with a SECOND, unrelated
            batch's full schema appended as extra context, reaching the
            ~4k target with genuine schema text. As a side effect this is
            a more realistic "long context with irrelevant noise" scenario
            than clean synthetic padding — the model still has to find its
            actual target schema among the noise, same as a real large
            multi-tenant schema would present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from forge.capstone_bridge import CapstoneHarness
from forge.logging_config import get_logger

log = get_logger(__name__)

# Loose sanity bounds, not hard targets — real schema content doesn't land
# on an exact token count. Logged as a warning, not raised, if exceeded.
# Calibrated against real measured output, not guessed: a first pass used
# guessed ranges and both medium and long undershot them on a live run
# (medium's single-database slice landed 164-435 tokens against a naive
# 400-2500 guess; long's fixed "add exactly one other batch" landed
# 2200-2300 against a 2500-8000 guess). Medium now uses the case's own
# whole batch (natural range 1066-2799, per split_manifest.json's own
# approx_tokens_range across all 6 batches) instead of one database's
# slice; long accumulates other batches until it actually crosses
# LONG_TARGET_TOKENS, measured with the real tokenizer as it goes, instead
# of assuming a fixed number of batches will be enough.
BUCKET_SANITY_RANGES = {
    "short": (20, 300),
    "medium": (800, 3000),
    "long": (3000, 8000),
}
LONG_TARGET_TOKENS = 4000

_INSTRUCTIONS_SEPARATOR = re.compile(r"\n\n+")


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    database: str
    prompt_bucket: str
    system_prompt: str
    question: str
    token_count: int


def _split_paragraphs(system_prompt: str) -> list[str]:
    return _INSTRUCTIONS_SEPARATOR.split(system_prompt.strip())


def _find_database_block(paragraphs: list[str], database: str) -> str | None:
    header = f"{database}:"
    for paragraph in paragraphs:
        first_line = paragraph.split("\n", 1)[0].strip()
        if first_line == header:
            return paragraph
    return None


def _first_collection_line(database_block: str, gold_collection: str | None) -> str | None:
    """Returns the "- Name: { ... }" line for gold_collection if given and
    present, otherwise the block's first collection line. gold_collection
    may arrive as "database.Collection" (per the held-out cases' own
    gold_collections field) — only the part after the dot is matched
    against the schema's own collection names."""
    lines = [line for line in database_block.splitlines()[1:] if line.strip().startswith("- ")]
    if not lines:
        return None
    if gold_collection:
        collection_name = gold_collection.split(".")[-1]
        for line in lines:
            if line.strip().startswith(f"- {collection_name}:"):
                return line
    return lines[0]


def _build_short(
    case: dict[str, Any], preamble: str, database_block: str
) -> tuple[str, str] | None:
    gold_collections = case.get("gold_collections") or []
    if len(gold_collections) != 1:
        return None  # join question — not answerable from one collection's schema
    collection_line = _first_collection_line(database_block, gold_collections[0])
    if collection_line is None:
        return None
    header = database_block.split("\n", 1)[0]
    schema = f"{header}\n{collection_line}"
    system_prompt = f"{preamble}\n\nSchema (1 database, 1 collection):\n\n{schema}"
    return system_prompt, case["question"]


def _build_medium(case: dict[str, Any], batch_prompt: str) -> tuple[str, str]:
    """The case's own whole batch prompt, multi-database, unmodified — the
    natural per-batch token range (1066-2799, see split_manifest.json's own
    approx_tokens_range) lands close enough to ~1k that slicing to a single
    database (tried first, undershot to 164-435 on a live measurement) was
    an unnecessary complication."""
    return batch_prompt, case["question"]


def _schema_only(batch_prompt: str) -> str:
    """Strips a batch prompt down to just its database schema blocks —
    drops the preamble/header/footer, which would be redundant noise when
    concatenated onto a second batch's full prompt in _build_long."""
    paragraphs = _split_paragraphs(batch_prompt)
    return "\n\n".join(
        p for p in paragraphs if not p.startswith(("You are", "Schema (", "Note:", "Rules:"))
    )


def _build_long(
    case: dict[str, Any],
    own_batch_prompt: str,
    other_batches: list[dict[str, Any]],
    tokenizer: Any,
    target_tokens: int = LONG_TARGET_TOKENS,
) -> tuple[str, str]:
    """Appends OTHER batches' schema-only content — real content, not
    filler — one at a time, checking the real token count after each
    addition, until target_tokens is crossed or batches run out. A first
    version added exactly one fixed other batch and undershot (2200-2300
    measured vs. a ~4k target); accumulating dynamically against the actual
    tokenizer is what makes this reliably land near the target regardless
    of which specific batches happen to be involved for a given case."""
    system_prompt = own_batch_prompt
    for batch in other_batches:
        if len(tokenizer.encode(system_prompt)) >= target_tokens:
            break
        system_prompt = f"{system_prompt}\n\n{_schema_only(batch['system_prompt'])}"
    return system_prompt, case["question"]


def build_prompt_buckets(
    harness: CapstoneHarness, tokenizer: Any, n_per_bucket: int = 10, seed: int = 42
) -> dict[str, list[PromptCase]]:
    import json
    import random

    cases = json.loads(harness.rag_test_path.read_text(encoding="utf-8"))
    manifest = json.loads(harness.split_manifest_path.read_text(encoding="utf-8"))
    batches = manifest["batches"]
    db_to_batch_prompt = {db: b["system_prompt"] for b in batches for db in b["databases"]}
    db_to_batch_index = {db: b["index"] for b in batches for db in b["databases"]}

    rng = random.Random(seed)
    rng.shuffle(cases)

    buckets: dict[str, list[PromptCase]] = {"short": [], "medium": [], "long": []}

    for case in cases:
        if all(len(buckets[b]) >= n_per_bucket for b in buckets):
            break

        database = case["database"]
        batch_prompt = db_to_batch_prompt.get(database)
        if batch_prompt is None:
            continue
        paragraphs = _split_paragraphs(batch_prompt)
        preamble = paragraphs[0]
        database_block = _find_database_block(paragraphs, database)
        if database_block is None:
            continue

        if len(buckets["short"]) < n_per_bucket:
            built = _build_short(case, preamble, database_block)
            if built is not None:
                _append_bucket(buckets, "short", case, tokenizer, *built)

        if len(buckets["medium"]) < n_per_bucket:
            built = _build_medium(case, batch_prompt)
            _append_bucket(buckets, "medium", case, tokenizer, *built)

        if len(buckets["long"]) < n_per_bucket:
            own_batch_index = db_to_batch_index[database]
            other_batches = [b for b in batches if b["index"] != own_batch_index]
            rng.shuffle(other_batches)
            if other_batches:
                built = _build_long(case, batch_prompt, other_batches, tokenizer)
                _append_bucket(buckets, "long", case, tokenizer, *built)

    for bucket_name, prompt_cases in buckets.items():
        if len(prompt_cases) < n_per_bucket:
            log.warning(
                "prompts.bucket_under_target",
                bucket=bucket_name,
                built=len(prompt_cases),
                target=n_per_bucket,
            )

    return buckets


def _append_bucket(
    buckets: dict[str, list[PromptCase]],
    bucket_name: str,
    case: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    question: str,
) -> None:
    token_count = len(tokenizer.encode(system_prompt + question))
    low, high = BUCKET_SANITY_RANGES[bucket_name]
    if not low <= token_count <= high:
        log.warning(
            "prompts.token_count_outside_sanity_range",
            bucket=bucket_name,
            case_id=case["id"],
            token_count=token_count,
            expected_range=(low, high),
        )
    buckets[bucket_name].append(
        PromptCase(
            case_id=str(case["id"]),
            database=case["database"],
            prompt_bucket=bucket_name,
            system_prompt=system_prompt,
            question=question,
            token_count=token_count,
        )
    )
