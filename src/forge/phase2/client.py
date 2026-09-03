"""The one client every arm goes through. Streams a chat completion and
records, per token-bearing chunk, the wall-clock monotonic time it arrived —
that raw timeline is what result_schema.RequestResult stores, and what
metrics.py later turns into TTFT/ITL percentiles.

Assumption worth stating plainly: one content-bearing SSE chunk is treated
as one token-arrival event. This holds for mlx_lm.server, Ollama, and vLLM
in normal (non-batched-chunk) operation, but isn't a protocol guarantee — a
server is free to coalesce multiple tokens into one chunk. If a specific
arm's ITL numbers look implausibly smooth relative to its known
tokens/sec, check this assumption first before trusting the percentiles.

Retry/backoff and connection pooling are the openai SDK's own responsibility
here (AsyncOpenAI reuses an httpx connection pool across calls, and
max_retries below hands off to its built-in exponential backoff) rather than
reimplemented — per AGENTS.md, this is exactly the kind of provider-adapter
plumbing that doesn't need a bespoke implementation.

Every request sets `stop=STOP_SEQUENCES` — not optional, not arm-specific.
This model's chat turns end with `<|im_end|>`, but Qwen2.5's config.json
eos_token_id points at `<|endoftext|>` instead; mlx_lm's TokenizerWrapper
only checks that single configured id (a known upstream gap — the capstone
repo's own spot_check.py works around it the same way, citing mlx-lm issue
#973), so without an explicit stop sequence the model correctly emits
`<|im_end|>` and generation just keeps going past it. Confirmed live against
a real mlx_lm.server: the identical request produced ~100 tokens of
`!<|im_end|>!<|im_end|>...` garbage without this, and the correct
`db.singer.count_documents({})` with it. `stop` is a standard
chat-completions field every arm here understands, so this one line fixes
mlx_lm, Ollama, vLLM, and any hosted API uniformly — no per-arm special
casing needed.
"""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from forge.hardware import HardwareInfo, get_hardware_info
from forge.logging_config import get_logger
from forge.phase2.arms import ArmConfig
from forge.phase2.result_schema import RequestResult

log = get_logger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT_S = 120.0
# Matches fine_tuning/spot_check.py's STOP_MARKERS in the capstone repo exactly —
# same model, same known mlx-lm EOS gap, same fix. See module docstring.
STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>"]


def make_client(arm_config: ArmConfig, max_retries: int = DEFAULT_MAX_RETRIES) -> AsyncOpenAI:
    # Local servers ignore api_key entirely; the SDK just requires a non-empty string.
    api_key = arm_config.api_key or "not-needed"
    return AsyncOpenAI(
        base_url=arm_config.base_url,
        api_key=api_key,
        max_retries=max_retries,
        timeout=DEFAULT_TIMEOUT_S,
    )


async def run_request(
    client: AsyncOpenAI,
    arm_config: ArmConfig,
    *,
    run_id: str,
    concurrency: int,
    prompt_bucket: str,
    case_id: str,
    system_prompt: str,
    question: str,
    max_tokens: int,
    hardware: HardwareInfo | None = None,
) -> RequestResult:
    start = time.monotonic()
    token_monotonics: list[float] = []
    generated_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    try:
        stream = await client.chat.completions.create(
            model=arm_config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            stop=STOP_SEQUENCES,
        )
        async for chunk in stream:
            now = time.monotonic()
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    token_monotonics.append(now)
                    generated_parts.append(delta)
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
    except Exception as exc:  # noqa: BLE001 — captured as a result field, not raised, so one
        # failed request doesn't abort an entire sweep; see sweep.py's error-rate handling.
        error = str(exc)
        log.warning(
            "client.request_error",
            arm=arm_config.name,
            model_variant=arm_config.model_variant,
            case_id=case_id,
            error=error,
        )

    if completion_tokens is None and token_monotonics:
        # Fallback for servers that don't send a final usage chunk (not all
        # OpenAI-compatible servers implement stream_options.include_usage).
        completion_tokens = len(token_monotonics)

    return RequestResult(
        run_id=run_id,
        arm=arm_config.name,
        model_variant=arm_config.model_variant,
        concurrency=concurrency,
        prompt_bucket=prompt_bucket,
        case_id=case_id,
        request_start_monotonic=start,
        first_token_monotonic=token_monotonics[0] if token_monotonics else None,
        last_token_monotonic=token_monotonics[-1] if token_monotonics else None,
        token_monotonics=token_monotonics,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        generated_text="".join(generated_parts) or None,
        error=error,
        hardware=hardware or get_hardware_info(),
    )
