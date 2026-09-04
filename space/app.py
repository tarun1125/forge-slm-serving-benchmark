"""FORGE demo — Qwen2.5-Coder-1.5B, LoRA fine-tuned for natural-language ->
PyMongo query generation, served here via the Q4_K_M GGUF quantization.

Runs on llama-cpp-python, not mlx_lm: MLX only runs on Apple Silicon, and
this Space runs on standard Linux CPU infrastructure — the GGUF variant
(produced in Phase 1 for the Ollama benchmark arm, see
docs/parity-check-design.md) is the portable one. This is a genuinely
different quantization implementation than the MLX 4-bit variant benchmarked
in Phase 2 (llama.cpp's K-quant scheme vs. MLX's own INT4 scheme) — labeled
here honestly, not silently conflated with "the 4-bit results" from the
sweep.

The default schema (system_prompt.txt) is the exact, unmodified system
prompt used during training for this batch of 4 databases — not a
simplified or reconstructed version. Phase 2's own sweep found that
reduced/reshaped schema prompts, even objectively simpler ones, produce a
model that's out-of-distribution and answers incorrectly (0% execution
accuracy on the "short" prompt bucket, see docs/cost-model.md) — so this
demo deliberately does NOT let a visitor edit the schema free-form, to
avoid handing out a demo that looks broken for a documented, understood
reason unrelated to the model actually being broken.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO_ID = "REPLACE_WITH_HF_USERNAME/forge-qwen2.5-coder-1.5b-mongodb-gguf"
MODEL_FILENAME = "model-Q4_K_M.gguf"
# Set to a local .gguf path to bypass the HF Hub download entirely — used to
# test this app before the model exists on the Hub, and useful for local
# debugging afterward too.
LOCAL_MODEL_PATH = os.environ.get("FORGE_LOCAL_MODEL_PATH")
SYSTEM_PROMPT = Path(__file__).parent.joinpath("system_prompt.txt").read_text(encoding="utf-8")

STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>"]  # see src/forge/phase2/client.py's own note on this

EXAMPLE_QUESTIONS = [
    "display those departments where more than ten employees work who got a commission percentage.",
    "return the smallest salary for every departments.",
    "What is the average salary of employees in each department?",
]
# Note on what got cut: "Find the ids of the departments where any manager is
# managing 4 or more employees" was tried here first. Verified (5 repeated
# calls, deterministic once the KV-cache reset() fix above was in place) to
# reliably produce a malformed, syntactically-broken query — a real accuracy
# limit of the 1.5B fine-tune on this $size/$expr-nesting question shape, not
# a demo bug. That's honest and consistent with Phase 2's measured ~25-50%
# medium-bucket accuracy, and belongs in the write-up's failure gallery
# (Phase 5) — not as a default example a first-time visitor clicks and
# concludes the demo itself is broken.

_llm: Llama | None = None


def get_model() -> Llama:
    global _llm
    if _llm is None:
        model_path = LOCAL_MODEL_PATH or hf_hub_download(
            repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME
        )
        _llm = Llama(
            model_path=model_path,
            n_ctx=8192,  # see fine_tuning/ollama_register.py's own note on Ollama's default 4096 being too small
            n_threads=4,
            verbose=False,
        )
    return _llm


def generate_query(question: str) -> str:
    if not question.strip():
        return ""
    llm = get_model()
    # llama-cpp-python's Llama object keeps its KV cache / token history across
    # calls on the same instance. Since this Space reuses one global _llm for
    # every visitor's request, skipping reset() here means one question's
    # context can silently bleed into the next generation — verified this
    # empirically: the same question, same temperature=0, produced different
    # output depending on what was asked right before it in the same process,
    # until reset() was added. Always start from a clean context.
    llm.reset()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=0,
        stop=STOP_SEQUENCES,
    )
    return response["choices"][0]["message"]["content"].strip()


with gr.Blocks(title="FORGE — NL to MongoDB") as demo:
    gr.Markdown(
        "# FORGE — Natural language to PyMongo\n"
        "A LoRA fine-tuned Qwen2.5-Coder-1.5B, served here as a Q4_K_M GGUF "
        "on CPU. Schema is fixed to a real training-format prompt covering "
        "4 databases (college_3, flight_4, hr_1, inn_1) — try a question "
        "about employees, departments, flights, courses, or hotel rooms. "
        "[Full benchmark write-up](https://github.com/) — throughput, "
        "accuracy, and cost across 3 serving stacks and 3 quantization levels."
    )
    with gr.Row():
        question = gr.Textbox(
            label="Question", placeholder="e.g. return the smallest salary for every department."
        )
    generate_btn = gr.Button("Generate PyMongo query", variant="primary")
    output = gr.Code(label="Generated query", language="python")

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)
    generate_btn.click(fn=generate_query, inputs=question, outputs=output)
    question.submit(fn=generate_query, inputs=question, outputs=output)


if __name__ == "__main__":
    demo.launch()
