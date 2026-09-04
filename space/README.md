---
title: FORGE — NL to MongoDB
emoji: 🔨
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

# FORGE — Natural language to PyMongo

A LoRA fine-tuned Qwen2.5-Coder-1.5B, generating PyMongo queries from
natural-language questions against a fixed real schema (4 databases,
20 collections). Served here as a Q4_K_M GGUF quantization via
`llama-cpp-python` on CPU — a different quantization implementation than
the MLX 4-bit variant benchmarked in the full write-up (llama.cpp's
K-quant scheme vs. MLX's own INT4 scheme), used here because MLX only
runs on Apple Silicon and this Space runs on standard Linux infrastructure.

Full benchmark — throughput, accuracy, and cost across 3 serving stacks
(mlx_lm, Ollama, vLLM-Metal) and 3 quantization levels each, plus a real
comparison against a hosted API: see the
[project repo](https://github.com/) (link filled in once pushed).

The schema shown is intentionally fixed, not free-form editable: the
benchmark itself found that reduced or reshaped schema prompts — even
objectively simpler ones — put the model out of its training distribution
and it answers incorrectly (0% execution accuracy on a "short" prompt
bucket in the real sweep data). A free-form schema box would demo that
failure mode by accident rather than the model working as intended.
