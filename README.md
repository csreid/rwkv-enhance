# rwkv-enhance

Distillation (now) and RL (later) for RWKV7 "goose" models.

## Distillation on Nemotron
`nvidia/Nemotron-Post-Training-Dataset-v1` contains responses written by strong teacher models,
so the default mode is **sequence-level distillation**: CE on assistant tokens only.
Set `teacher:` in the config (a larger RWKV7 sharing the tokenizer) to add token-level KL.

    uv sync
    uv run python -m rwkv_enhance.distill configs/distill_nemotron.yaml

Needs a CUDA GPU (fla triton kernels). Not yet run/tested — dev box has no GPU.

## Planned
- `rl/`: GRPO-style RL on verifiable math/code prompts from Nemotron, starting from the distilled checkpoint.

## Old Pascal GPU (GTX 1070) demo
Recent torch wheels dropped Pascal; use the cu126 build:

    uv sync
    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126
    uv run python scripts/probe_gpu.py          # does fla's Triton RWKV7 work on this card?
    uv run python -m rwkv_enhance.distill configs/demo_1070.yaml
