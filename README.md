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
`pyproject.toml` pins torch to the cu126 index (recent builds dropped Pascal), so plain sync works:

    rm -rf .venv && uv sync
    uv run python scripts/probe_gpu.py          # does fla's Triton RWKV7 work on this card?
    uv run python -m rwkv_enhance.distill configs/demo_1070.yaml

On an H100, remove the `[tool.uv.sources]`/`[[tool.uv.index]]` blocks to get a current torch.
