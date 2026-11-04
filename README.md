# rwkv-enhance

Distillation (now) and RL (later) for RWKV7 "goose" models.

## Distillation on Nemotron
`nvidia/Nemotron-Post-Training-Dataset-v1` contains responses written by strong teacher models,
so the default mode is **sequence-level distillation**: CE on assistant tokens only.
Set `teacher:` in the config (a larger RWKV7 sharing the tokenizer) to add token-level KL.

    uv sync
    uv run python -m rwkv_enhance.distill configs/distill_nemotron.yaml

Needs a CUDA GPU (fla triton kernels). Not yet run/tested — dev box has no GPU.

## Agentic RL (bash tool use, long horizon)
RWKV's recurrent state is fixed-size, so an episode of any number of turns costs constant memory:
the rollout feeds each command's output into the carried state instead of re-encoding history, and
training uses truncated BPTT (segments of `seg_len`, state carried but detached).

    uv run python scripts/sandbox_check.py                  # bubblewrap isolation self-test (do this first)
    uv run python scripts/make_demos.py demos.jsonl --n 3000   # scripted-expert demos, run in the sandbox
    uv run python -m rwkv_enhance.distill configs/sft_demos.yaml   # warm start: learn the tool format
    uv run python -m rwkv_enhance.rl_train configs/rl_agent.yaml   # GRPO + automatic curriculum
    uv run python scripts/plot_log.py ...                   # (distill logs only for now)

- Tasks (`tasks.py`): procedural, verifiable, JSON-serialisable; `level` controls horizon
  (file_exists -> read_file -> create_file/count_matches -> hop_chain/collect_codes -> needle_log).
- Turn format: a ```bash block runs in the sandbox; `Final answer: ...` ends the episode.
- Reward: 1 if the answer / resulting files are correct, else 0. GRPO advantages are group-relative;
  groups where every rollout got the same reward are skipped (zero gradient).
- Curriculum: level goes up when the success rate at the current level reaches `level_up_at`.

Tests: `PYTHONPATH=src uv run pytest tests`.
