# Gated LoRA for Qwen3-VL-8B VQA

QLoRA baseline and **gated LoRA** SFT on Qwen3-VL-8B-Instruct for ChartQA, TextVQA, GQA, DocVQA, and Visual Genome.

Gated LoRA is standard LoRA plus a per-rank gate:

```
y = Wx + (α / r) B (g ⊙ (A x)),    g = sigmoid(gate_logits)
```

Adapters attach to **language** projections only. The vision tower stays frozen. After training, ranks with small `g` can be dropped for a smaller inference adapter.

This repo is **code only**. Weights, images, and checkpoints stay on Google Drive / Colab (see `.gitignore`).

## Layout

```
src/gated_lora.py       GatedLoRALinear, inject / save / load
src/data_utils.py       manifests, image-path remap, collator
src/metrics.py          dataset-specific VQA scores
src/profiler.py         per-step train latency + GPU memory
src/jobs.py             Drive job queue
scripts/train_qlora.py
scripts/train_gated_lora.py
scripts/eval_qlora.py   --adapter_type peft | gated
scripts/ctl.py          queue jobs from a laptop
workplace/colab_worker.ipynb
workplace/worker.py     A100 poll loop
workspace/config.json   default paths and hparams
```

## Training recipe (defaults)

| | |
|---|---|
| Base | Qwen3-VL-8B-Instruct, NF4 QLoRA |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Batch | 8 × grad accum 4 (effective 32) |
| LR / schedule | 2e-4 cosine, ~3% warmup |
| Epoch | 1 (~1048 optimizer steps on the filtered 33.5k train rows) |
| Save | every 200 steps, keep all gated checkpoints |

Mid-train eval is loss-only. VQA scores and inference latency are from `eval` after train.

## Colab A100 + local Cursor

The laptop has no NVIDIA GPU. Colab runs `worker.py`; Cursor writes jobs onto Drive.

1. Colab: open `workplace/colab_worker.ipynb`, runtime **A100**, Run all, leave the last cell running.
2. Local:

```powershell
python scripts/ctl.py train          # PEFT QLoRA → checkpoints/qlora_r8
python scripts/ctl.py gated-train    # gated LoRA → checkpoints/gated_qlora_r8
python scripts/ctl.py eval
python scripts/ctl.py gated-eval
python scripts/ctl.py watch
python scripts/ctl.py status
```

Gated checkpoints store `gated_adapter.pt` (A, B, gates), not PEFT `adapter_model.safetensors`. Resume uses the latest folder that has both `gated_adapter.pt` and `trainer_state.json`.

## Local CPU check (no 8B model)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/ctl.py cpu-test
```

## Expected efficiency (train)

On A100, both methods are dominated by the 8B VL forward/backward (~14–15 s/step, ~30 GB peak allocated). Gated LoRA is not meant to be faster in training; compare **eval scores** and **inference** (especially after pruning dead ranks).
