# YOLOS Tiny BF16 Accumulation Experiments

This repository fine-tunes [hustvl/yolos-tiny](https://huggingface.co/hustvl/yolos-tiny) with two regimes:

1. **Baseline AMP bf16/fp16/fp32** using standard PyTorch accumulators.
2. **Progressive bf16-accum forward + STE-like backward** where YOLOS attention and linear layers are patched to use custom bf16 accumulating implementations.

The project is self-contained: metrics are logged to stdout and local JSONL files; optional W&B reporting can be enabled but is not required.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch, transformers, datasets, accelerate, torchmetrics, pycocotools (plus optional albumentations/wandb).

## COCO Data

The training scripts expect either the COCO 2017 layout or CPPE-5 for quick debugging. For COCO, download and extract locally then pass `--coco_dir /path/to/coco`. The loader uses `torchvision.datasets.CocoDetection` directly and no longer relies on the deprecated Hugging Face loading script `yonigozlan/coco_detection_dataset_script`.

- Expected layout: `${coco_dir}/train2017`, `${coco_dir}/val2017`, `${coco_dir}/annotations/instances_train2017.json`, `${coco_dir}/annotations/instances_val2017.json`.
- Debug option: `--dataset cppe5` or `--max_train_samples`/`--max_eval_samples` to iterate quickly.

## Baseline Training (bf16 AMP)

```bash
python train_yolos_baseline_bf16.py \
  --output_dir outputs/baseline \
  --dataset coco2017 \
  --coco_dir /path/to/coco \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --precision bf16
```

The script logs to `output_dir/logs/train.jsonl` and `eval.jsonl` and writes a `metrics_summary.json` plus checkpoints under `output_dir/`.

## Progressive bf16-accum + STE

```bash
python train_yolos_progressive_bf16accum_ste.py \
  --output_dir outputs/progressive \
  --dataset coco2017 \
  --coco_dir /path/to/coco \
  --schedule "attn@0.2,mlp@0.4,qkv@0.6,head@0.8" \
  --ramp_mode blocks --blocks_per_stage 2
```

Progressively enables the bf16-accum attention patch and Triton-based linear replacements. The enabled groups and block ramps are logged during eval.

## Logging and W&B

- Local JSONL logs are always written to `output_dir/logs/`.
- Optional: `--report_to wandb --wandb_project <project> --wandb_run_name <name>` enables W&B if installed; otherwise the run continues with local logging only.

## Tests

Run the lightweight smoke tests (CPU-friendly):

```bash
pytest tests/test_smoke_train_step.py tests/test_patching_is_safe.py
```

These tests validate a single forward/backward step and ensure the linear patching keeps parameter identities intact.
