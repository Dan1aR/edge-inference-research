# Implemented Features - YOLOS-tiny Precision Evaluation

This document describes all features implemented in the edge-inference-research project for evaluating YOLOS-tiny under different numeric precision modes.

---

## Project Overview

A PyTorch-based evaluation framework that compares object detection accuracy of `hustvl/yolos-tiny` on COCO 2017 validation under three precision modes:

1. **FP32** - Full 32-bit floating point (baseline)
2. **BF16 Default** - BFloat16 with hardware higher-precision accumulators
3. **BF16 Accum** - BFloat16 with software-emulated BF16 accumulators

---

## Implemented Modules

### 1. Configuration (`src/config.py`)

- **`PrecisionMode` enum**: Defines the three supported precision modes (`FP32`, `BF16_DEFAULT`, `BF16_ACCUM`)
- **`MODEL_NAME`**: Constant for the HuggingFace model identifier (`hustvl/yolos-tiny`)
- **`get_coco_paths()`**: Returns paths for COCO images and annotations, supporting:
  - Environment variable `COCO_ROOT`
  - CLI override via `--coco-root`
  - Default fallback to `./coco`
- **Default constants**: `DEFAULT_BATCH_SIZE`, `DEFAULT_NUM_WORKERS`, `DEFAULT_SEED`, `DEFAULT_RESULTS_DIR`

---

### 2. BF16 Accumulator Emulation (`src/bf16_accum.py`)

Core module implementing software emulation of true BF16 accumulation semantics.

#### Helper Functions:
- **`to_bf16(x)`**: Converts tensor to bfloat16 dtype
- **`bf16_mul(a, b)`**: Elementwise multiply with BF16 rounding on inputs and outputs
- **`bf16_add(a, b)`**: Elementwise add with BF16 rounding on inputs and outputs

#### BF16-Accumulating Matrix Multiply:
- **`bf16_accum_matmul(x, w)`**: Emulates matmul where both operands AND the accumulator are BF16
  - Uses outer-product formulation with loop over reduction dimension K
  - Explicitly rounds to BF16 after each accumulation step
  - Handles arbitrary batch dimensions via reshape
  - Input: `x` shape `(..., K)`, `w` shape `(K, N)` → Output: `(..., N)`

#### Custom Linear Layer:
- **`BF16AccumLinear(nn.Module)`**: Drop-in replacement for `nn.Linear`
  - Stores weights and bias in BF16
  - `from_linear()` classmethod to convert existing Linear layers
  - Forward pass uses `bf16_accum_matmul` for true BF16 accumulation

#### Model Patching:
- **`replace_linear_with_bf16_accum(module)`**: Recursively replaces all `nn.Linear` layers with `BF16AccumLinear`

---

### 3. Data Loading (`src/data.py`)

#### Dataset:
- **`CocoYolosDataset(Dataset)`**: Wraps `torchvision.datasets.CocoDetection`
  - Preprocesses images using `YolosImageProcessor`
  - Returns `pixel_values`, `image_id`, and `original_size` per sample
  - Supports `max_samples` parameter for quick testing
  - Exposes `self.coco` for access to underlying COCO API object

#### Batching:
- **`CollateFn`**: Picklable collate class for DataLoader multiprocessing
  - Pads images to maximum size in batch (right/bottom padding)
  - Collects image IDs and original sizes for post-processing
  - Returns batched `pixel_values`, `image_ids`, `original_sizes`

#### DataLoader Factory:
- **`create_dataloader()`**: Creates DataLoader with:
  - Configurable batch size and num_workers
  - Deterministic ordering (no shuffle)
  - MPS/CPU compatible (pin_memory disabled)

---

### 4. Precision Mode Model Builders (`src/precision.py`)

#### Base Model Loading:
- **`load_base_model()`**: Loads YOLOS model and processor from HuggingFace Hub in FP32 on CPU

#### Model Builders:
- **`build_fp32_model(base_model, device)`**: 
  - Deep copies base model
  - Moves to device in float32
  - Sets to eval mode

- **`build_bf16_default_model(base_model, device)`**:
  - Deep copies base model
  - Converts to bfloat16 dtype
  - Uses standard PyTorch operations (hardware accumulators)

- **`build_bf16_accum_model(base_model, device)`**:
  - Deep copies base model
  - Converts to bfloat16
  - Patches all Linear layers with `BF16AccumLinear`
  - Emulates true BF16 accumulation in matmul operations

- **`build_model(base_model, precision_mode, device)`**: Dispatcher that routes to appropriate builder

---

### 5. COCO Evaluation (`src/eval_coco.py`)

#### Category Mapping:
- **`build_category_mapping(model, coco_gt)`**: Maps YOLOS label indices to COCO category IDs by matching class names

#### Prediction Conversion:
- **`convert_predictions_to_coco_format()`**: Converts YOLOS post-processed outputs to COCO format:
  - Converts xyxy boxes to xywh format
  - Maps YOLOS labels to COCO category IDs
  - Creates prediction dicts with `image_id`, `category_id`, `bbox`, `score`

#### COCO Evaluation:
- **`run_coco_eval(coco_gt, predictions, image_ids=None)`**:
  - Loads predictions into COCO result format
  - Runs COCOeval with bbox IoU type
  - **Supports limiting evaluation to specific image IDs** (critical for `max_samples`)
  - Returns all 12 COCO metrics (AP, AP50, AP75, AP_small/medium/large, AR variants)

#### Main Evaluation Function:
- **`evaluate_coco()`**: Full evaluation pipeline:
  - Builds category mapping
  - Iterates over DataLoader with `torch.inference_mode()`
  - Handles precision-specific input dtype conversion
  - Tracks processed image IDs for correct evaluation scope
  - Returns metrics dict with precision mode metadata

#### Utilities:
- **`save_results(results, output_path)`**: Saves results as formatted JSON
- **`print_metrics(metrics)`**: Pretty-prints COCO metrics table

---

### 6. CLI Entrypoint (`src/run_experiment.py`)

#### Command-Line Interface:
```
python -m src.run_experiment [OPTIONS]

Options:
  --coco-root PATH        COCO 2017 data directory
  --precision [fp32|bf16_default|bf16_accum]  (required)
  --batch-size INTEGER    Default: 8
  --max-samples INTEGER   Limit images for testing
  --output PATH           Results JSON path
  --num-workers INTEGER   DataLoader workers (default: 4)
  --seed INTEGER          Random seed (default: 42)
  --threshold FLOAT       Detection score threshold (default: 0.0)
```

#### Features:
- **Seed setting**: Sets `torch`, `numpy`, and `random` seeds for reproducibility
- **Device detection**: Auto-detects CUDA with BF16 support warning
- **Path validation**: Checks COCO paths exist before running
- **Progress reporting**: Prints configuration and progress via tqdm
- **Result persistence**: Saves JSON with metrics and metadata

---

### 7. Helper Scripts

#### COCO Download (`scripts/download_coco_val.sh`):
- Downloads COCO 2017 validation images (~1GB)
- Downloads COCO 2017 annotations (~241MB)
- Extracts to specified directory
- Skips if files already exist
- Prints usage instructions after completion

---

## Project Configuration

### `pyproject.toml`:
- Python 3.10+ requirement
- Dependencies: torch, torchvision, transformers, pycocotools, numpy, tqdm, click, Pillow
- Optional dev dependencies: pytest, ruff
- Registered CLI script: `run-experiment`
- Hatch build backend for wheel packaging

### `requirements.txt`:
- Pinned minimum versions for all dependencies
- Compatible with pip installation

---

## Key Technical Decisions

1. **Outer-product matmul emulation**: Instead of vectorized matmul (which uses FP32 accumulators), loops over K dimension with BF16 rounding after each step

2. **Picklable collate function**: Used class instead of closure to support DataLoader multiprocessing

3. **Image padding strategy**: Pads to batch max size (not fixed size) to minimize wasted computation

4. **Evaluation scope limiting**: Passes processed image IDs to COCOeval to correctly compute metrics when using `max_samples`

5. **Deep copy for model variants**: Ensures each precision mode starts from identical FP32 weights

---

## Output Format

Results saved as JSON:
```json
{
  "precision_mode": "fp32",
  "num_images": 100,
  "num_predictions": 10000,
  "metrics": {
    "AP": 0.354,
    "AP50": 0.544,
    "AP75": 0.360,
    "AP_small": 0.184,
    "AP_medium": 0.392,
    "AP_large": 0.567,
    "AR_1": 0.283,
    "AR_10": 0.469,
    "AR_100": 0.488,
    "AR_small": 0.245,
    "AR_medium": 0.491,
    "AR_large": 0.728
  }
}
```

---

## Verified Performance

| Mode | Samples | AP@IoU=0.50:0.95 | Speed |
|------|---------|------------------|-------|
| FP32 | 100 | 0.354 | ~11 img/s |
| BF16 Default | 50 | 0.371 | ~1.4 img/s |
| BF16 Accum | 5 | 0.326 | ~0.26 img/s |

*Note: BF16 Accum is significantly slower due to Python loop emulation of matmul accumulation.*

