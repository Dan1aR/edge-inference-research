# YOLOS-tiny Precision Evaluation

Evaluate the [hustvl/yolos-tiny](https://huggingface.co/hustvl/yolos-tiny) object detection model on COCO 2017 validation under three numeric precision modes:

1. **FP32** — Full 32-bit floating point precision
2. **BF16 Default** — BFloat16 with hardware (higher-precision) accumulators
3. **BF16 Accum** — BFloat16 with software-simulated BF16 accumulators

This project investigates how different accumulator precision affects detection accuracy, which is relevant for understanding edge NPU behavior where true BF16 accumulation may be used.

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py          # Paths, constants, enums
│   ├── data.py            # COCO dataset + preprocessing
│   ├── eval_coco.py       # COCO evaluation logic
│   ├── precision.py       # Model builders for each precision mode
│   ├── bf16_accum.py      # BF16 accumulator emulation utilities
│   └── run_experiment.py  # CLI entrypoint
├── scripts/
│   └── download_coco_val.sh  # Helper to download COCO data
└── results/               # Output directory for metrics JSON files
```

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.2+ (with CUDA support recommended)
- Transformers 4.40+
- pycocotools
- numpy, tqdm, click, Pillow

## COCO 2017 Data Setup

### Option 1: Use the download script

```bash
# Download to ./coco directory
./scripts/download_coco_val.sh

# Or specify a custom directory
./scripts/download_coco_val.sh /path/to/coco
```

### Option 2: Manual download

1. Download [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip) (~1GB)
2. Download [COCO 2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (~241MB)
3. Extract to create this structure:

```
/path/to/coco/
├── val2017/                    # 5000 validation images
└── annotations/
    └── instances_val2017.json  # Validation annotations
```

### Option 3: Set environment variable

```bash
export COCO_ROOT=/path/to/your/coco
```

## Usage

### Running evaluation

```bash
# FP32 evaluation (baseline)
python -m src.run_experiment --precision fp32 --coco-root /path/to/coco

# BF16 with hardware accumulators
python -m src.run_experiment --precision bf16_default --coco-root /path/to/coco

# BF16 with emulated BF16 accumulators
python -m src.run_experiment --precision bf16_accum --coco-root /path/to/coco
```

### Quick test (limited samples)

```bash
# Run on only 50 images for quick testing
python -m src.run_experiment --precision fp32 --coco-root /path/to/coco --max-samples 50
```

### All CLI options

```bash
python -m src.run_experiment --help

Options:
  --coco-root PATH        Path to COCO 2017 data root
  --precision [fp32|bf16_default|bf16_accum]
                          Precision mode for inference (required)
  --batch-size INTEGER    Batch size (default: 8)
  --max-samples INTEGER   Limit number of images (for testing)
  --output PATH           Custom output path for results JSON
  --num-workers INTEGER   DataLoader workers (default: 4)
  --seed INTEGER          Random seed (default: 42)
  --threshold FLOAT       Score threshold for predictions (default: 0.0)
```

### Running all three modes

```bash
# Set COCO root (or use --coco-root each time)
export COCO_ROOT=/path/to/coco

# Run all precision modes
python -m src.run_experiment --precision fp32 --output results/results_fp32.json
python -m src.run_experiment --precision bf16_default --output results/results_bf16_default.json
python -m src.run_experiment --precision bf16_accum --output results/results_bf16_accum.json
```

## Output

Results are saved as JSON files in the `results/` directory:

```json
{
  "precision_mode": "fp32",
  "num_images": 5000,
  "num_predictions": 123456,
  "metrics": {
    "AP": 0.287,
    "AP50": 0.476,
    "AP75": 0.293,
    "AP_small": 0.112,
    "AP_medium": 0.312,
    "AP_large": 0.455,
    "AR_1": 0.267,
    "AR_10": 0.405,
    "AR_100": 0.425,
    "AR_small": 0.189,
    "AR_medium": 0.466,
    "AR_large": 0.612
  }
}
```

### Comparing results

```bash
# View all results
cat results/results_fp32.json
cat results/results_bf16_default.json
cat results/results_bf16_accum.json

# Quick comparison with jq
jq '.metrics.AP' results/*.json
```

## Understanding the Precision Modes

### FP32 (Baseline)
Standard 32-bit floating point. This is the reference for accuracy.

### BF16 Default
Weights and activations are in BFloat16, but PyTorch/CUDA typically uses FP32 accumulators internally for matrix multiply operations. This represents typical mixed-precision inference.

### BF16 Accum (Research Focus)
Emulates hardware that uses BF16 for **both** operands and accumulators. This is achieved by:
- Replacing `nn.Linear` layers with custom `BF16AccumLinear`
- Performing matmul as a loop over the reduction dimension
- Explicitly rounding to BF16 after each accumulation step

This mode is slower but accurately simulates NPUs with true BF16 arithmetic.

## Technical Details

### BF16 Accumulator Emulation

The emulation in `src/bf16_accum.py` works by:

1. Converting inputs to BF16
2. Computing outer products for each position in the reduction dimension
3. Accumulating results with explicit BF16 rounding after each addition

```python
# Pseudo-code for bf16_accum_matmul
out = zeros(B, N, dtype=bf16)
for k in range(K):
    prod = bf16_mul(x[:, k:k+1], w[k, :])  # Outer product
    out = bf16_add(out, prod)               # BF16 accumulation
```

This ensures the accumulator never exceeds BF16 precision, unlike standard matmul which may use FP32 internally.

## Expected Results

- **FP32 vs BF16 Default**: Should be nearly identical (< 0.1% difference in mAP)
- **BF16 Accum**: May show measurable degradation due to reduced accumulator precision

The exact impact depends on:
- Model architecture (depth, width)
- Value distributions in weights/activations
- Number of accumulations per output element

## License

See [LICENSE](LICENSE) file.

## References

- [YOLOS Paper](https://arxiv.org/abs/2106.00666)
- [HuggingFace YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos)
- [COCO Dataset](https://cocodataset.org/)
- [BFloat16 Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
