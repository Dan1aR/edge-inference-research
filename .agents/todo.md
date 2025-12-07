You are an autonomous coding agent starting from an EMPTY Git repo.

Your goal: build a small, clean PyTorch project that evaluates the Hugging Face model `hustvl/yolos-tiny` on COCO 2017 validation under three numeric modes:

1. Full FP32
2. BF16 with default (hardware) higher-precision accumulators
3. BF16 with SOFTWARE-SIMULATED BF16 ACCUMULATORS (i.e., emulate true bf16 accumulation in matmul-style ops)

For each mode, run inference on COCO-val and compute standard COCO detection metrics (mAP etc.). Save metrics to disk so they can be compared later.

Focus ONLY on inference (no training).

-------------------------------------------------------------------------------
HIGH-LEVEL REQUIREMENTS
-------------------------------------------------------------------------------
- Use Python 3.10+.
- Use PyTorch (GPU if available), Hugging Face Transformers, and pycocotools.
- Use the Hugging Face model: `hustvl/yolos-tiny` and its corresponding `YolosImageProcessor`.
- Use the COCO 2017 validation split and compute COCO-style bbox metrics.
- Provide a single CLI entrypoint to run evaluation in any precision mode.
- Provide a clean, well-documented code structure; avoid notebooks.

-------------------------------------------------------------------------------
PROJECT STRUCTURE
-------------------------------------------------------------------------------
Create something like:

- README.md
- requirements.txt
- src/
  - __init__.py
  - config.py          # paths, constants, enums
  - data.py            # COCO dataset + preprocessing for YOLOS
  - eval_coco.py       # COCO evaluation logic
  - precision.py       # helpers to build models in different precision modes
  - bf16_accum.py      # core bf16-accumulator emulation utilities
  - run_experiment.py  # CLI to run all modes
- scripts/
  - download_coco_val.sh (optional helper to download COCO val)

You can adapt names if you keep things organized and readable.

-------------------------------------------------------------------------------
ENVIRONMENT & DEPENDENCIES
-------------------------------------------------------------------------------
Set up a standard Python project with:

- torch
- torchvision
- transformers
- pycocotools
- numpy
- tqdm
- click or argparse (your choice) for CLI

Create a `requirements.txt` and pin reasonably recent versions (e.g., PyTorch 2.x, Transformers >= 4.40). Assume CUDA-capable GPU with bf16 support if available; otherwise, fall back to CPU but keep code generic.

-------------------------------------------------------------------------------
COCO DATA HANDLING
-------------------------------------------------------------------------------
1. Assume the user will either:
   - Provide an existing COCO 2017 val directory, OR
   - Run a helper script you provide to download it.

2. Implement (or document) a script in `scripts/download_coco_val.sh` that:
   - Downloads COCO 2017 val images archive.
   - Downloads COCO 2017 annotations archive.
   - Unzips them into a user-specified root, e.g. `${COCO_ROOT}/val2017` and `${COCO_ROOT}/annotations`.

3. In `src/config.py`:
   - Define a small config object or constants:
     - `DEFAULT_COCO_ROOT` (can come from env var `COCO_ROOT` or CLI override).
     - Paths for images and annotation JSON under that root.

4. In `src/data.py`:
   - Implement a `CocoYolosDataset` that:
     - Wraps `torchvision.datasets.CocoDetection`.
     - On `__getitem__`, returns:
       - `pixel_values`: preprocessed tensor suitable for YOLOS.
       - `target`: labels dict in YOLOS/DETR format.
     - Use `YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")` for preprocessing.
       - Call the processor with `images=img` and `annotations={"image_id": image_id, "annotations": raw_target}` and `return_tensors="pt"`.
       - Extract `pixel_values` and `labels` similar to HF DETR/YOLOS examples.
   - Implement a `collate_fn` that:
     - Stacks / pads `pixel_values` appropriately.
     - Returns a batch dict ready for the model (`pixel_values`, and optionally any masks/labels needed).

5. Ensure COCO’s `image_id` from the dataset is preserved; we’ll need it later for metrics.

-------------------------------------------------------------------------------
COCO EVALUATION LOGIC
-------------------------------------------------------------------------------
In `src/eval_coco.py`:

1. Use `pycocotools` to compute detection mAP:
   - Get the underlying COCO object from the dataset (`dataset.coco` or similar).
   - For each image, YOLOS output should be converted to COCO-style predictions:
     - For each predicted box:
       - `image_id`: int
       - `category_id`: int (COCO category id)
       - `bbox`: [x_min, y_min, width, height] in **absolute pixel coordinates**
       - `score`: float confidence
   - Use `cocoGt = dataset.coco`.
   - Use `cocoDt = cocoGt.loadRes(predictions_list)` where `predictions_list` is a list of prediction dicts across all images.
   - Instantiate `COCOeval(cocoGt, cocoDt, iouType="bbox")`; then run `evaluate()`, `accumulate()`, and `summarize()`.

2. Category ID mapping:
   - `model.config.id2label` gives label indices -> class names.
   - The COCO dataset’s `cats` dict provides `category_id -> {"name": ...}`.
   - Build a mapping from YOLOS label index to COCO `category_id` by matching class names:
     - `name_to_cat_id = {cat["name"]: cid for cid, cat in cocoGt.cats.items()}`
     - `yolos_idx_to_coco_id = {idx: name_to_cat_id[name] for idx, name in id2label.items()}`
   - Use this mapping when generating prediction dicts.

3. Build a function like:

   - `def evaluate_coco(model, image_processor, dataloader, device, precision_mode, max_samples=None) -> dict:`
     - Loops over validation batches under `torch.inference_mode()` or `torch.no_grad()`.
     - For each batch:
       - Run model forward.
       - Use `image_processor.post_process_object_detection(outputs, target_sizes=...)` to get boxes in xyxy format for each image.
       - Convert each box to `[x, y, w, h]`.
       - Map YOLOS label index to `category_id`.
       - Append prediction dicts to a list.
     - Run COCOeval and capture summary metrics.
     - Return metrics as a Python dict (e.g., keys like `"AP"`, `"AP50"`, etc.).

4. Save the metrics for each run as JSON, e.g. `results_fp32.json`, `results_bf16_default.json`, `results_bf16_accum.json` in a `results/` directory.

-------------------------------------------------------------------------------
PRECISION MODES & MODEL CONSTRUCTION
-------------------------------------------------------------------------------
Create `src/precision.py` with helpers that build models under different modes.

Always start from the same FP32 pretrained checkpoint so that differences are only due to numeric handling:

- Load base model ONCE in FP32 on CPU, e.g.:
  - `base_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")`
  - `base_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")`

Then define functions:

1. `build_fp32_model(base_model, device) -> model`:
   - Deep-copy the base model or reload from pretrained.
   - Move to `device`.
   - Ensure all weights are `float32`.
   - Use standard PyTorch matmul/conv behavior.
   - No autocast / AMP.

2. `build_bf16_default_model(base_model, device) -> model`:
   - Deep-copy or reload.
   - Move to `device` and cast to `torch.bfloat16` (`model.to(torch.bfloat16)`).
   - Use the default kernels (these will multiply in bf16 but accumulate internally in higher precision).
   - Run under `torch.inference_mode()` without any extra emulation; this is the “hardware-type” behavior we’re comparing against.

3. `build_bf16_accum_model(base_model, device) -> model`:
   - Deep-copy or reload.
   - Move to `device`.
   - Convert weights to `torch.bfloat16`.
   - Replace all `nn.Linear` modules (and optionally `nn.Conv2d` used for patch embeddings) with custom modules that emulate **true bf16 accumulation** in their matmul-like operations.
   - Details of the emulation are in `bf16_accum.py` (see below).
   - Ensure the rest of the model runs in bf16 (inputs and weights) so the data type is consistent with the envisioned NPU.

-------------------------------------------------------------------------------
BF16 ACCUMULATOR EMULATION (CORE LOGIC)
-------------------------------------------------------------------------------
In `src/bf16_accum.py`, implement utilities that emulate BF16 accumulation semantics in software. The key idea: do NOT use `torch.matmul` or `.sum(dim=...)` for the reduction axis because they may internally accumulate in higher precision and only quantize at the end. Instead, explicitly build the sum as a loop over the reduction dimension, with BF16 rounding after each addition.

Design:

1. Helper conversion:

   ```python
   import torch

   def to_bf16(x: torch.Tensor) -> torch.Tensor:
       # Ensure values are representable in bfloat16
       return x.to(torch.bfloat16)
````

2. Emulated BF16 elementwise ops:

   ```python
   def bf16_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
       # elementwise multiply with bf16 rounding on inputs and outputs
       return (to_bf16(a) * to_bf16(b)).to(torch.bfloat16)

   def bf16_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
       # elementwise add with bf16 rounding on inputs and outputs
       return (to_bf16(a) + to_bf16(b)).to(torch.bfloat16)
   ```

3. Emulated BF16-accumulating matmul using an outer-product formulation:

   * Target shape: `x` with shape `[..., K]`, `w` with shape `[K, N]` produces `[..., N]`.
   * Implementation idea (vectorized over batch and output dim, loop over K):

   ```python
   def bf16_accum_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
       """
       Emulate matmul where both inputs and the accumulator are bf16.
       x: (..., K)
       w: (K, N)
       Returns: (..., N) in bf16.
       """
       assert x.dtype == torch.bfloat16
       assert w.dtype == torch.bfloat16

       orig_shape = x.shape[:-1]
       K = x.shape[-1]
       N = w.shape[1]

       x2d = x.reshape(-1, K)          # (B, K)
       w2d = w                         # (K, N)
       B = x2d.shape[0]

       out = torch.zeros(B, N, dtype=torch.bfloat16, device=x.device)

       # Loop over reduction dimension K; each step is outer product + bf16 add
       for k in range(K):
           x_k = x2d[:, k].unsqueeze(1)      # (B, 1)
           w_k = w2d[k, :].unsqueeze(0)      # (1, N)
           prod = bf16_mul(x_k, w_k)         # (B, N) in bf16
           out = bf16_add(out, prod)         # bf16 accumulation

       out = out.reshape(*orig_shape, N)
       return out
   ```

4. Custom linear layer:

   ```python
   import torch.nn as nn

   class BF16AccumLinear(nn.Module):
       def __init__(self, in_features, out_features, bias=True):
           super().__init__()
           self.in_features = in_features
           self.out_features = out_features
           self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
           if bias:
               self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
           else:
               self.bias = None

       @classmethod
       def from_linear(cls, linear: nn.Linear) -> "BF16AccumLinear":
           new = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
           new.weight.data.copy_(linear.weight.data.to(torch.bfloat16))
           if linear.bias is not None:
               new.bias.data.copy_(linear.bias.data.to(torch.bfloat16))
           return new

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           x = x.to(torch.bfloat16)
           # weight shape in Linear is (out_features, in_features); we need (in_features, out_features)
           y = bf16_accum_matmul(x, self.weight.transpose(0, 1))
           if self.bias is not None:
               y = bf16_add(y, self.bias)
           return y
   ```

5. Model patching function:

   ```python
   def replace_linear_with_bf16_accum(module: torch.nn.Module):
       for name, child in list(module.named_children()):
           if isinstance(child, torch.nn.Linear):
               setattr(module, name, BF16AccumLinear.from_linear(child))
           else:
               replace_linear_with_bf16_accum(child)
   ```

   * Call this on the YOLOS model instance for the `bf16_accum` mode **after** moving to device and converting weights to bf16.
   * For an initial version, it is acceptable to patch only `nn.Linear` layers; later, this can be extended to cover patch-embedding convolutions by emulating conv2d via `torch.nn.functional.unfold` + `bf16_accum_matmul`.

---

## CLI ENTRYPOINT

In `src/run_experiment.py`:

* Implement a CLI that supports at least:

  * `--coco-root`: path to COCO 2017 data root.
  * `--precision`: one of `fp32`, `bf16_default`, `bf16_accum`.
  * `--batch-size`: integer, default e.g. 4 or 8.
  * `--max-samples`: optional int to limit number of val images for quick tests (e.g. 100); if `None`, run full val set.
  * `--output`: path to JSON file to store metrics.

* Flow:

  1. Parse args.
  2. Set seeds for reproducibility (`torch.manual_seed`, `numpy`, etc.).
  3. Detect device (`cuda` if available else `cpu`).
  4. Load base model + processor from HF hub (once).
  5. Instantiate `CocoYolosDataset` and DataLoader with `collate_fn`.
  6. Depending on `--precision`, build the appropriate model via functions in `precision.py`:

     * `fp32` -> `build_fp32_model`
     * `bf16_default` -> `build_bf16_default_model`
     * `bf16_accum` -> `build_bf16_accum_model` + patch linear layers.
  7. Call `evaluate_coco(...)`.
  8. Print metrics to stdout and save to JSON file.

* Also implement a convenience command in the README to run all three modes sequentially, e.g.:

  * `python -m src.run_experiment --precision fp32 ...`
  * `python -m src.run_experiment --precision bf16_default ...`
  * `python -m src.run_experiment --precision bf16_accum ...`

---

## README & DOCUMENTATION

Write a concise README that explains:

* What the project does (compare FP32 vs BF16 vs BF16-accum on `hustvl/yolos-tiny`).
* How to install dependencies.
* How to download / point to COCO 2017 val data.
* Example commands to run each precision mode.
* Where the metrics JSON files will be written, and how to compare them.

---

## ACCEPTANCE CRITERIA

* `python -m src.run_experiment --precision fp32 --coco-root /path/to/coco --batch-size 8 --max-samples 50`:

  * Runs without error.
  * Prints COCO-style detection metrics.
  * Saves a JSON file with key metrics.

* `python -m src.run_experiment --precision bf16_default ...`:

  * Runs without error.
  * Metrics are close to FP32 (expect small difference only).

* `python -m src.run_experiment --precision bf16_accum ...`:

  * Runs without error (may be slower).
  * Uses the custom BF16-accumulating linear layers.
  * Produces metrics saved to separate JSON; values may be measurably different, which is the point of the research.

Focus on correctness and clarity first; micro-optimizations of the BF16-accum emulation can be done later.
