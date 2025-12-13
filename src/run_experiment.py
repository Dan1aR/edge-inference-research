"""
CLI Entrypoint for YOLOS Precision Evaluation

Run inference on COCO 2017 validation under different precision modes.
"""
import click
import torch
import numpy as np
import random
from pathlib import Path

from .config import (
    PrecisionMode,
    get_coco_paths,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_RESULTS_DIR,
)
from .precision import load_base_model, build_model
from .datasets import create_dataloader
from .eval_coco import evaluate_coco, save_results, print_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Check for BF16 support
        if torch.cuda.is_bf16_supported():
            print("BF16 is supported on this device")
        else:
            print("Warning: BF16 may not be fully supported on this device")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA not available)")
    return device


@click.command()
@click.option(
    "--coco-root",
    type=click.Path(exists=True),
    default=None,
    help="Path to COCO 2017 data root. Can also be set via COCO_ROOT env var.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "bf16_default", "bf16_accum"]),
    required=True,
    help="Precision mode for inference.",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size for inference. Default: {DEFAULT_BATCH_SIZE}",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Limit number of validation images (for quick tests). Default: use all.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Path to save results JSON. Default: results/results_{precision}.json",
)
@click.option(
    "--num-workers",
    type=int,
    default=DEFAULT_NUM_WORKERS,
    help=f"Number of DataLoader workers. Default: {DEFAULT_NUM_WORKERS}",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help=f"Random seed for reproducibility. Default: {DEFAULT_SEED}",
)
@click.option(
    "--threshold",
    type=float,
    default=0.0,
    help="Score threshold for predictions. Default: 0.0",
)
@click.option(
    "--bf16-accum-linears/--no-bf16-accum-linears",
    default=True,
    show_default=True,
    help="When using bf16_accum, replace Linear layers with BF16AccumLinear.",
)
@click.option(
    "--bf16-accum-patch-embed/--no-bf16-accum-patch-embed",
    default=True,
    show_default=True,
    help="When using bf16_accum, replace patch embedding conv with BF16AccumConv2d.",
)
@click.option(
    "--bf16-accum-attention/--no-bf16-accum-attention",
    default=True,
    show_default=True,
    help="When using bf16_accum, patch attention matmuls to use BF16 accumulators.",
)
def main(
    coco_root: str | None,
    precision: str,
    batch_size: int,
    max_samples: int | None,
    output: str | None,
    num_workers: int,
    seed: int,
    threshold: float,
    bf16_accum_linears: bool,
    bf16_accum_patch_embed: bool,
    bf16_accum_attention: bool,
) -> None:
    """
    Evaluate YOLOS-tiny on COCO 2017 validation under different precision modes.

    Examples:

    \b
    # Run FP32 evaluation
    python -m src.run_experiment --precision fp32 --coco-root /path/to/coco

    \b
    # Run BF16 (default accumulators) evaluation
    python -m src.run_experiment --precision bf16_default --coco-root /path/to/coco

    \b
    # Run BF16 (emulated BF16 accumulators) evaluation
    python -m src.run_experiment --precision bf16_accum --coco-root /path/to/coco

    \b
    # Quick test with limited samples
    python -m src.run_experiment --precision fp32 --coco-root /path/to/coco --max-samples 50
    """
    print("=" * 60)
    print("YOLOS-tiny Precision Evaluation")
    print("=" * 60)

    # Set seeds
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Get device
    device = get_device()

    # Parse precision mode
    precision_mode = PrecisionMode(precision)
    print(f"Precision mode: {precision_mode.value}")

    # Get COCO paths
    coco_paths = get_coco_paths(coco_root)
    print(f"COCO images: {coco_paths['images']}")
    print(f"COCO annotations: {coco_paths['annotations']}")

    # Validate paths exist
    if not coco_paths["images"].exists():
        raise click.ClickException(
            f"COCO images directory not found: {coco_paths['images']}\n"
            "Please download COCO 2017 val data or specify --coco-root"
        )
    if not coco_paths["annotations"].exists():
        raise click.ClickException(
            f"COCO annotations file not found: {coco_paths['annotations']}\n"
            "Please download COCO 2017 annotations or specify --coco-root"
        )

    # Determine output path
    if output is None:
        output = DEFAULT_RESULTS_DIR / f"results_{precision}.json"
    output = Path(output)

    print(f"\nBatch size: {batch_size}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Output: {output}")

    # Load base model
    print("\nLoading base model from HuggingFace Hub...")
    base_model, processor = load_base_model()
    print("Base model loaded successfully")

    # Create dataloader
    print("\nCreating DataLoader...")
    dataloader, dataset = create_dataloader(
        images_dir=coco_paths["images"],
        annotations_file=coco_paths["annotations"],
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Build model for specified precision
    print(f"\nBuilding {precision_mode.value} model...")
    if precision_mode == PrecisionMode.BF16_ACCUM:
        print(
            "BF16-accum component toggles -> "
            f"linears: {bf16_accum_linears}, "
            f"patch_embed: {bf16_accum_patch_embed}, "
            f"attention: {bf16_accum_attention}"
        )
    model = build_model(
        base_model,
        precision_mode,
        device,
        use_bf16_accum_linears=bf16_accum_linears,
        use_bf16_accum_patch_embed=bf16_accum_patch_embed,
        use_bf16_accum_attention=bf16_accum_attention,
    )
    print(f"Model ready on {device}")

    # Run evaluation
    results = evaluate_coco(
        model=model,
        processor=processor,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        precision_mode=precision_mode,
        threshold=threshold,
    )

    # Print metrics
    print_metrics(results["metrics"])

    # Save results
    save_results(results, output)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

