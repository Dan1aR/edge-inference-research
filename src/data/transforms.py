from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

try:
    import albumentations as A

    _HAVE_ALBUMENTATIONS = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_ALBUMENTATIONS = False
    A = None

from PIL import Image
import numpy as np

ImageLike = Image.Image | np.ndarray


def build_transform(pipeline: str | None = None) -> Callable[[Sequence[ImageLike]], List[ImageLike]]:
    """Build an augmentation pipeline.

    Args:
        pipeline: Optional pipeline identifier. When ``None`` or ``"none"`` a
            no-op transform is returned.

    Returns:
        Callable that accepts a sequence of images and returns a list of transformed
        images.
    """

    if pipeline is None or pipeline.lower() == "none":
        def _noop(images: Sequence[ImageLike]) -> List[ImageLike]:
            return [img for img in images]

        return _noop

    if pipeline.lower() == "basic" and _HAVE_ALBUMENTATIONS:
        aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ]
        )

        def _apply(images: Sequence[ImageLike]) -> List[ImageLike]:
            outputs: List[ImageLike] = []
            for img in images:
                if isinstance(img, Image.Image):
                    img_np = np.array(img)
                else:
                    img_np = img
                transformed = aug(image=img_np)["image"]
                outputs.append(Image.fromarray(transformed))
            return outputs

        return _apply

    # Fallback to no-op if unknown pipeline
    def _default(images: Sequence[ImageLike]) -> List[ImageLike]:
        return [img for img in images]

    return _default
