#!/bin/bash
# Download COCO 2017 validation dataset
#
# Usage:
#   ./scripts/download_coco_val.sh [COCO_ROOT]
#
# Arguments:
#   COCO_ROOT - Directory to download COCO data to (default: ./coco)
#
# This script will create the following structure:
#   COCO_ROOT/
#   ├── val2017/           # Validation images
#   └── annotations/
#       └── instances_val2017.json

set -e

# Parse arguments
COCO_ROOT="${1:-./coco}"

echo "=============================================="
echo "COCO 2017 Validation Dataset Downloader"
echo "=============================================="
echo "Target directory: ${COCO_ROOT}"
echo ""

# Create directories
mkdir -p "${COCO_ROOT}"
cd "${COCO_ROOT}"

# URLs for COCO 2017 validation data
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Download validation images
if [ -d "val2017" ]; then
    echo "val2017/ already exists, skipping download..."
else
    echo "Downloading validation images (~1GB)..."
    curl -L -O "${VAL_IMAGES_URL}"
    echo "Extracting validation images..."
    unzip -q val2017.zip
    rm val2017.zip
    echo "Done!"
fi

# Download annotations
if [ -f "annotations/instances_val2017.json" ]; then
    echo "annotations/instances_val2017.json already exists, skipping download..."
else
    echo "Downloading annotations (~241MB)..."
    curl -L -O "${ANNOTATIONS_URL}"
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "Done!"
fi

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "COCO data is now available at: ${COCO_ROOT}"
echo ""
echo "Structure:"
echo "  ${COCO_ROOT}/"
echo "  ├── val2017/                    # 5000 validation images"
echo "  └── annotations/"
echo "      └── instances_val2017.json  # Validation annotations"
echo ""
echo "To run evaluation:"
echo "  export COCO_ROOT=${COCO_ROOT}"
echo "  python -m src.run_experiment --precision fp32"
echo ""

