#!/bin/bash
# Download COCO 2017 training dataset
#
# Usage:
#   ./scripts/download_coco_train.sh [COCO_ROOT]
#
# Arguments:
#   COCO_ROOT - Directory to download COCO data to (default: ./coco)
#
# This script will create the following structure:
#   COCO_ROOT/
#   ├── train2017/         # Training images
#   └── annotations/
#       └── instances_train2017.json

set -e

# Parse arguments
COCO_ROOT="${1:-./coco}"

echo "=============================================="
echo "COCO 2017 Training Dataset Downloader"
echo "=============================================="
echo "Target directory: ${COCO_ROOT}"
echo ""

# Create directories
mkdir -p "${COCO_ROOT}"
cd "${COCO_ROOT}"

# URLs for COCO 2017 training data
TRAIN_IMAGES_URL="http://images.cocodataset.org/zips/train2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Download training images
if [ -d "train2017" ]; then
    echo "train2017/ already exists, skipping download..."
else
    echo "Downloading training images (~18GB)..."
    curl -L -O "${TRAIN_IMAGES_URL}"
    echo "Extracting training images..."
    unzip -q train2017.zip
    rm train2017.zip
    echo "Done!"
fi

# Download annotations (contains instances_train2017.json)
if [ -f "annotations/instances_train2017.json" ]; then
    echo "annotations/instances_train2017.json already exists, skipping download..."
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
echo "  ├── train2017/                    # 118287 training images"
echo "  └── annotations/"
echo "      └── instances_train2017.json  # Training annotations"
echo ""
echo "To run training/evaluation:"
echo "  export COCO_ROOT=${COCO_ROOT}"
echo "  python -m src.run_experiment --precision fp32"
echo ""
