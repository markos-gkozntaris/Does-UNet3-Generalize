#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" 

LIVER_DIR="$DIR/LiverCT"
LIVER2_DIR="$DIR/LiverCT2"

mkdir -p "$LIVER_DIR/volumes"

find "$LIVER_DIR" -name volume-*.nii | grep volume_pt* | xargs -I {} mv {} "$LIVER_DIR/volumes" \
    && find "$LIVER_DIR" -name volume_pt* -type d -empty -delete

find "$LIVER2_DIR" -name volume-*.nii -type f -exec mv {} "$LIVER_DIR/volumes" \; \
    && find "$LIVER2_DIR" -type d -empty -delete
