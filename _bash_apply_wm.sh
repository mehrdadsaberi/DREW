#!/bin/bash

WM="trustmark"


CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method ${WM} \
--data-dir "DATA_DIR" \
--out-dir ./out/coco_${WM}_polar_full \
--n-bits 100 \
--n-clusters 1024 \