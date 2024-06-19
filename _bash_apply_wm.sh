#!/bin/bash

DS=$1
DATA_DIR=$2
WM="trustmark"
SZ="full"


python apply_watermark.py \
--wm-method ${WM} \
--data-dir ${DATA_DIR} \
--out-dir ./out/${DS}_${WM}_polar_${SZ} \
--n-bits 100 \
--n-clusters 1024 \