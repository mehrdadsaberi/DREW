#!/bin/bash

# MODEL="clip-openai"
MODEL="dinov2"
WM="trustmark"
DS="coco_"
SZ="full"

CUDA_VISIBLE_DEVICES=0 python create_embeddings.py \
--model "$MODEL" \
--data-dir ./out/${DS}${WM}_polar_${SZ}/images \
--out-path "./out/${DS}${WM}_polar_${SZ}/${MODEL}_embeddings.pkl"

