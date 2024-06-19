#!/bin/bash

DS=$1
MODEL="dinov2"
WM="trustmark"
SZ="full"

python create_embeddings.py \
--model "$MODEL" \
--data-dir ./out/${DS}_${WM}_polar_${SZ}/images \
--out-path "./out/${DS}_${WM}_polar_${SZ}/${MODEL}_embeddings.pkl"

