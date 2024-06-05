#!/bin/bash

MODEL="clip-openai"
WM="trustmark"
DS="coco_"
SZ="full"
K=1

AUGS=(
    "no_aug"
    "rotation,1.0"
    "rotation,0.5"
    "rotation,0.25"
    "crop,1.0"
    "crop,0.5"
    "crop,0.25"
    "flip"
    "stretch,1.0"
    "stretch,0.5"
    "stretch,0.25"
    "blur"
    "combo,1.0"
    "combo,0.5"
    "combo,0.25"
    "jitter"
    "diffpure,0.2"
    "diffpure,0.15"
    "diffpure,0.1"
)

# Iterate through the array
for AUG in "${AUGS[@]}"; do
    python evaluate_retrieval.py \
    --model "$MODEL" \
    --query-dir "./out/${DS}${WM}_polar_${SZ}/queries/queries_${AUG}" \
    --emb-path "./out/${DS}${WM}_polar_${SZ}/${MODEL}_embeddings.pkl" \
    --cluster-info-path ./out/${DS}${WM}_polar_${SZ}/cluster_info.pkl \
    --wm-th 1.0 \
    --log-path "./out/${DS}${WM}_polar_${SZ}/result_${MODEL}_${K}_scdecoder_1.0.log" \
    --wm-method ${WM} \
    --top-k ${K}
done


