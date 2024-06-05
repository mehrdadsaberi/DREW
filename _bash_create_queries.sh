#!/bin/bash

WM="trustmark"
DS="coco_"
SZ="full"

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
    python create_queries.py \
    --aug-type $AUG \
    --data-dir "./out/${DS}${WM}_polar_${SZ}/images" \
    --out-dir "./out/${DS}${WM}_polar_${SZ}/queries/queries_${AUG}" \
    --n-query 1000 
done




