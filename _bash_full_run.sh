#!/bin/bash

DS=$1
DATA_DIR=$2
MODEL="dinov2"
WM="trustmark"
SZ="full"
K=1

# apply watermark
python apply_watermark.py \
--wm-method ${WM} \
--data-dir ${DATA_DIR} \
--out-dir ./out/${DS}_${WM}_polar_${SZ} \
--n-bits 100 \
--n-clusters 1024 \


# create embeddings
python create_embeddings.py \
--model "$MODEL" \
--data-dir ./out/${DS}_${WM}_polar_${SZ}/images \
--out-path "./out/${DS}_${WM}_polar_${SZ}/${MODEL}_embeddings.pkl"


# define augs
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
    # "diffpure,0.2"
    # "diffpure,0.15"
    # "diffpure,0.1"
)

# create queries
for AUG in "${AUGS[@]}"; do
    python create_queries.py \
    --aug-type $AUG \
    --data-dir "./out/${DS}_${WM}_polar_${SZ}/images" \
    --out-dir "./out/${DS}_${WM}_polar_${SZ}/queries/queries_${AUG}" \
    --n-query 1000 
done

# # evaluate
for AUG in "${AUGS[@]}"; do
    python evaluate_retrieval.py \
    --model "$MODEL" \
    --query-dir "./out/${DS}_${WM}_polar_${SZ}/queries/queries_${AUG}" \
    --emb-path "./out/${DS}_${WM}_polar_${SZ}/${MODEL}_embeddings.pkl" \
    --cluster-info-path ./out/${DS}_${WM}_polar_${SZ}/cluster_info.pkl \
    --wm-th 1.0 \
    --log-path "./out/${DS}_${WM}_polar_${SZ}/result_${MODEL}_${K}_scdecoder_1.0.log" \
    --wm-method ${WM} \
    --top-k ${K}
done


