#!/bin/bash

submit_job_crop_ensemble() {
    local DC=$1
    local ST=$2
    local SHIFT=$3
    local ARCH=$4
    local EN=$5
    local COM=$6
    local MODEL=$7
    local METHOD=$8

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=all,AGG=zero,CS=-1,ARCH=$ARCH,EN=$EN,COM=$COM,MODEL=$MODEL $EVAL_SCRIPT
}

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
COM=max
DC=512
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 2