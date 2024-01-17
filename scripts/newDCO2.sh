#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=newdataCO2
EXP_DIR=/netscratch/mudraje/spatial-explicit-ml/logs/

srun -K\
    --job-name="${JOB_NAME}" \
    --partition "RTX6000-SDS" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=0 \
    --cpus-per-task=2 \
    --mem=100G \
    --container-image=/netscratch/mudraje/spatial-explicit-ml/spatial_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time 12:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/spatial-explicit-ml/utils/newdataCO2.py \
