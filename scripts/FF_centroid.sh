#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=SatclipFFcentroid
EXP_DIR=/netscratch/mudraje/spatial-explicit-ml/logs/

srun -K\
    --job-name="${JOB_NAME}" \
    --partition "V100-16GB-SDS" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=0 \
    --cpus-per-task=4\
    --mem=40G \
    --container-image=/netscratch/mudraje/spatial-explicit-ml/spatial_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time 12:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/spatial-explicit-ml/utils/Satclip_FF_centorid.py \