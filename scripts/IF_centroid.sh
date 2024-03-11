#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=IFCentroid
EXP_DIR=/netscratch/mudraje/spatial-explicit-ml/logs/
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
# --task-prolog="pwd/install.sh" \
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
    --time 03:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/spatial-explicit-ml/utils/InputF_centroid.py \
