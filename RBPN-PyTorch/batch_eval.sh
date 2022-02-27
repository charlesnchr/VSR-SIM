#!/bin/bash

thisdir=$(pwd)
model=weights/1x_cc-B450-AORUS-MRBPNPlanetEarth_SIMdata_6k_RBPN_epoch_100.pth
echo $thisdir
evalroot=./20210621_test
outdir=./20210621_test_out
mkdir -p $outdir

for folder in $evalroot/*; do
    subfolder=$(basename "$folder")
    echo $subfolder
    cd $folder
    ls *.tif > tri_filelist.txt 
    cd $thisdir
    python eval.py --gpus 1 --upscale_factor 1 --data_dir $folder --nFrames 9 --threads 4 --model_type RBPN --model $model --output $outdir/$evalroot
done

