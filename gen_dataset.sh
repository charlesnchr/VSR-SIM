#!/bin/bash

python scripts/datagen_pipeline.py --root SIMdata-out --sourceimages_path attenborough-frames --nrep 1 --datagen_workers 10 --imageSize 512 --nch_in 9 --nch_out 1 --ntrain 9000 --ntest 0 --scale 2 --NoiseLevel 8 --NoiseLevelRandFac 8 --Nangle 3 --Nshift 3 --phaseErrorFac 0.05 --alphaErrorFac 0.05 --seqSIM --ext imagefolder

