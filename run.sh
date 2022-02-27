
PYTHONPATH="./:${PYTHONPATH}" python inference/inference_options.py --task simrec --model_path experiments/SwinIR_RCAB/net_g_latest.pth --scale 2 --input testdir/inputs/Extra_TestImages --output testdir/outputs/VSR-SIM -opt options/train/VSR-SIM/SwinIR_RCAB_kiiara.yml
