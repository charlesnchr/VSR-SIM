# Hyperparameters of the models in the paper

The option files are in `options/train/`.

## VSR-SIM

| | 3D windows | 2D windows (released weights) |
| --- | --- | --- |
| Option file | `VSR-SIM/VSR-SIM.yml` | `VSR-SIM/SwinIR_RCAB_kiiara.yml` |
| Implementation | `basicsr/archs/vsr-sim_arch.py` | `basicsr/archs/swinir_rcab_arch.py` |
| Channels in | 1, with 9 frames along the time axis | 9, one per frame |
| Channels out | 1 | 1 |
| Patch size | (3, 4, 4) | 1 |
| Window size | (2, 8, 8) | 8 |
| MLP ratio | 2 | 2 |
| Residual groups | 5 | 5 |
| Swin layers per group | 6 | 6 |
| Embedding dimension | 192 | 64 |
| Attention heads | 8 | 8 |
| Parameters | 13.9 million | 22.2 million |

## Models used for comparison

The values below are the ones reported in the supplementary material. The option files for RCAN, SwinIR and Video Swin differ from them in places. `RCAN.yml` has 5 groups of 12 blocks, `SwinIR.yml` has 7 groups with a window of 16 and an embedding dimension of 120, and `VideoSwin.yml` has 6 stages of depth 6. Check the file before reusing it.

**Video Swin**, from [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). Option file `VideoSwin/VideoSwin.yml`.

- Channels in and out: 1 and 1
- Patch size: (3, 4, 4)
- Window size: (2, 7, 7)
- MLP ratio: 4
- Swin transformer stages: 4, with depths (2, 2, 6, 2)
- Embedding dimension: 96
- Attention heads: (3, 6, 12, 24)

**RBPN**, from [RBPN-PyTorch](https://github.com/alterzero/RBPN-PyTorch). The code is in `RBPN-PyTorch/` and is trained with its own `main.py`.

- Channels in and out: 9 and 1
- Initial feature channels: 256
- Deep feature channels: 64
- Stages: 3
- Residual blocks: 5

**RCAN**, from [BasicSR](https://github.com/xinntao/BasicSR). Option file `RCAN/RCAN.yml`.

- Channels in and out: 9 and 1
- Feature channels: 64
- Residual groups: 10
- Residual blocks per group: 20
- Squeeze factor: 16
- Residual scale: 1

**SwinIR**, from [SwinIR](https://github.com/JingyunLiang/SwinIR). Option file `SwinIR/SwinIR.yml`.

- Channels in and out: 9 and 1
- Window size: 8
- Residual groups: 6, with 6 Swin layers each
- Embedding dimension: 180
- Attention heads: 6 in every group
