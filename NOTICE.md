# Third-party code

The VSR-SIM code is released under the GNU General Public License v3.0, see [LICENSE](LICENSE). The repository also contains code from the projects below. Those files keep the license of the project they come from.

| Path | Source | License |
| --- | --- | --- |
| `basicsr/` | [BasicSR](https://github.com/xinntao/BasicSR) 1.3.4.6, modified | Apache-2.0, [licenses/LICENSE-BasicSR](licenses/LICENSE-BasicSR) |
| `basicsr/archs/swinir_arch.py`, `swinir_rcab_arch.py`, `swinir3d_arch.py`, `inference/inference_*.py` | [SwinIR](https://github.com/JingyunLiang/SwinIR), modified | Apache-2.0, [licenses/LICENSE-SwinIR](licenses/LICENSE-SwinIR) |
| `basicsr/archs/vsr-sim_arch.py`, `swin3d_arch.py`, `swin3d_residual_flatten_arch.py` | [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), modified | Apache-2.0, [licenses/LICENSE-Video-Swin-Transformer](licenses/LICENSE-Video-Swin-Transformer) |
| Window attention in the files above | [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | MIT, [licenses/LICENSE-Swin-Transformer](licenses/LICENSE-Swin-Transformer) |
| `DropPath` and `trunc_normal_` in `basicsr/archs/` | [pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0, [licenses/LICENSE-pytorch-image-models](licenses/LICENSE-pytorch-image-models) |
| `basicsr/archs/uformer_arch.py` | [uformer-pytorch](https://github.com/lucidrains/uformer-pytorch), modified | MIT, [licenses/LICENSE-uformer-pytorch](licenses/LICENSE-uformer-pytorch) |
| `RBPN-PyTorch/` | [RBPN-PyTorch](https://github.com/alterzero/RBPN-PyTorch), modified | MIT, [RBPN-PyTorch/LICENSE](RBPN-PyTorch/LICENSE) |
| `RBPN-PyTorch/pyflow/` | [pyflow](https://github.com/pathak22/pyflow) | MIT for the Python wrapper, [RBPN-PyTorch/pyflow/LICENSE](RBPN-PyTorch/pyflow/LICENSE). The C++ sources in `src/` are Ce Liu's [optical flow code](https://people.csail.mit.edu/celiu/OpticalFlow/) and carry his terms. |

The SIM image formation model in `scripts/im_form_model/` is a Python port of the simulation in [OpenSIM](https://github.com/LanMai/OpenSIM), first written for [ML-SIM](https://github.com/charlesnchr/ML-SIM).
