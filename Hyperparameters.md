# Hyperparameters used in models of the paper
-   **Video Swin** [@Liu2021a]\
    Option file: `VideoSwin/VideoSwin.yml`\
    Reference implementation: Video Swin Github\
    <https://github.com/SwinTransformer/Video-Swin-Transformer>\
    No. channels in: 1\
    No. channels out: 1\
    Patch size: (3,4,4)\
    Window size: (2, 7, 7)\
    MLP ratio: 4\
    No. of Swin transformer layers: 4\
    Depths of Swin transformer layers: (2, 2, 6, 2)\
    Embedding dimension: 96\
    Attention head number: (3, 6, 12, 24)

-   **RBPN** [@Haris2019]\
    Option file: `RCAN/RCAN.yml`\
    Reference implementation: RBPN Github.\
    <https://github.com/alterzero/RBPN-PyTorch>\
    No. channels in: 9\
    No. channels out: 1\
    No. of inital feature channels: 256\
    No. of deep feature channels: 64\
    No. of stages: 3\
    No. of residual blocks: 5

-   **RCAN** [@Zhang2018d]\
    Option file: `RCAN/RCAN.yml`\
    Reference implementation: BasicSR [@wang2020basicsr].\
    <https://github.com/xinntao/BasicSR>\
    No. channels in: 9\
    No. channels out: 1\
    No. of feature channels: 64\
    No. of residual groups: 10\
    No. of residual blocks: 20\
    Squeeze factor: 16\
    Residual scale: 1

-   **SwinIR** [@Liang2021]\
    Option file: `SwinIR/SwinIR.yml`\
    Reference implementation: SwinIR Github\
    <https://github.com/JingyunLiang/SwinIR>\
    No. channels in: 9\
    No. channels out: 1\
    Window size: 8\
    No. of Swin transformer layers: 6\
    Depths of Swin transformer layers: (6, 6, 6, 6, 6, 6)\
    Embedding size: 180:\
    Attention head number: (6, 6, 6, 6, 6, 6)

