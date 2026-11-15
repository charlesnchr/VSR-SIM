"""Reconstruct raw SIM stacks with the pre-trained VSR-SIM model.

    python reconstruct.py stack.tif

Each input is a TIFF stack whose first 9 frames are the raw SIM acquisition
(3 orientations x 3 phase shifts, in any order). For every stack the script
writes the VSR-SIM reconstruction, at twice the pixel count of the raw frames
in each direction, and the wide-field image (the mean of the raw frames) to
the output folder.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io

from basicsr.archs.swinir_rcab_arch import SwinIR_RCAB

WEIGHTS_NAME = "vsr-sim_swinir-rcab_20220202.pth"
WINDOW = 8  # side of the attention windows, the input is padded to a multiple of it


def load_model(weights, device):
    """Build the network of the released checkpoint and load its weights."""
    net = SwinIR_RCAB(
        upscale=2, in_chans=9, out_chans=1, img_size=128, window_size=WINDOW, img_range=1.0,
        depths=[6] * 5, embed_dim=64, num_heads=[8] * 5, mlp_ratio=2,
        upsampler=None, resi_connection="1conv", pixelshuffleFactor=1,
    )
    checkpoint = torch.load(weights, map_location="cpu")
    net.load_state_dict(checkpoint["params_ema" if "params_ema" in checkpoint else "params"])
    return net.to(device).eval()


def normalise(frames):
    """Scale raw frames to [0, 1], the range of the simulated training data."""
    frames = np.asarray(frames)
    integer = np.issubdtype(frames.dtype, np.integer)
    if integer:
        frames = frames / np.iinfo(frames.dtype).max
    frames = frames.astype("float32")
    lo, hi = frames.min(), frames.max()
    if not integer or hi - lo < 0.9:  # dim data, or a camera that does not fill its bit depth
        frames = (frames - lo) / (hi - lo + 1e-12)
    return frames


def widefield(frames):
    """Mean of the raw frames, stretched to [0, 1] for display."""
    wf = frames.mean(0)
    return (wf - wf.min()) / (wf.max() - wf.min() + 1e-12)


def reconstruct(net, frames, device):
    """Reconstruct 9 normalised frames (9, H, W) into one image (2H, 2W) in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).unsqueeze(0).to(device)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, -w % WINDOW, 0, -h % WINDOW), mode="reflect")
    with torch.no_grad():
        sr = net(x)
    return sr[0, 0, : 2 * h, : 2 * w].clamp(0, 1).cpu().numpy()


def save_png(path, image):
    io.imsave(path, (image * 255).round().astype("uint8"), check_contrast=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stacks", nargs="+", help="TIFF stacks with at least 9 frames")
    p.add_argument("--weights", default=WEIGHTS_NAME, help="path to the model file")
    p.add_argument("--out", default="reconstructions", help="output folder")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args.weights, device)
    os.makedirs(args.out, exist_ok=True)

    for path in args.stacks:
        stack = io.imread(path)
        if stack.ndim != 3 or stack.shape[0] < 9:
            print("Skipping %s: expected a stack of at least 9 frames, got shape %s" % (path, stack.shape))
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        frames = normalise(stack[:9])
        save_png(os.path.join(args.out, name + "_widefield.png"), widefield(frames))
        save_png(os.path.join(args.out, name + "_vsrsim.png"), reconstruct(net, frames, device))
        print("Reconstructed", path)


if __name__ == "__main__":
    main()
