"""Reconstruct raw SIM stacks with the pre-trained VSR-SIM model.

    python reconstruct.py --examples
    python reconstruct.py stack.tif
    python reconstruct.py --rolling sequence.tif

Each input is a TIFF stack whose first 9 frames are the raw SIM acquisition
(3 orientations x 3 phase shifts, in the order of acquisition). For every stack the script
writes the VSR-SIM reconstruction, at twice the pixel count of the raw frames
in each direction, and the wide-field image (the mean of the raw frames) to
the output folder. The weights are downloaded on first use.

With --rolling every frame of the input is a raw frame of one continuous
acquisition, with the 9 illumination patterns repeating. A window of 9 frames
slides along the sequence one frame at a time and each position gives one
reconstruction, so N raw frames give N - 8 images instead of N / 9.

--examples downloads the example data to examples/ and reconstructs the
stacks in it. The folder also gets a simulated sequence to try --rolling on.
"""

import argparse
import hashlib
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io

from basicsr.archs.swinir_rcab_arch import SwinIR_RCAB

RELEASE = "https://github.com/charlesnchr/VSR-SIM/releases/download/v1.0/"
MIRROR = "https://huggingface.co/spaces/charlesnchr/VSR-SIM/resolve/main/"

WEIGHTS_NAME = "vsr-sim_swinir-rcab_20220202.pth"
WEIGHTS_SHA256 = "33e053fe6cf54effac05de7385f94e7385de9ba20d59819f843aa396a4017b3a"
WEIGHTS_URLS = (RELEASE + WEIGHTS_NAME, MIRROR + "model/swinir_rcab_nov2021.pth")

# name and SHA-256 of the example data, 9-frame stacks first, then the sequence for --rolling
EXAMPLE_STACKS = {
    "Static_endoplasmic.tif": "2535eb96855a053b93a5ede741e6a4c97401964e67a332cc09119200d323e82f",
    "Static_microtubules.tif": "fa1a04c5e1d9982090286099c7214630eee86794327860de7ba8a99704f194f0",
    "Dynamic_microtubules.tif": "a61d88cce760a5aca6c12f7914ea67e5202a0564ec3c6085d8bb5cfbdded3edf",
    "Synthetic_beads.tif": "3351e0b67fe12c3b2d7de2bd04379bd56871d1bf1b8318964132691acd6a8885",
    "Synthetic_birds.tif": "c656985346c579f3c0a2f169f615d60d46a3c234b78cee3f6469472e10ec4ae7",
    "Synthetic_city.tif": "98717e9d82c83c9161b18dc2133c14b74a78e8e0d15fe4ca27dea9e827df1aa9",
    "Synthetic_wings.tif": "9d6b8b3bdb0326888f5dd8aa5741780e62ebd92ddb6013f6112f3ba9421c4e13",
}
EXAMPLE_SEQUENCE = {
    "Sequence_microtubules.tif": "5c2056f416394d80c5a015f2d778c534238659f775fc272492e1e9a9b708bb7d",
    "Sequence_microtubules_ground_truth.tif": "a9156c7dbb3dfac09938c93020f4786f0c7b7314280476da188b4c053585da03",
}

WINDOW = 8  # side of the attention windows, the input is padded to a multiple of it


def download(urls, dest, sha256):
    """Fetch a file, trying each URL in turn, and check its hash."""
    for url in urls:
        print("Downloading", url)
        try:
            urllib.request.urlretrieve(url, dest + ".part")
        except OSError as err:
            print("  failed:", err)
            continue
        with open(dest + ".part", "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest == sha256:
            os.replace(dest + ".part", dest)
            return
        print("  checksum mismatch, discarding the file")
        os.remove(dest + ".part")
    raise SystemExit("Could not download %s. Fetch it by hand from %s" % (os.path.basename(dest), urls[0]))


def fetch_examples(folder):
    """Download the example data that is not in the folder yet, return the 9-frame stacks."""
    os.makedirs(folder, exist_ok=True)
    for name, sha256 in {**EXAMPLE_STACKS, **EXAMPLE_SEQUENCE}.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            download((RELEASE + name, MIRROR + name), path, sha256)
    return [os.path.join(folder, name) for name in EXAMPLE_STACKS]


def load_model(weights, device):
    """Build the network of the released checkpoint and load its weights."""
    net = SwinIR_RCAB(
        upscale=2, in_chans=9, out_chans=1, img_size=128, window_size=WINDOW, img_range=1.0,
        depths=[6] * 5, embed_dim=64, num_heads=[8] * 5, mlp_ratio=2,
        upsampler=None, resi_connection="1conv", pixelshuffleFactor=1,
    )
    if not os.path.isfile(weights):
        download(WEIGHTS_URLS, weights, WEIGHTS_SHA256)
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


def rolling(net, frames, device):
    """Yield one reconstruction for every window of 9 consecutive frames."""
    for t in range(len(frames) - 8):
        yield reconstruct(net, frames[t:t + 9], device)


def save(path, image):
    io.imsave(path, (image * 255).round().astype("uint8"), check_contrast=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stacks", nargs="*", help="TIFF stacks with at least 9 frames")
    p.add_argument("--examples", action="store_true", help="download the example data and reconstruct it")
    p.add_argument("--weights", default=WEIGHTS_NAME, help="path to the model file")
    p.add_argument("--rolling", action="store_true",
                   help="treat every frame as a raw frame and reconstruct each window of 9")
    p.add_argument("--out", default="reconstructions", help="output folder")
    p.add_argument("--cpu", action="store_true", help="run on CPU even if a GPU is available")
    args = p.parse_args()
    if args.examples and args.rolling:
        p.error("--examples reconstructs 9-frame stacks, run --rolling on the sequence afterwards")
    if args.examples:
        args.stacks += fetch_examples("examples")
    if not args.stacks:
        p.error("give at least one stack, or --examples")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args.weights, device)
    os.makedirs(args.out, exist_ok=True)

    for path in args.stacks:
        stack = io.imread(path)
        if stack.ndim != 3 or stack.shape[0] < 9:
            print("Skipping %s: expected a stack of at least 9 frames, got shape %s" % (path, stack.shape))
            continue
        name = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        if args.rolling:
            frames = normalise(stack)
            n = len(frames) - 8
            images = []
            for t, image in enumerate(rolling(net, frames, device)):
                images.append(image)
                print("\r%s: window %d of %d" % (path, t + 1, n), end="", flush=True)
            print()
            save(name + "_widefield.tif", np.array([widefield(frames[t:t + 9]) for t in range(n)]))
            save(name + "_vsrsim.tif", np.array(images))
        else:
            frames = normalise(stack[:9])
            save(name + "_widefield.png", widefield(frames))
            save(name + "_vsrsim.png", reconstruct(net, frames, device))
        print("Reconstructed", path)

    if args.examples:
        print("The results are in %s. For a moving sample, run" % args.out)
        print("    python reconstruct.py --rolling examples/%s" % next(iter(EXAMPLE_SEQUENCE)))


if __name__ == "__main__":
    main()
