"""Regenerate the README figures from the example data.

    python reconstruct.py --examples
    python reconstruct.py --rolling examples/Sequence_microtubules.tif
    git clone https://github.com/charlesnchr/ML-SIM ../ML-SIM
    python fig/make_figures.py --mlsim ../ML-SIM

Every VSR-SIM image is the output of the released weights, read from the
folder that reconstruct.py writes. ML-SIM is the comparison, run here with
its own released weights from a clone of its repository. The motion benchmark
simulates acquisitions of the ten DIV2K images in that clone, so it also needs
the clone. Display contrast is stretched identically for every method in a
panel. Needs matplotlib, pillow and ffmpeg (for the GIF).
"""

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import io, transform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import reconstruct as vsrsim  # noqa: E402
import simulate_sequence as sim  # noqa: E402

BG = (13, 17, 23)
INK = (240, 246, 252)
MUTED = (139, 148, 158)
MAGENTA = (255, 70, 200)
GREEN = (60, 230, 110)
GREY = (255, 255, 255)

SEQUENCE = "Sequence_microtubules"
# display range of the sequence as percentiles of each movie, the background goes to black
BLACK_WHITE = (30, 99.9)


def font(size, weight="Regular"):
    for path in ("/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight,
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight != "Regular" else "")):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read(path):
    return np.asarray(io.imread(path)).astype("float32") / 255


def upsample(image, factor=2):
    """Bicubic upsampling, to show images of the camera grid next to the 2x reconstructions."""
    h, w = image.shape
    return np.clip(transform.resize(image, (factor * h, factor * w), order=3), 0, 1)


def psnr(image, target, border=4):
    """PSNR in dB for images in [0, 1], ignoring a border as inference_options.py does."""
    a, b = image[border:-border, border:-border], target[border:-border, border:-border]
    return 10 * np.log10(1 / np.mean((a - b) ** 2))


def stretch(image, limits):
    lo, hi = limits
    return np.clip((image - lo) / (hi - lo + 1e-9), 0, 1)


def colourise(grey, colour):
    """Black to the given colour to white, like the lookup tables used in microscopy."""
    c = np.asarray(colour, dtype="float32") / 255
    ramp = np.clip(grey[..., None] * 1.6, 0, 1) * c
    white = np.clip((grey[..., None] - 0.55) / 0.45, 0, 1)
    return np.clip(ramp + white * (1 - ramp), 0, 1)


def tile(image, colour, size=None, limits=None):
    """An RGB PIL tile of a grey image in [0, 1]."""
    if limits is None:
        limits = np.percentile(image, (0.5, 99.9))
    rgb = colourise(stretch(image, limits), colour) if colour != GREY else np.repeat(stretch(image, limits)[..., None], 3, 2)
    out = Image.fromarray((rgb * 255).astype("uint8"))
    return out.resize((size, size), Image.LANCZOS) if size and size != out.size[0] else out


def pill(draw, xy, text, fnt, anchor="la", fill=(0, 0, 0), ink=INK):
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=fnt, anchor=anchor)
    draw.rounded_rectangle((x0 - 8, y0 - 5, x1 + 8, y1 + 5), radius=6, fill=fill)
    draw.text(xy, text, font=fnt, anchor=anchor, fill=ink)


# ---------------------------------------------------------------------------
# ML-SIM, the comparison


def load_mlsim(repo, device):
    """Load ML-SIM's own reconstruct.py and released weights from a clone of its repository."""
    sys.path.append(repo)
    spec = importlib.util.spec_from_file_location("mlsim_reconstruct", os.path.join(repo, "reconstruct.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.load_model(os.path.join(repo, module.WEIGHTS_NAME), device)
    return lambda stack: module.reconstruct(net, stack, device)[1]


# ---------------------------------------------------------------------------
# Hero animation: rolling reconstruction of the simulated sequence


def sequence_panels(recon_dir, examples_dir, mlsim):
    """Wide-field, ML-SIM, VSR-SIM and ground truth for every frame of the sequence.

    ML-SIM needs the patterns in a fixed order, so it gets the stacks that start
    a pattern cycle, one image per 9 raw frames. Each image is shown until the
    next one is complete. VSR-SIM gives one image per raw frame.
    """
    raw = io.imread(os.path.join(examples_dir, SEQUENCE + ".tif"))
    truth = read(os.path.join(examples_dir, SEQUENCE + "_ground_truth.tif"))
    out = read(os.path.join(recon_dir, SEQUENCE + "_vsrsim.tif"))
    wide = read(os.path.join(recon_dir, SEQUENCE + "_widefield.tif"))
    stacks = [upsample(mlsim(raw[i:i + 9])) for i in range(0, len(raw) - 8, 9)]
    n = len(out)
    return {
        "widefield": np.array([upsample(w) for w in wide]),
        "mlsim": np.array([stacks[min(t // 9, len(stacks) - 1)] for t in range(n)]),
        "vsrsim": out,
        "truth": truth[4:4 + n],  # a window is assigned to its centre frame
    }


HERO_TILES = (
    ("widefield", "Wide-field", "average of 9 raw frames"),
    ("mlsim", "ML-SIM", "1 image per 9 raw frames"),
    ("vsrsim", "VSR-SIM", "1 image per raw frame"),
    ("truth", "Ground truth", "the simulated sample"),
)


def make_hero(panels, out, size=224, fps=12, colours=32):
    gap, margin, caption_h = 12, 16, 46
    n = len(panels["vsrsim"])
    crop = (panels["vsrsim"].shape[-1] - size) // 2
    limits = {k: np.percentile(v, BLACK_WHITE) for k, v in panels.items()}
    width = 2 * margin + len(HERO_TILES) * size + (len(HERO_TILES) - 1) * gap
    caption, small = font(16, "Bold"), font(13)
    tmp = tempfile.mkdtemp()
    try:
        for t in range(n):
            canvas = Image.new("RGB", (width, margin + size + caption_h), BG)
            draw = ImageDraw.Draw(canvas)
            for i, (key, name, note) in enumerate(HERO_TILES):
                x0 = margin + i * (size + gap)
                frame = panels[key][t, crop:crop + size, crop:crop + size]
                canvas.paste(tile(frame, MAGENTA, limits=limits[key]), (x0, margin))
                draw.text((x0, margin + size + 12), name, font=caption, fill=INK)
                draw.text((x0 + size, margin + size + 14), note, font=small, fill=MUTED, anchor="ra")
            canvas.save(os.path.join(tmp, "f%04d.png" % t))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
             "-vf", "split[a][b];[a]palettegen=max_colors=%d:stats_mode=full[p];[b][p]paletteuse=dither=none" % colours,
             out], check=True)
    finally:
        shutil.rmtree(tmp)


def make_social(panels, out, frame=20):
    """1280x640 image for the repository's social preview."""
    w, h, size = 1280, 640, 256
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 52), "VSR-SIM", font=font(84, "Black"), fill=INK)
    draw.text((66, 156), "A video transformer that reconstructs super-resolution microscopy of moving samples",
              font=font(29, "Medium"), fill=MUTED)
    draw.text((w - 64, 100), "arXiv 2203.00030", font=font(22), fill=MUTED, anchor="ra")
    gap = 22
    crop = (panels["vsrsim"].shape[-1] - size) // 2
    x = (w - (len(HERO_TILES) * size + (len(HERO_TILES) - 1) * gap)) // 2
    for key, name, _ in HERO_TILES:
        frame_img = panels[key][frame, crop:crop + size, crop:crop + size]
        img = tile(frame_img, MAGENTA, limits=np.percentile(panels[key], BLACK_WHITE))
        canvas.paste(img, (x, 250))
        draw.text((x, 250 + size + 14), name, font=font(22, "Bold"), fill=INK)
        x += size + gap
    draw.text((w // 2, h - 40), "Simulated sample moving one camera pixel per raw frame", font=font(20), fill=MUTED, anchor="mm")
    canvas.save(out, optimize=True)


# ---------------------------------------------------------------------------
# Results on the example stacks

# stack, label, crop (x, y) of the 2x image
WITH_TRUTH = (
    ("Dynamic_microtubules", "Microtubules", (500, 330)),
    ("Synthetic_birds", "Flamingos", (300, 180)),
    ("Synthetic_city", "Street", (380, 420)),
    ("Synthetic_wings", "Bird", (330, 330)),
)
EXPERIMENTAL = (
    ("Static_endoplasmic", "Endoplasmic reticulum", GREEN, (300, 330)),
    ("Static_microtubules", "Microtubules", MAGENTA, (360, 360)),
)


def example(recon_dir, examples_dir, mlsim, name):
    """Wide-field, ML-SIM, VSR-SIM and, if the stack has one, the ground truth, all on the 2x grid."""
    stack = io.imread(os.path.join(examples_dir, name + ".tif"))
    frames = vsrsim.normalise(stack[:9])
    images = {
        "widefield": upsample(frames.mean(0)),
        "mlsim": upsample(mlsim(stack)),
        "vsrsim": read(os.path.join(recon_dir, name + "_vsrsim.png")),
    }
    if len(stack) == 14:
        images["truth"] = sim.ground_truth_of(stack).astype("float32")
    return images


def make_results(recon_dir, examples_dir, mlsim, out, size=300):
    """Simulated acquisitions of moving scenes. The PSNR is for the full image, the tile is a crop."""
    columns = (("widefield", "WIDE-FIELD"), ("mlsim", "ML-SIM"), ("vsrsim", "VSR-SIM"), ("truth", "GROUND TRUTH"))
    gap, margin, left, head = 10, 20, 44, 52
    width = left + margin + len(columns) * size + (len(columns) - 1) * gap + margin
    height = head + len(WITH_TRUTH) * size + (len(WITH_TRUTH) - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    scores = {}
    for col, (key, title) in enumerate(columns):
        x0 = left + margin + col * (size + gap)
        draw.text((x0 + size // 2, head // 2), title, font=font(23, "Bold"),
                  fill=INK if key == "vsrsim" else MUTED, anchor="mm")
    for row, (name, label, (cx, cy)) in enumerate(WITH_TRUTH):
        images = example(recon_dir, examples_dir, mlsim, name)
        y0 = head + row * (size + gap)
        limits = np.percentile(images["truth"][cy:cy + size, cx:cx + size], (0.5, 99.9))
        scores[name] = {k: float(psnr(images[k], images["truth"])) for k in ("widefield", "mlsim", "vsrsim")}
        for col, (key, _) in enumerate(columns):
            img = tile(images[key][cy:cy + size, cx:cx + size], GREY, limits=limits)
            if key in scores[name]:
                pill(ImageDraw.Draw(img), (size - 10, size - 12), "%.1f dB" % scores[name][key], font(15, "Bold"), anchor="rs")
            canvas.paste(img, (left + margin + col * (size + gap), y0))
        strip = Image.new("RGB", (size, 36), BG)
        ImageDraw.Draw(strip).text((size // 2, 18), label, font=font(20, "Bold"), fill=INK, anchor="mm")
        canvas.paste(strip.rotate(90, expand=True), (margin - 4, y0))
    canvas.save(out, optimize=True)
    return scores


def make_experimental(recon_dir, examples_dir, mlsim, out, size=380):
    """Real acquisitions from an SLM-based SIM microscope."""
    gap, margin, left, head = 10, 20, 44, 52
    width = left + margin + 2 * size + gap + margin
    height = head + len(EXPERIMENTAL) * size + (len(EXPERIMENTAL) - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("WIDE-FIELD", "VSR-SIM")):
        x0 = left + margin + col * (size + gap)
        draw.text((x0 + size // 2, head // 2), title, font=font(23, "Bold"), fill=INK if col else MUTED, anchor="mm")
    for row, (name, label, colour, (cx, cy)) in enumerate(EXPERIMENTAL):
        images = example(recon_dir, examples_dir, mlsim, name)
        y0 = head + row * (size + gap)
        for col, key in enumerate(("widefield", "vsrsim")):
            crop = images[key][cy:cy + size, cx:cx + size]
            canvas.paste(tile(crop, colour), (left + margin + col * (size + gap), y0))
        strip = Image.new("RGB", (size, 36), BG)
        ImageDraw.Draw(strip).text((size // 2, 18), label, font=font(20, "Bold"), fill=INK, anchor="mm")
        canvas.paste(strip.rotate(90, expand=True), (margin - 4, y0))
    canvas.save(out, optimize=True)


# ---------------------------------------------------------------------------
# Motion benchmark: PSNR against the speed of the sample

SPEEDS = (0, 0.25, 0.5, 1, 1.5, 2, 3)  # camera pixels per raw frame


def drifting_stack(image, speed, size, rng):
    """Simulate one 9-frame acquisition of an image that drifts in a straight line.

    The image fills the field of view, as in the training data generator. It is
    first resized to 4x the camera grid, so that a drift of a quarter of a
    camera pixel is a whole pixel and needs no interpolation. Returns the raw
    frames and the ground truth, which is the centre frame on the 2x grid,
    normalised like the targets of the training data.
    """
    field = 4 * size
    reach = int(np.ceil(4 * 4 * max(SPEEDS))) + 1
    image = transform.resize(image, (field + 2 * reach, field + 2 * reach), order=3, anti_aliasing=True)
    angle = rng.uniform(0, 2 * np.pi)
    frames = []
    for t in range(9):
        d = 4 * speed * (t - 4)
        y0, x0 = reach + int(round(d * np.sin(angle))), reach + int(round(d * np.cos(angle)))
        frames.append(image[y0:y0 + field, x0:x0 + field])
    truth = transform.resize(frames[4], (2 * size, 2 * size), order=3, anti_aliasing=True)
    truth = (truth - truth.min()) / (truth.max() - truth.min())
    camera = np.array([transform.resize(f, (size, size), order=3, anti_aliasing=True) for f in frames])
    raw = sim.acquire(camera, noise=8, rng=rng, k2=126 + 30 * (rng.random() - 0.5),
                      alpha=0.05 * np.pi * (rng.random() - 0.5))
    return raw, truth.astype("float32")


def motion_benchmark(net, device, mlsim, image_dir, size=192):
    """PSNR of wide-field, ML-SIM and VSR-SIM on the same simulated stacks, per speed."""
    rows = []
    for i, path in enumerate(sorted(glob.glob(os.path.join(image_dir, "*.png")))):
        image = io.imread(path).astype("float64").mean(2) / 255
        for speed in SPEEDS:
            rng = np.random.default_rng(i)  # the same direction, pattern and noise at every speed
            raw, truth = drifting_stack(image, speed, size, rng)
            raw8 = (raw * 255).round().astype("uint8")
            frames = vsrsim.normalise(raw8)
            rows.append({
                "image": os.path.basename(path), "speed": speed,
                "widefield": float(psnr(upsample(frames.mean(0)), truth)),
                "mlsim": float(psnr(upsample(mlsim(raw8)), truth)),
                "vsrsim": float(psnr(vsrsim.reconstruct(net, frames, device), truth)),
            })
            print(rows[-1], flush=True)
    return rows


def make_motion_plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for weight in ("Regular", "Bold"):
        path = "/usr/share/fonts/truetype/lato/Lato-%s.ttf" % weight
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = "Lato"

    bg, ink, muted = [tuple(v / 255 for v in c) for c in (BG, INK, MUTED)]
    series = (("vsrsim", "VSR-SIM", "#3987e5", "o"), ("mlsim", "ML-SIM", "#d95926", "s"),
              ("widefield", "Wide-field", muted, "D"))
    fig = plt.figure(figsize=(9, 4.4), dpi=160, facecolor=bg)
    ax = fig.add_axes((0.085, 0.16, 0.78, 0.76), facecolor=bg)
    for key, name, colour, marker in series:
        mean = [np.mean([r[key] for r in rows if r["speed"] == s]) for s in SPEEDS]
        ax.plot(SPEEDS, mean, color=colour, lw=2, marker=marker, ms=6.5, mec=bg, mew=1.2, clip_on=False, label=name)
        ax.annotate(name, (SPEEDS[-1], mean[-1]), xytext=(10, 0), textcoords="offset points",
                    color=ink, fontsize=14, fontweight="bold", va="center")
    ax.set_xlabel("Motion of the sample (camera pixels per raw frame)", color=muted, fontsize=13)
    ax.set_ylabel("PSNR (dB)", color=muted, fontsize=13)
    ax.set_xlim(-0.08, SPEEDS[-1])
    ax.set_xticks(SPEEDS)
    ax.set_xticklabels(["%g" % s for s in SPEEDS])
    ax.tick_params(colors=muted, labelsize=12, length=0)
    ax.grid(axis="y", color=muted, alpha=0.18, lw=0.8)
    for side, s in ax.spines.items():
        s.set_visible(side == "bottom")
        s.set_color(muted)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mlsim", required=True, help="clone of https://github.com/charlesnchr/ML-SIM")
    p.add_argument("--recon", default="reconstructions", help="folder written by reconstruct.py")
    p.add_argument("--examples", default="examples", help="folder with the example stacks")
    p.add_argument("--out", default=HERE)
    p.add_argument("--benchmark", action="store_true",
                   help="rerun the motion benchmark instead of plotting fig/motion-benchmark.json")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    mlsim = load_mlsim(args.mlsim, device)

    panels = sequence_panels(args.recon, args.examples, mlsim)
    make_hero(panels, os.path.join(args.out, "hero.gif"))
    make_social(panels, os.path.join(args.out, "social-preview.png"))
    scores = make_results(args.recon, args.examples, mlsim, os.path.join(args.out, "results.png"))
    for name, s in scores.items():
        print("%-22s wide-field %.2f dB, ML-SIM %.2f dB, VSR-SIM %.2f dB" % (name, s["widefield"], s["mlsim"], s["vsrsim"]))
    make_experimental(args.recon, args.examples, mlsim, os.path.join(args.out, "experimental.png"))

    benchmark = os.path.join(args.out, "motion-benchmark.json")
    if args.benchmark:
        net = vsrsim.load_model(os.path.join(ROOT, vsrsim.WEIGHTS_NAME), device)
        rows = motion_benchmark(net, device, mlsim, os.path.join(args.mlsim, "MLSIM_datagen", "DIV2K_subset"))
        with open(benchmark, "w") as f:
            json.dump(rows, f, indent=1)
    with open(benchmark) as f:
        make_motion_plot(json.load(f), os.path.join(args.out, "motion-benchmark.png"))
