"""Simulate one continuous SIM acquisition of a moving sample.

    python scripts/simulate_sequence.py examples/Dynamic_microtubules.tif examples/Sequence_microtubules.tif

datagen_pipeline.py writes independent stacks of nine frames for training.
This script simulates what a microscope records over a longer time: the nine
illumination patterns repeat without a pause while the sample keeps moving.
The output is a TIFF with one raw frame per time point, which is the input
that `reconstruct.py --rolling` expects. The ground truth is saved next to it.

The sample is the ground truth image stored in one of the example stacks,
moved along a circle, with beads on their own orbits on top of it. Every
path closes after `--period` frames, so the sequence loops.
"""

import argparse
import os

import numpy as np
from numpy import cos, pi, sin
from numpy.fft import fft2, fftshift, ifft2
from scipy.ndimage import shift
from skimage import io, transform

from im_form_model.SeqSIMulator_functions import PsfOtf


def illumination(w, k2=126.0, modfac=0.8, alpha=0.0, n_angles=3, n_shifts=3, ref=512):
    """The stripe patterns of one cycle, as in SeqSIMulator_functions.SIMimages.

    k2 is the stripe frequency in cycles per `ref` pixels, the convention of
    the training data, so a pattern keeps its period when the frame size changes.
    """
    x = np.arange(w)
    X, Y = np.meshgrid(x, x)
    patterns = []
    for i_a in range(n_angles):
        theta = i_a * pi / n_angles + alpha
        kx, ky = (k2 / ref) * cos(theta), (k2 / ref) * sin(theta)
        for i_s in range(n_shifts):
            phase = 2 * pi * i_s / n_shifts
            patterns.append(0.5 + 0.5 * modfac * cos(2 * pi * (kx * (X - w / 2) + ky * (Y - w / 2)) + phase))
    return patterns


def acquire(sample, noise=8.0, psf_scale=0.9, rng=None, **pattern_args):
    """Raw frames for a sample of shape (T, w, w) sampled at the camera pixel size.

    Frame t sees pattern t modulo 9. Each frame is blurred by the OTF and gets
    Gaussian noise with a standard deviation of `noise` percent of its own.
    """
    rng = rng or np.random.default_rng(0)
    w = sample.shape[-1]
    _, otf = PsfOtf(w, psf_scale)
    patterns = illumination(w, **pattern_args)
    frames = []
    for t, s in enumerate(sample):
        frame = np.real(ifft2(fft2(s * patterns[t % len(patterns)]) * fftshift(otf)))
        frames.append(frame + rng.normal(0, noise / 100 * np.std(frame, ddof=1), frame.shape))
    frames = np.array(frames)
    return (frames - frames.min()) / (frames.max() - frames.min())


def to_camera(sample_hr):
    """Downsample frames from the ground truth grid to the camera grid, half the size."""
    w = sample_hr.shape[-1] // 2
    return np.array([transform.resize(f, (w, w), anti_aliasing=True, order=3) for f in sample_hr])


def ground_truth_of(stack):
    """The 2x ground truth that the simulator stores as four quadrants at the end of a stack."""
    top = np.hstack((stack[-4], stack[-2]))
    bottom = np.hstack((stack[-3], stack[-1]))
    return np.vstack((top, bottom)).astype("float64") / 255


def moving_sample(image, n_frames, period, size, radius, n_beads, rng):
    """Frames (n_frames, 2*size, 2*size) of `image` on a circular path with orbiting beads.

    `size` and `radius` are in camera pixels, the frames are on the ground
    truth grid with twice the sampling.
    """
    hr = 2 * size
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    yy, xx = np.mgrid[:hr, :hr]

    centre = rng.uniform(0.1 * hr, 0.9 * hr, (n_beads, 2))
    orbit = 2 * rng.uniform(4, 14, n_beads)
    turns = rng.choice([-2, -1, 1, 2], n_beads)
    phase = rng.uniform(0, 2 * pi, n_beads)
    amplitude = rng.uniform(0.6, 1.0, n_beads)

    frames = []
    for t in range(n_frames):
        a = 2 * pi * t / period
        moved = shift(image, (2 * radius * sin(a), 2 * radius * cos(a)), order=3, mode="reflect")
        frame = 0.8 * moved[cy - size:cy + size, cx - size:cx + size]
        for c, r, n, p, amp in zip(centre, orbit, turns, phase, amplitude):
            bx, by = c[0] + r * cos(n * a + p), c[1] + r * sin(n * a + p)
            frame = frame + amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 1.3 ** 2))
        frames.append(frame)
    return np.clip(np.array(frames), 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="example stack that ends with its ground truth, e.g. Dynamic_microtubules.tif")
    p.add_argument("out", help="output TIFF with the raw frames")
    p.add_argument("--frames", type=int, default=80, help="number of raw frames")
    p.add_argument("--period", type=int, default=72, help="frames after which the motion repeats")
    p.add_argument("--size", type=int, default=160, help="frame size in camera pixels")
    p.add_argument("--radius", type=float, default=12, help="radius of the sample's path in camera pixels")
    p.add_argument("--beads", type=int, default=24)
    p.add_argument("--noise", type=float, default=8, help="noise level in percent, as in datagen_pipeline.py")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    image = ground_truth_of(io.imread(args.source))
    sample = moving_sample(image, args.frames, args.period, args.size, args.radius, args.beads, rng)
    raw = acquire(to_camera(sample), noise=args.noise, rng=rng)

    io.imsave(args.out, (raw * 255).round().astype("uint8"), check_contrast=False)
    gt_path = os.path.splitext(args.out)[0] + "_ground_truth.tif"
    io.imsave(gt_path, (sample * 255).round().astype("uint8"), check_contrast=False)
    speed = 2 * pi * args.radius / args.period
    print("Wrote %d raw frames to %s and the ground truth to %s" % (args.frames, args.out, gt_path))
    print("The sample moves %.2f camera pixels per frame" % speed)


if __name__ == "__main__":
    main()
