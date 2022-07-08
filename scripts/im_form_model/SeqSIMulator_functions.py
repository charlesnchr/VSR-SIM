import numpy as np
from numpy import pi, cos, sin
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import io,transform
from scipy.signal import convolve2d
import scipy.special


def PsfOtf(w, scale):
    # AIM: To generate PSF and OTF using Bessel function
    # INPUT VARIABLES
    #   w: image size
    #   scale: a parameter used to adjust PSF/OTF width
    # OUTPUT VRAIBLES
    #   yyo: system PSF
    #   OTF2dc: system OTF
    eps = np.finfo(np.float64).eps

    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    X, Y = np.meshgrid(x, y)

    # Generation of the PSF with Besselj.
    R = np.sqrt(np.minimum(X, np.abs(X-w))**2+np.minimum(Y, np.abs(Y-w))**2)
    yy = np.abs(2*scipy.special.jv(1, scale*R+eps) / (scale*R+eps)
                )**2  # 0.5 is introduced to make PSF wider
    yy0 = fftshift(yy)

    # Generate 2D OTF.
    OTF2d = fft2(yy)
    OTF2dmax = np.max([np.abs(OTF2d)])
    OTF2d = OTF2d/OTF2dmax
    OTF2dc = np.abs(fftshift(OTF2d))

    return (yy0, OTF2dc)


def conv2(x, y, mode='same'):
    # Make it equivalent to Matlab's conv2 function
    # https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def SIMimages(opt, DIo, PSFo, OTFo):

    # AIM: to generate raw sim images
    # INPUT VARIABLES
    #   k2: illumination frequency
    #   DIo: specimen image
    #   PSFo: system PSF
    #   OTFo: system OTF
    #   UsePSF: 1 (to blur SIM images by convloving with PSF)
    #           0 (to blur SIM images by truncating its fourier content beyond OTF)
    #   NoiseLevel: percentage noise level for generating gaussian noise
    # OUTPUT VARIABLES
    #   frames:  raw sim images
    #   DIoTnoisy: noisy wide field image
    #   DIoT: noise-free wide field image

    w = DIo.shape[0]
    wo = w/2
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, w-1, w)
    [X, Y] = np.meshgrid(x, y)

    # Illuminating pattern

    # orientation direction of illumination patterns
    orientation = np.zeros(opt.Nangles)
    for i in range(opt.Nangles):
        orientation[i] = i*pi/opt.Nangles + opt.alpha + opt.angleError

    if opt.shuffleOrientations:
        np.random.shuffle(orientation)

    # illumination frequency vectors
    k2mat = np.zeros((opt.Nangles, 2))
    for i in range(opt.Nangles):
        theta = orientation[i]
        k2mat[i, :] = (opt.k2/w)*np.array([cos(theta), sin(theta)])

    # illumination phase shifts along directions with errors
    ps = np.zeros((opt.Nangles, opt.Nshifts))
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            ps[i_a, i_s] = 2*pi*i_s/opt.Nshifts + opt.phaseError[i_a, i_s]

    # illumination patterns
    frames = []
    sigs = []
    for i_a in range(opt.Nangles):
        for i_s in range(opt.Nshifts):
            # illuminated signal
            if not opt.noStripes:
                sig = opt.meanInten[i_a] + opt.ampInten[i_a] * cos(2*pi*(k2mat[i_a, 0]*(X-wo) +
                            k2mat[i_a, 1]*(Y-wo))+ps[i_a, i_s])
            else:
                sig = 1 # simulating widefield
            sigs.append(sig)


    # random shift
    shift_num = np.random.randint(0,len(sigs))
    ind = list(range(len(sigs)))
    ind = np.roll(ind, shift_num)

    for signal_idx,illum_idx in enumerate(ind):
        sup_sig = DIo[:,:,signal_idx]*sigs[illum_idx]  # superposed signal

        # superposed (noise-free) Images
        if opt.UsePSF == 1:
            ST = conv2(sup_sig, PSFo, 'same')
        else:
            ST = np.real(ifft2(fft2(sup_sig)*fftshift(OTFo)))

        # Gaussian noise generation
        aNoise = opt.NoiseLevel/100  # noise
        # SNR = 1/aNoise
        # SNRdb = 20*log10(1/aNoise)

        nST = np.random.normal(0, aNoise*np.std(ST, ddof=1), (w, w))
        NoiseFrac = 1  # may be set to 0 to avoid noise addition
        # noise added raw SIM images
        STnoisy = ST + NoiseFrac*nST
        frames.append(STnoisy)

    return frames


# %%
def Generate_SIM_Image(opt, Io, in_dim=512, gt_dim=1024):

    DIo = Io.astype('float')

    if in_dim is not None:
        DIo = transform.resize(Io, (in_dim, in_dim), anti_aliasing=True, order=3)

    w = DIo.shape[0]

    # Generation of the PSF with Besselj.

    PSFo, OTFo = PsfOtf(w, opt.scale)

    frames = SIMimages(opt, DIo, PSFo, OTFo)


    if opt.OTF_and_GT:
        frames.append(OTFo)

        gt_img = Io[:,:,opt.Nangles*opt.Nshifts // 2].astype('float') # center frame
        gt_img = transform.resize(gt_img, (gt_dim,gt_dim), order=3)

        if gt_dim > in_dim: # assumes a upscale factor of 2 is given
            # gt_img = skimage.transform.resize(gt_img, (gt_dim,gt_dim), order=3)
            gt11 = gt_img[:in_dim,:in_dim]
            gt21 = gt_img[in_dim:,:in_dim]
            gt12 = gt_img[:in_dim,in_dim:]
            gt22 = gt_img[in_dim:,in_dim:]
            # frames.extend([gt11,gt21,gt12,gt22])
            frames.append(gt11)
            frames.append(gt21)
            frames.append(gt12)
            frames.append(gt22)
        else:
            frames.append(Io[:,:,-1])
    stack = np.array(frames)

    # NORMALIZE

    # does not work well with partitioned GT
    # for i in range(len(stack)):
        # stack[i] = (stack[i] - np.min(stack[i])) / \
            # (np.max(stack[i]) - np.min(stack[i]))

    # normalised SIM stack
    simstack = stack[:opt.Nangles*opt.Nshifts]
    stack[:opt.Nangles*opt.Nshifts] = (simstack - np.min(simstack)) / (np.max(simstack) - np.min(simstack))

    # normalised gt
    if gt_dim > in_dim:
        gtstack = stack[-4:]
        stack[-4:] = (gtstack - np.min(gtstack)) / (np.max(gtstack) - np.min(gtstack))
        # normalised OTF
        stack[-5] = (stack[-5] - np.min(stack[-5])) / (np.max(stack[-5] - np.min(stack[-5])))
    else:
        stack[-1] = (stack[-1] - np.min(stack[-1])) / (np.max(stack[-1] - np.min(stack[-1])))
        # normalised OTF
        stack[-2] = (stack[-2] - np.min(stack[-2])) / (np.max(stack[-2] - np.min(stack[-2])))


    stack = (stack * 255).astype('uint8')

    if opt.outputname is not None:
        io.imsave(opt.outputname, stack)

    return stack
