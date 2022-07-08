import sys
import numpy as np
from numpy import pi, cos, sin
import math
from skimage import io, transform
import glob
import os
import argparse
from multiprocessing import Pool
import subprocess
import im_form_model.SIMulator_functions
import im_form_model.SeqSIMulator_functions
import run
import shutil
import wandb

# ------------ Options --------------
from options import parser
parser.add_argument('--sourceimages_path', type=str, default='/Volumes/datasets/DIV2K/DIV2K_train_HR')
parser.add_argument('--nrep', type=int, default=1, help='instances of same source image')
parser.add_argument('--datagen_workers', type=int, default=8, help='')
parser.add_argument('--ext', nargs='+', default=['png'])


# SIM options to control from command line
parser.add_argument('--Nshifts', type=int, default=3)
parser.add_argument('--Nangles', type=int, default=3)
parser.add_argument('--k2', type=float, default=126.0)
parser.add_argument('--k2_err', type=float, default=30.0)
parser.add_argument('--PSFOTFscale', type=float, default=0.9)
parser.add_argument('--ModFac', type=float, default=0.8)
parser.add_argument('--usePSF', type=int, default=0)
parser.add_argument('--NoiseLevel', type=float, default=8)
parser.add_argument('--NoiseLevelRandFac', type=float, default=8)
parser.add_argument('--phaseErrorFac', type=float, default=0.33) # pi/3 quite large but still feasible
parser.add_argument('--alphaErrorFac', type=float, default=0.33) # pi/3 quite large but still feasible
parser.add_argument('--angleError', type=float, default=10) # pi/3 quite large but still feasible
parser.add_argument('--usePoissonNoise', action='store_true')
parser.add_argument('--dontShuffleOrientations', action='store_true')
parser.add_argument('--dataonly', action='store_true')
parser.add_argument('--applyOTFtoGT', action='store_true')
parser.add_argument('--noStripes', action='store_true')
parser.add_argument('--seqSIM', action='store_true')
parser.add_argument('--skip_datagen', action='store_true')

opt = parser.parse_args()

if opt.root == 'auto':
    opt.root = opt.out + '_SIMdata'



# ------------ Parameters-------------
def GetParams(): # uniform randomisation
    SIMopt = argparse.Namespace()

    # phase shifts for each stripe
    SIMopt.Nshifts = opt.Nshifts
    # number of orientations of stripes
    SIMopt.Nangles = opt.Nangles
    # used to adjust PSF/OTF width
    SIMopt.scale = opt.PSFOTFscale# + 0.1*(np.random.rand()-0.5)
    # modulation factor
    SIMopt.ModFac = opt.ModFac# + 0.3*(np.random.rand()-0.5)
    # orientation offset
    SIMopt.alpha = opt.alphaErrorFac*pi*(np.random.rand()-0.5)
    # orientation error
    SIMopt.angleError = opt.angleError*pi/180*(np.random.rand()-0.5)
    # shuffle the order of orientations
    SIMopt.shuffleOrientations = not opt.dontShuffleOrientations
    # random phase shift errors
    SIMopt.phaseError = opt.phaseErrorFac*pi*(0.5-np.random.rand(SIMopt.Nangles, SIMopt.Nshifts))
    # mean illumination intensity
    SIMopt.meanInten = np.ones(SIMopt.Nangles)*0.5
    # amplitude of illumination intensity above mean
    SIMopt.ampInten = np.ones(SIMopt.Nangles)*0.5*SIMopt.ModFac
    # illumination freq
    SIMopt.k2 = opt.k2 + opt.k2_err*(np.random.rand()-0.5)
    # noise type
    SIMopt.usePoissonNoise = opt.usePoissonNoise
    # noise level (percentage for Gaussian)
    SIMopt.NoiseLevel = opt.NoiseLevel + opt.NoiseLevelRandFac*(np.random.rand()-0.5)
    # 1(to blur using PSF), 0(to blur using OTF)
    SIMopt.UsePSF = opt.usePSF
    # include OTF and GT in stack
    SIMopt.OTF_and_GT = True
    # use a blurred target (according to theoretical optimal construction)
    SIMopt.applyOTFtoGT = opt.applyOTFtoGT
    # whether to simulate images using just widefield illumination 
    SIMopt.noStripes = opt.noStripes


    return SIMopt



# ------------ Main loop --------------
def processImage(file):
    if 'npy' in opt.ext:
        Io = np.load(file, allow_pickle=True) / 255
        filename = os.path.basename(file).replace('.npy', '')

        if len(Io.shape) > 2 and Io.shape[2] > 3:
            Io = Io[:,:,8] # assuming t-stack
        elif Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale
    else:
        Io = io.imread(file) / 255
        # Io = transform.resize(Io, (opt.imageSize, opt.imageSize), anti_aliasing=True)

        if len(Io.shape) > 2 and Io.shape[2] > 1:
            Io = Io.mean(2)  # if not grayscale

        filename = os.path.basename(file).replace('.png', '')

    print('Generating SIM frames for', file)

    for n in range(opt.nrep):
        SIMopt = GetParams()
        SIMopt.outputname = '%s/%s_%d.tif' % (opt.root, filename, n)
        I = im_form_model.SIMulator_functions.Generate_SIM_Image(SIMopt, Io, opt.imageSize, opt.imageSize*opt.scale)

# ------------ Main loop --------------
def processSeqImage(file):
    if 'npy' in opt.ext:
        Io = np.load(file, allow_pickle=True) / 255
    else:
        Io = io.imread(file).transpose(1,2,0) / 255
    # Io = transform.resize(Io, (256, 256), anti_aliasing=True)

    # if len(Io.shape) > 2 and Io.shape[2] > 1:
    #     Io = Io.mean(2)  # if not grayscale

    filename = os.path.basename(file).replace('.npy', '')

    print('Generating SIM frames for', file)

    for n in range(opt.nrep):
        SIMopt = GetParams()
        SIMopt.outputname = '%s/%s_%d.tif' % (opt.root, filename, n)
        I = im_form_model.SeqSIMulator_functions.Generate_SIM_Image(SIMopt, Io)


def processSeqImageFolder(filepath):
    # Io = np.load(file, allow_pickle=True) / 255
    Io = []
    # for i in range(9):
        # im = io.imread('%s/%02d.jpg' % (filepath,(i+1))) / 255
        # im = im.mean(axis=2)
        # Io.append(im)

    for file in sorted(glob.glob('%s/*' % filepath)):
        im = io.imread(file) / 255
        if len(im.shape) > 2:
            im = im.mean(axis=2)
        Io.append(im)

    Io = np.array(Io).transpose(1,2,0) / 255
    # Io = transform.resize(Io, (256, 256), anti_aliasing=True)

    # if len(Io.shape) > 2 and Io.shape[2] > 1:
    #     Io = Io.mean(2)  # if not grayscale
    filename = os.path.basename(filepath)
    pardir = os.path.basename(os.path.abspath(os.path.join(filepath, os.pardir)))

    print('Generating SIM frames for', filepath)

    for n in range(opt.nrep):
        SIMopt = GetParams()
        SIMopt.outputname = '%s/%s_%s_%d.tif' % (opt.root, pardir, filename, n)
        I = im_form_model.SeqSIMulator_functions.Generate_SIM_Image(SIMopt, Io, opt.imageSize, opt.imageSize*opt.scale)


if __name__ == '__main__':
    
    # wandb.init(project="phd")
    # wandb.config.update(opt)
    # opt.wandb = wandb

    print(opt)

    if not opt.skip_datagen:

        os.makedirs(opt.root, exist_ok=True)
        os.makedirs(opt.out, exist_ok=True)
        
        shutil.copy2('datagen_pipeline.py',opt.out)
        shutil.copy2('im_form_model/SIMulator_functions.py',opt.out)

        files = []
        if 'imagefolder' not in opt.ext:
            for ext in opt.ext:
                files.extend(sorted(glob.glob(opt.sourceimages_path + "/*." + ext)))
        else:
            print('looking in opt',opt.sourceimages_path)
            folders = glob.glob("%s/*" % opt.sourceimages_path)
            for folder in folders:
                subentities = glob.glob("%s/*" % folder)
                if len(subentities) > 0:
                    if subentities[0].endswith(('.jpg','.png')):
                        # found a directory of sequence of individual frames
                        files.extend(folders)
                        break
                    # assume that subentities are tiff stacks
                    files.extend(subentities)

        if len(files) == 0:
            print('source images not found')
            sys.exit(0)
        elif opt.ntrain + opt.ntest > opt.nrep*len(files) and opt.ntrain + opt.ntest > 0:
            print('ntrain + opt.ntest is too high given nrep and number of source images')
            sys.exit(0)
        elif opt.nch_in > opt.Nangles*opt.Nshifts:
            print('nch_in cannot be greater than Nangles*Nshifts - not enough SIM frames')
            sys.exit(0)
        
        files = files[:math.ceil( (opt.ntrain + opt.ntest) / opt.nrep )]

        if opt.ntrain + opt.ntest > 0: # if == 0, use all
            files = files[:math.ceil( (opt.ntrain + opt.ntest) / opt.nrep )]

        for file in files:
            print(file)

        with Pool(opt.datagen_workers) as p:
            if not opt.seqSIM:
                p.map(processImage,files)
            elif 'imagefolder' not in opt.ext:
                p.map(processSeqImage,files)
            else:
                p.map(processSeqImageFolder,files) # processSeqImage if using tif files instead of folders of jpgs


        print('Done generating images,',opt.root)


    # cmd = '\npython run.py ' + ' '.join(sys.argv[:])
    # print(cmd,end='\n\n')
    # subprocess.Popen(cmd,shell=True)
    if not opt.dataonly:
        print('Now starting training:\n')
        
        run.main(opt)
    
