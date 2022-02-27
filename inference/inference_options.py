# Modified from https://github.com/JingyunLiang/SwinIR
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from skimage import io,transform, exposure, img_as_float
import yaml
from collections import OrderedDict
from basicsr.models import build_model
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from basicsr.archs.swinir_arch import SwinIR
from pathlib import Path

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/Set5/LRbicx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/SwinIR/Set5', help='output folder')
    parser.add_argument(
        '--task',
        type=str,
        default='classical_sr',
        help='classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--patch_size', type=int, default=None, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=0, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--large_model', action='store_true', help='Use large model, only used for real image sr')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/SwinIR/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')

    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()


    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    opt['auto_resume'] = False
    opt['is_train'] = False
    opt['dist'] = False
    opt['num_gpu'] = 1
    opt['rank'], opt['world_size'] = 0,1

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    # model = define_model(args)

    model = build_model(opt).net_g # based on train script and options

    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    model.eval()
    model = model.to(device)

    if args.task == 'jpeg_car':
        window_size = 7
    else:
        window_size = 4

    psnr = []
    ssim = []
    psnrWF = []
    ssimWF = []

    def is_low_contrast(im,threshold=0.9): # for handling unsigned floats (in contrast to skimage.exposure)
        limits = np.percentile(im,[0,100])
        ratio = (limits[1] - limits[0])
        return ratio < threshold

    files = glob.glob(os.path.join(args.input, '*.tif'))
    if len(files) == 0 and 'nostripe' in args.input:
        files = glob.glob(os.path.join(args.input.replace('-nostripe',''), '*.tif'))
        opt['use_wf_for_sisr'] = True

    files = sorted(files)


    for idx, path in enumerate(files):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        stack = io.imread(path)
        stack = img_as_float(stack)
        # stack = stack/255

        if 'singleImage' in opt['name'] or 'SISR' in opt['name']:
            if opt.get('use_wf_for_sisr') is None:
                inp = stack[4:5]
            else:
                inp = np.mean(stack[:9],axis=0,keepdims=1)
        else:
            inp = stack[:9]

        if is_low_contrast(inp):
            inp = exposure.rescale_intensity(inp,out_range=(0,1))
            # for frameidx,frame in enumerate(inp):
                # inp[frameidx] = exposure.rescale_intensity(frame,out_range=(0,1))


        # gt = None
        # if len(stack) == 14:
            # toprow = np.hstack((stack[-4,:,:],stack[-2,:,:]))
            # botrow = np.hstack((stack[-3,:,:],stack[-1,:,:]))
            # gt = np.vstack((toprow,botrow)).reshape(2*stack.shape[1],2*stack.shape[2])
            # gt = (gt * 255.0).round().astype(np.uint8)


        # newstack = []
        # for i in range(9):
            # newstack.append(stack[4])
        # newstack = np.array(newstack)
        # img = torch.from_numpy(newstack).float()

        img = torch.from_numpy(inp).float()
        img = img.unsqueeze(0).to(device)

        # stack = stack / 255
        # img = torch.from_numpy(stack[:9]).float()
        # img = img.unsqueeze(0).to(device)


        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            output = model(img)
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

        # save image
        os.makedirs(os.path.join(args.output, 'out'),exist_ok=True)
        output = output.data.squeeze().float().cpu().clamp(0,1).numpy()
        output = (output*255).round().astype(np.uint8)
        # output = exposure.match_histograms(output, gt)
        # output = exposure.rescale_intensity(output,out_range=(0,1))
        # output = np.clip(output,(0,1))
        cv2.imwrite(os.path.join(args.output, f'out/{imgname}_out.png'), output)

        # wf
        os.makedirs(os.path.join(args.output, 'wf'),exist_ok=True)
        wf = np.mean(inp,axis=0)
        wf = transform.resize(wf,output.shape,order=3)
        wf = (wf * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'wf/{imgname}_wf.png'), wf)

        # gt
        if len(stack) == 14:
            toprow = np.hstack((stack[-4,:,:],stack[-2,:,:]))
            botrow = np.hstack((stack[-3,:,:],stack[-1,:,:]))
            gt = np.vstack((toprow,botrow)).reshape(2*stack.shape[1],2*stack.shape[2])
            gt = (gt * 255.0).round().astype(np.uint8)
            comb = np.hstack((wf,output,gt))
            os.makedirs(os.path.join(args.output, 'comb'),exist_ok=True)
            os.makedirs(os.path.join(args.output, 'gt'),exist_ok=True)
            cv2.imwrite(os.path.join(args.output, f'comb/{imgname}_comb.png'), comb)
            cv2.imwrite(os.path.join(args.output, f'gt/{imgname}_gt.png'), gt)

            crop_border = 4
            psnr.append(calculate_psnr(output,gt,crop_border))
            ssim.append(calculate_ssim(output,gt,crop_border))

            psnrWF.append(calculate_psnr(wf,gt,crop_border))
            ssimWF.append(calculate_ssim(wf,gt,crop_border))

            print('PSNR: %0.3f SR / %0.3f WF, SSIM: %0.3f SR / %0.3f WF' % (psnr[-1],psnrWF[-1],ssim[-1],ssimWF[-1]))
        elif len(stack) > 9: # assumes with gt/ref with same dimensions at
            if len(stack) == 13: # legacy encoding with OTF, GT, WF, SIM ref after the SIM stack
                gt = stack[10]
            else:
                gt = stack[-1] # last index

            gt = exposure.rescale_intensity(gt.astype('float'),out_range=(0,1))
            if gt.shape[0] < output.shape[0]:
                gt = transform.resize(gt,output.shape,order=3)
            gt = (gt*255).round().astype('uint8')
            comb = np.hstack((wf,output,gt))
            os.makedirs(os.path.join(args.output, 'comb'),exist_ok=True)
            os.makedirs(os.path.join(args.output, 'gt'),exist_ok=True)
            cv2.imwrite(os.path.join(args.output, f'comb/{imgname}_comb.png'), comb)
            cv2.imwrite(os.path.join(args.output, f'gt/{imgname}_gt.png'), gt)

            crop_border = 4
            psnr.append(calculate_psnr(output,gt,crop_border))
            ssim.append(calculate_ssim(output,gt,crop_border))

            psnrWF.append(calculate_psnr(wf,gt,crop_border))
            ssimWF.append(calculate_ssim(wf,gt,crop_border))
        else:
            siblings = Path(path).parent.iterdir()
            for s in siblings:
                if '/ref' in s.as_posix():
                    for c in s.iterdir():
                        if imgname in c.as_posix():
                            gt = io.imread(c.as_posix())
                            # the ref images are -1 to 1
                            # gt = ((gt+1)/2 * 255.0).round().astype(np.uint8)
                            gt = exposure.rescale_intensity(gt.astype('float'),
                                                            out_range=(0,255)).round().astype('uint8')
                            comb = np.hstack((wf,output,gt))
                            os.makedirs(os.path.join(args.output, 'comb'),exist_ok=True)
                            os.makedirs(os.path.join(args.output, 'ref'),exist_ok=True)
                            cv2.imwrite(os.path.join(args.output, f'comb/{imgname}_comb.png'), comb)
                            cv2.imwrite(os.path.join(args.output, f'ref/{imgname}_ref.png'), gt)
                            break


    if len(psnr) > 0:
        fid = open('%s/scores.csv' % args.output, 'w')
        fid.write('psnr_sr,psnr_wf,ssim_sr,ssim_wf\n')

        for i in range(len(psnr)):
            fid.write('%0.4f,%0.4f,%0.4f,%0.4f\n' % (psnr[i],psnrWF[i],ssim[i],ssimWF[i]))

        agg_psnr = np.mean(np.array(psnr))
        agg_psnrWF = np.mean(np.array(psnrWF))
        agg_ssim = np.mean(np.array(ssim))
        agg_ssimWF = np.mean(np.array(ssimWF))

        aggheader = '\n\taggregatedScore for %d tests:' % len(psnr)
        aggscores = '\t\tPSNR: %0.3f SR / %0.3f WF, SSIM: %0.3f SR / %0.3f WF' % (agg_psnr,agg_psnrWF,agg_ssim,agg_ssimWF)
        print(aggheader)
        print(aggscores)
        agg_fid = open('%s/agg_scores.txt' % args.output, 'w')
        agg_fid.write('%s\n' % aggheader)
        agg_fid.write(aggscores)



def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.patch_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv')

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            resi_connection='1conv')

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=248,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='3conv')

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's slightly better than 1
    elif args.task == 'jpeg_car':
        model = SwinIR(
            upscale=1,
            in_chans=1,
            img_size=126,
            window_size=7,
            img_range=255.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv')
    elif args.task == 'simrec':
        model = SwinIR(
            upscale=2,
            in_chans=9,
            img_size=512,
            window_size=4,
            img_range=1,
            depths=[6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv',
            vis=True,
            **{'pixelshuffleFactor':2})


    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()


