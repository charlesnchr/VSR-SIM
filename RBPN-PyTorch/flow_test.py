import pyflow
import numpy as np
from skimage import io
from PIL import Image


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    if len(im1.shape) < 3:
        im1 = np.expand_dims(im1, axis=2)
        im2 = np.expand_dims(im2, axis=2)
    print(im1.shape)
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    print(flow.shape)
    return flow

im = io.imread('test.tif')
im1 = im[0]
im2 = im[2]


im1 = Image.fromarray(im1)
im2 = Image.fromarray(im2)

get_flow(im1,im2)
