import glob
from skimage import io
import os
import numpy as np

dirname = '20210711_test_out'
files = glob.glob('%s/**/*.png' % dirname,recursive=True)

testsets = {'showcase1':[],'showcase2':[],'showcase3':[]}

for file in files:
    for k in testsets.keys():
        if k in file:
            testsets[k].append(file)

ordered_testsets = {}

for k in testsets.keys():
    ordered_arr = ['']*len(testsets[k])
    print('ordered array now',len(ordered_arr))

    for f in testsets[k]:
        basename = os.path.basename(f)
        idx = int(basename.split('_')[0]) - 1
        ordered_arr[idx] = f
    
    ordered_testsets[k] = ordered_arr


# creating stacks
os.makedirs('%s/stacks' % dirname, exist_ok=True)

for k in testsets.keys():

    imgarr = []
    for f in ordered_testsets[k]:
        imgarr.append(io.imread(f))

    io.imsave('%s/stacks/%s.tif' % (dirname, k),np.array(imgarr))
    print('generated stack for',k)
