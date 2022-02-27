import sys
import subprocess
from pathlib import Path
import tqdm
import os
import glob
import threading
import time

# FUNCTIONS

def get_nested_folder_structure_for_experiments(folder):
    files = glob.glob('%s/inputs/%s/**/*.tif' % (rootfolder,folder),recursive=True)
    folders = {}

    for file in files:
        if 'ref' in file:
            continue
        pardir = Path(file).parents[0]
        rel_path = pardir.as_posix()
        rel_path = rel_path.split('%s/inputs/' % rootfolder)[1]
        folders[rel_path] = True

    return list(folders.keys())



def run(cmd):
    subprocess.run(cmd.replace('\n',' '), shell=True,stderr=sys.stderr, stdout=sys.stdout)

def sync():
    run('rclone sync /local/scratch/username/testdir/outputs gdrive:/0main/CVPR2022/outputs -P --transfers 40')

def compress():
    run('tar cvf /local/scratch/username/testdir/outputs.tar -C /local/scratch/username/testdir outputs')

def upload():
    import datetime
    dt = datetime.datetime.now()
    dt = dt.strftime("%Y%m%d_%H%M%S")
    run('mv /local/scratch/username/testdir/outputs.tar /local/scratch/username/testdir/outputs-%s.tar' % dt)
    run('rclone move /local/scratch/username/testdir/outputs-%s.tar gdrive:/0main/CVPR2022 -P' % dt)

def transfer():
    import datetime
    dt = datetime.datetime.now()
    dt = dt.strftime("%Y%m%d_%H%M%S")
    run('mv /local/scratch/username/testdir/outputs /local/scratch/username/testdir/outputs-%s' % dt)
    run('tar cf - -C /local/scratch/username/testdir outputs-%s | ssh -p 2222 cc@radian.dk "cd /Users/cc/Desktop && tar xvzf -"' % dt)

def prepare_exp_data():
    run('python ~/CVPR2022/prepare_Meng_exp_data.py')


def eval(cuda_idx,model,config,inp):

    print('\nNow running',config['name'],'\n')
    print('\n\n\n\tFolder:%s\n' % inp)

    if 'rbpn' in config['name']:
        if 'bbc' in config['name']:
            upscale = 2
        else:
            upscale = 1
        run('''
            cd %s; CUDA_VISIBLE_DEVICES=%d python eval.py --gpus 1
            --upscale_factor %d --data_dir '%s/inputs/%s' --nFrames 9 --threads 4
            --model_type %s --model %s --output '%s/outputs/%s/%s'
            ''' % (config['directory'],cuda_idx,upscale,rootfolder,inp,config['model_type'],
                   config['path'],rootfolder,config['name'],inp))
    else:
        run('''
            CUDA_VISIBLE_DEVICES=%d PYTHONPATH="./:${PYTHONPATH}" python
            inference/inference_options.py --task simrec --model_path %s
            --scale 2 --input '%s/inputs/%s' --output '%s/outputs/%s/%s'
            -opt %s
            ''' % (cuda_idx,config['path'],rootfolder,inp,rootfolder,config['name'],inp,config['opt'])
            )


class myThread (threading.Thread):
    def __init__(self, cuda_idx):
        threading.Thread.__init__(self)
        self.cuda_idx = cuda_idx
        self.queue = []

    def add_job(self,args):
        self.queue.append(args)

    def run(self):
        for item in tqdm.tqdm(self.queue):
            eval(self.cuda_idx,*item)

def run_threads(threads):
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# CONFIGS
rootfolder = '/local/scratch/username/testdir'
configs = [
    dict(
        name='swinir_small',
        path='experiments/train_SwinIR_MLSIM_x2/models/net_g_180000.pth',
        opt='options/train/SwinIR/train_SwinIR_MLSIM_kiiara_x2.yml'
    ),
    dict(
        name='rcan_kiiara_07',
        path='experiments/RCAN_MLSIM_archived_20211107_210155/models/net_g_latest.pth',
        opt='options/train/RCAN/train_RCAN_MLSIM.yml'
    ),
    dict(
        name='rbpn_planetearth',
        path='weights/1x_crunchy.cl.cam.ac.ukRBPNPlanetEarth_SIMdata_3k_RBPN_2_epoch_165.pth',
        opt='',
        directory='/home/username/RBPN-PyTorch',
        model_type='RBPN'
    ),
    dict(
        name='rbpn_bbc9k',
        path='weights/2x_kiiara.cl.cam.ac.ukRBPNSIMdata_9k_RBPN_epoch_5.pth',
        opt='',
        directory='/home/username/RBPN-PyTorch',
        model_type='RBPN'
    ),
    dict(
        name='swinir_rcab',
        path='experiments/train_SwinIR_MLSIM_x2/models/net_g_latest.pth',
        opt='options/train/SwinIR/SwinIR_RCAB_kiiara.yml',
    ),
    dict(
        name='rcan_nostripe9',
        path='experiments/RCAN_MLSIM_noStripes/models/net_g_latest.pth',
        opt='options/train/ablation_study/RCAN_MLSIM_noStripes.yml',
    ),
    dict(
        name='rcan_nostripe1',
        path='experiments/RCAN_MLSIM_noStripes_singleImage/models/net_g_latest.pth',
        opt='options/train/ablation_study/RCAN_MLSIM_noStripes_singleImage.yml',
    ),
    dict(
        name='rcan_div2k_sim9',
        path='experiments/RCAN_SIM_9/models/net_g_120000.pth',
        opt='options/train/MLSIM_DIV2K/RCAN_SIM_9.yml',
    ),
    dict(
        name='rcan_div2k_nostripe1',
        path='experiments/RCAN_SISR/models/net_g_130000.pth',
        opt='options/train/MLSIM_DIV2K/RCAN_SISR.yml',
    ),
]


# RUN SETTINGS
cuda_idx = [0,1,2,3]
inputFolders = [
    # 'bbc_512', # this is just a single sample from the train set
    # 'bbc_sharp',
    # 'SIMdata-3-valid',
    # 'div2k-subset',
    # 'SIMdata-div2k-valid',
    # 'Meng_converted',
    # 'Meng_converted_adapthist',
    # 'Meng_converted_strided',
    # 'SIMdata-otherDatasets-512',
    # 'SIMdata-otherDatasets-720',
    # 'SIMdata-otherDatasets-1024',
    # 'Extra_TestImages',
    'expsim',
    # 'showcase1_raw',
    # 'showcase2_raw',
    # 'showcase3_raw',
]
models = [
    # 'swinir_small',
    # 'swinir_rcab',
    # 'rcan_kiiara_07',
    # 'rbpn_planetearth',
    # 'rbpn_bbc9k',
    # 'rcan_nostripe9',
    # 'rcan_nostripe1',
    'rcan_div2k_sim9',
    # 'rcan_div2k_nostripe1'
]

args = []

# RUN

# Prep
expandedFolders = []
for folder in inputFolders:
    if 'Meng_converted' in folder:
        expandedFolders.extend(get_nested_folder_structure_for_experiments(folder))
    else:
        expandedFolders.append(folder)
inputFolders = expandedFolders


# Generate args loop
for model in models:
    config = None
    for _config in configs:
        if model == _config['name']:
            config = _config
            break
    if config is None:
        print('model config not found',model)
        continue
    for folder in inputFolders:
        if 'nostripe' not in model:
            args.append((model,config,folder))
        else:
            nostripeFolder = '%s-nostripe' % folder # assuming equivalent folder has been prepared
            args.append((model,config,nostripeFolder))

# threads
threads = []
for idx in cuda_idx:
    threads.append(myThread(idx))

# assign tasks
for arg_idx,arg in enumerate(args):
    thread_idx = arg_idx % len(threads)
    threads[thread_idx].add_job(arg)
print('%d folders to process')


# prepare_exp_data()
run('rm -rf /local/scratch/username/testdir/outputs')
run_threads(threads)
# print('compression outputs')
# compress()
# print('starting upload')
# upload()
print('starting direct transfer')
transfer()

print('finished all jobs')
