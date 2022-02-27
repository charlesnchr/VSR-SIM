from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from rcan import RCAN
from data import get_preproc_training_set, get_training_set, get_eval_set
import pdb
import socket
import time
import torchvision.transforms as transforms
import wandb


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--norm', default=None, help='type of normalisation')


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


wandb.init(project="phd")
wandb.config.update(opt) 

toPIL = transforms.ToPILImage()    


def train(epoch):
    epoch_loss = 0
    running_avg_loss = 0

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        # print('SIZES',input.shape,target.shape,neigbor[0].shape,flow[0].shape)
        # print('input pytorch',flow[0].min(),flow[0].max(),neigbor[0].min(),neigbor[0].max())
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input, neigbor, flow)
        
        if opt.residual:
            prediction = prediction + bicubic
            
        loss = criterion(prediction, target)

        # for debugging
        # toPIL = transforms.ToPILImage()    
        # print('OUTPUT',prediction.min(),prediction.max(),target.min(),target.max()) 
        # toPIL(input[0]).save('input.png')
        # toPIL(neigbor[0][0]).save('neigh.png')
        # toPIL(target[0]).save('target.png')

        t1 = time.time()
        epoch_loss += loss.data
        running_avg_loss += loss.data
        loss.backward()
        optimizer.step()

        if iteration % 50 == 0:
            lr = toPIL(input[0].cpu())
            bc = torch.clone(input[0].cpu())
            for n in neigbor:
                bc += n[0].cpu()
            bc = bc / (1+len(neigbor))

            # torch.save((input,neigbor,target,prediction,bicubic,flow),'sample%d.pt' % iteration)
            
            bc = toPIL(bc)
            hr = toPIL(target[0].cpu())
            sr = toPIL(prediction[0].clamp(0,1).detach().cpu())
            
            imgs = [('lr',lr),('bc',bc),('sr',sr),('hr',hr)]
            wandb.log({'lr':optimizer.param_groups[0]['lr'],'avg_loss':running_avg_loss / 50,'batch_idx':iteration,'epoch':epoch,'valid_imgs_%d' % iteration: [wandb.Image(im, caption=ca) for ca,im in imgs]})
            running_avg_loss = 0

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    #wandb.log({'epoch':epoch,'avg_loss': epoch_loss / len(training_data_loader)},step=epoch)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_eval_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=opt.num_channels, base_filter=96,  feat = 48, num_stages=3, n_resblock=3, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 
elif opt.model_type == 'RCAN':
    model = RCAN(nch_in=opt.nFrames,nch_out=opt.num_channels,n_resgroups=3,n_resblocks=10,n_feats=64, scale=opt.upscale_factor, norm=opt.norm)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
#print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

print('Moving model to GPU')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

print('Creating optimizer')
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

wandb.watch(model, log_freq=100)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):

    train(epoch)
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
                
            
    if epoch % opt.snapshots == 0:
        checkpoint(epoch)
