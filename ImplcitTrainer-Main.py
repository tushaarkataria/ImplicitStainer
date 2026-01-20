""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""
from comet_ml import Experiment
import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import math
import datasets
import models
import utils
from generative.losses import PatchAdversarialLoss, PerceptualLoss
#from test import eval_psnr
from skimage import io
from test_mine import eval_psnr_mine,eval_psnr

import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse
import glob
import sys
import numpy as np
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psnr_and_ssim_paths(result_path1,result_path2):
    PSNR = 0
    ssim = 0
    l2   = 0
    count = 0
    for i in tqdm(os.listdir(os.path.join(result_path1))):
        fake = cv.imread(os.path.join(result_path1,i))
        real = cv.imread(os.path.join(result_path2,i))
        l2   = l2  + np.sum((fake - real) ** 2)/(255*255)
        PSNR = PSNR + peak_signal_noise_ratio(fake, real) 
        SSIM =0
        for channel in range(3):
            SSIM =SSIM+ structural_similarity(fake[:,:,channel], real[:,:,channel])
        SSIM = SSIM/3 
        ssim= ssim + SSIM
        count = count+1
    average_psnr=PSNR/count
    average_ssim=ssim/count
    average_l2 = l2/count
    return average_psnr, average_ssim, average_l2


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training(args):
    if args.resume:
        sv_file = torch.load(os.path.join(args.path,'epoch-last.pth'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(args,train_loader, model, optimizer):
    model.train()
    loss_fn = nn.SmoothL1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").cuda()
    perceptual_loss1 = PerceptualLoss(spatial_dims=2, network_type="vgg").cuda()
    perceptual_loss2 = PerceptualLoss(spatial_dims=2, network_type="resnet50").cuda()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        batch_size,_,_,_ = batch['inp'].shape
        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        pred = pred * gt_div + gt_sub

        gt_image = batch['gt'].view(batch_size,256,256,3).permute(0, 3, 1, 2).contiguous()
        predicted_image = pred.view(batch_size,256,256,3).permute(0, 3, 1, 2).contiguous()
        
        p_loss = perceptual_loss(predicted_image.float(), gt_image.float())
        p_loss1 = perceptual_loss1(predicted_image.float(), gt_image.float())
        p_loss2 = perceptual_loss2(predicted_image.float(), gt_image.float())
        
        loss = loss + args.lambda_p * p_loss + args.lambda_p1 * p_loss1 + args.lambda_p2 *p_loss2
        
        train_loss.add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = None; loss = None

    return train_loss.item()


def main(args,config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(args)
    #print(model)
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(args,train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

    print("***** Training Finished*****")
    print("**** Running Inference********")

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    save_path_results = os.path.join(save_path,'Results'+str(args.epoch))
    if not os.path.exists(save_path_results):
        os.makedirs(save_path_results)
    
    test_loader = DataLoader(dataset, batch_size=1,num_workers=4, pin_memory=True)

    model = model.cuda()
    print("**** Running EVAL********")
    res = eval_psnr_mine(test_loader, model,
        save_path_results,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=None,
        verbose=True)

    print('result: {:.4f}'.format(res))

    result_path1 = save_path_results
    result_path2 = config['test_dataset']['dataset']['args']['root_path_2']
    psnr_index, ssim_index, average_l2 =  psnr_and_ssim_paths(result_path1,result_path2)
    print("PSNR, SSIM, L2 ",psnr_index, ssim_index, average_l2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config' , default='configs/train-he2ihc/train_he_to_ihc_liif.yaml')
    parser.add_argument('--name'   , default=None)
    parser.add_argument('--tag'    , default='Simple')
    parser.add_argument('--dataset', default='HEMIT')
    parser.add_argument('--batch_size', default=8,type=int)
    parser.add_argument('--backbone', default='edsr')
    parser.add_argument('--local', default='implicitstainer')
    parser.add_argument('--dataPercentage', default='full')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--lr', default=0.0001,type=float)
    parser.add_argument('--modelType', default='normal')
    parser.add_argument('--lambda_p', default=1.0,type=float)
    parser.add_argument('--lambda_p1', default=1.0,type=float)
    parser.add_argument('--lambda_p2', default=1.0,type=float)
    parser.add_argument("--resume"     , action="store_true")
    parser.add_argument("--path"     ,default='<Saving Directory Path>',type=str)
    parser.add_argument('--epoch', default=200,type=int)
    args = parser.parse_args()



    OSname = os.uname().nodename
    save_dir = '<Saving Directory Path>'
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    OSname = os.uname().nodename
    if(args.dataset=='HEMIT'):
        # PATH for new dataset other than one listed in "configs/train-he2ihc/train_he_to_ihc_liif.yaml"
        config['train_dataset']['dataset']['args']['root_path_1'] = '<Path to HE Directory(Domain A) Training Data>'
        config['train_dataset']['dataset']['args']['root_path_2'] = '<Path to IHC Directory(Domain B) Training Data>' 
        config['val_dataset']['dataset']['args']['root_path_1']   = '<Path to HE Directory(Domain A) Validation Data>' 
        config['val_dataset']['dataset']['args']['root_path_2']   = '<Path to IHC Directory(Domain B) Validation Data>' 
        config['test_dataset']['dataset']['args']['root_path_1']  = '<Path to HE Directory(Domain A) Testing Data>' 
        config['test_dataset']['dataset']['args']['root_path_2']  = '<Path to IHC Directory(Domain B) Testing Data>' 

    save_dir = os.path.join(save_dir,args.dataset,args.backbone)
       
    config['model']['name'] = args.local
    config['train_dataset']['batch_size'] = args.batch_size
    config['epoch_max'] = args.epoch
    if(args.backbone=='edsr'):
        config['model']['args']['encoder_spec']['name']='edsr'
    elif(args.backbone=='swin-s'):
        config['model']['args']['encoder_spec']['name']='swin-s'
    elif(args.backbone=='swin'):
        config['model']['args']['encoder_spec']['name']='swin'
    elif(args.backbone=='swin-l'):
        config['model']['args']['encoder_spec']['name']='swin-l'
    elif(args.backbone=='swin-conv-parallel-add-small'):
        config['model']['args']['encoder_spec']['name']='swin-conv-parallel-add-small'
    elif(args.backbone=='swin-conv-parallel-add'):
        config['model']['args']['encoder_spec']['name']='swin-conv-parallel-add'
    elif(args.backbone=='swin-conv-parallel-add-l'):
        config['model']['args']['encoder_spec']['name']='swin-conv-parallel-add-l'
    
    if(args.dataPercentage=='tenth'):
        config['train_dataset']['dataset']['name'] = 'paired-image-folders-10'
   
    if(args.activation=='relu'):
        config['model']['args']['imnet_spec']['name']='mlp'
    if(args.activation=='prelu'):
        config['model']['args']['imnet_spec']['name']='mlp-prelu'
    if(args.activation=='elu'):
        config['model']['args']['imnet_spec']['name']='mlp-elu'

    if(args.modelType=='normal'):
        config['model']['args']['imnet_spec']['args']['hidden_list'] = [256,256,256,256]
    elif(args.modelType=='normal-large'):
        config['model']['args']['imnet_spec']['args']['hidden_list'] = [1024,1024,1024,1024]
    elif(args.modelType=='deep'):
        config['model']['args']['imnet_spec']['args']['hidden_list'] = [1024,1024,256,256,256,256]
    
    config['optimizer']['args']['lr'] = args.lr

    save_path = os.path.join(save_dir,args.dataPercentage,args.modelType,args.activation,args.local,str(args.lr),'Lambda'+str(args.lambda_p),'Lambda1'+str(args.lambda_p1),'Lambda2'+str(args.lambda_p2))
    args.path=save_path

    print(config)
    main(args,config, save_path)
