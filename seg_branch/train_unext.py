import argparse
import os
from collections import OrderedDict
from glob import glob
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from archs import UNext


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="UNeXt",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=352, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=352, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='CEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    parser.add_argument('--ratio', default=0.3, type=float)
    
    # dataset
    parser.add_argument('--dir', default='/local_data/dataset/polyp/detection/')
    parser.add_argument('--dataset', default='patients_complete',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--mode', default='train', type=str)
    config = parser.parse_args()

    return config

# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss_ce': AverageMeter(),
                  'loss_dc': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss_ce, loss_dc = criterion(output, target)
        loss = config['ratio']*loss_ce + (1-config['ratio'])*loss_dc
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss_ce'].update(loss_ce.item(), input.size(0))
        avg_meters['loss_dc'].update(loss_dc.item(), input.size(0))

        postfix = OrderedDict([
            ('loss_ce', avg_meters['loss_ce'].avg),
            ('loss_dc', avg_meters['loss_dc'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss_ce', avg_meters['loss_ce'].avg),
                        ('loss_dc', avg_meters['loss_dc'].avg)])


def validate(config, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    H, W = config['input_h'], config['input_w']
    with torch.no_grad():
        nc = config['num_classes']
        confusion_matrix = np.zeros((nc, nc))

        for input, target, data in val_loader:
            input = input.cuda()
            output = model(input)
            labels = data['label']
            for i in range(input.shape[0]):
                res = output[i:i+1].softmax(dim=1).squeeze()
                label = labels[i]
                for lbl in label:
                    cls, xx, yy, ww, hh = lbl
                    cls = int(cls)
                    ll = int(W*(xx - ww/2))
                    bb = int(H*(yy + hh/2))
                    rr = int(W*(xx + ww/2))
                    uu = int(H*(yy - hh/2))
                    vote = res[:-1, uu:bb, ll:rr].sum(dim=(1,2))
                    cls_pd = vote.argmax(dim=0).item()
                    cls_pd = nc-1 if vote[cls_pd].data==0 else cls_pd
                    confusion_matrix[cls, cls_pd] += 1
        print(confusion_matrix)
        mean_acc = []
        for ci in range(nc-1):
            acc = confusion_matrix[ci, ci] / np.sum(confusion_matrix[ci])
            mean_acc.append(acc)

        return mean_acc

            
def collate_fn( batch):
    """
    batch: list of (img, mask, data) from __getitem__
    - img:  np.ndarray (C,H,W), float32
    - mask: np.ndarray (H,W), uint8/int
    - data: {'img_id': str, 'label': np.ndarray with shape (K_i, C_box) or (0, C_box)}
    """
    imgs, masks, datas = zip(*batch)            # tuples of length B
    # stack fixed-size tensors
    imgs  = torch.from_numpy(np.stack(imgs,  axis=0))     # (B,C,H,W)
    masks = torch.from_numpy(np.stack(masks, axis=0))     # (B,H,W)
    # keep variable-length labels as lists
    img_ids = [d['img_id'] for d in datas]
    labels  = [d['label']   for d in datas]               # list of (K_i, C_box) arrays

    # (optional) convert mask dtype
    masks = masks.long()  # if you use CE/Dice expecting class ids

    data = {'img_id': img_ids, 'label': labels}
    return imgs, masks, data

def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('/nfs/P111yhchen/code/detection/seg_branch/runs/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('/nfs/P111yhchen/code/detection/seg_branch/runs/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']](n_classes=config['num_classes']).cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    if config['mode'] == 'val':                       
        model.load_state_dict(torch.load('/nfs/P111yhchen/code/detection/seg_branch/runs/%s/best.pth' %
                                        config['name']))
        model.eval()

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_img_ids = glob(os.path.join(config['dir'], config['dataset'], 'images', 'train', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join(config['dir'], config['dataset'], 'images', 'val', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    #train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['dir'], config['dataset'], 'images', 'train'),
        mask_dir=os.path.join(config['dir'], config['dataset'], 'masks', 'train'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['dir'], config['dataset'], 'images', 'val'),
        mask_dir=os.path.join(config['dir'], config['dataset'],  'masks', 'val'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        collate_fn=collate_fn)
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss_ce', []),
        ('loss_dc', []),
        ('val_acc', []),
    ])

    best = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        if config['mode'] == 'val':
            val_acc = validate(config, val_loader, model, criterion)
            val_mean_acc = sum(val_acc)/len(val_acc)
            break
        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_acc = validate(config, val_loader, model, criterion)
        val_mean_acc = sum(val_acc)/len(val_acc)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        # elif config['scheduler'] == 'ReduceLROnPlateau':
        #     scheduler.step(val_log['loss'])

        print('loss_ce %.4f - loss_dc %.4f '
              % (train_log['loss_ce'], train_log['loss_dc']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss_ce'].append(train_log['loss_ce'])
        log['loss_dc'].append(train_log['loss_dc'])
        log['val_acc'].append(val_mean_acc)
        pd.DataFrame(log).to_csv('/nfs/P111yhchen/code/detection/seg_branch/runs/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_mean_acc > best:
            torch.save(model.state_dict(), '/nfs/P111yhchen/code/detection/seg_branch/runs/%s/best.pth' %
                       config['name'])
            best = val_mean_acc
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
