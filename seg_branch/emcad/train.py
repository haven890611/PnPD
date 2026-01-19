import os
import numpy as np
import argparse
from datetime import datetime
import logging
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss

import matplotlib.pyplot as plt

from lib.networks import PVT_CASCADE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.utils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
        
# def structure_loss(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)

#     return (wbce + wiou).mean()


def test(model, path, dataset):
    model.eval()
    nc = model.n_class
    image_root = path + dataset + '/'
    gt_root = image_root.replace('images', 'masks')
    nc = model.n_class
    confusion_matrix = np.zeros((nc, nc))
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    for i in range(num1):
        image, gt, label, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        H, W = gt.shape
        image = image.cuda()
        results = model(image)
        res = F.upsample(results[-1], size=(H, W), mode='bilinear', align_corners=False)
        res = res.softmax(dim=1).squeeze()

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

def train(train_loader, model, optimizer, epoch, test_path, model_name = 'PVT-CASCADE'):
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(model.n_class)
    
    model.train()
    global best
    #size_rates = [0.75, 1, 1.25] 
    size_rates = [1] 
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='nearest', align_corners=True)
            # ---- forward ----
            results= model(images)
            # print(results[-1].shape, gts.shape)
            # ---- loss function ----
            # loss_P4 = structure_loss(results[-1], gts)
            loss_ce4 = ce_loss(results[-1], gts.long())
            loss_dc4 = dice_loss(results[-1], gts, softmax=True)
            loss = 0.3*loss_ce4 + 0.7*loss_dc4
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path,'last.pth'))

    # choose the best model
    # global dict_plot
    test1path = '/local_data/dataset/polyp/detection/patients_complete/images/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['val']:
            mean_acc = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, acc: {}'.format(epoch, dataset, mean_acc))
            print(dataset, ': ', mean_acc)
        
        acc = sum(mean_acc)/len(mean_acc)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(save_path,'best.pth'))
            print('############## best average acc:', best)
            logging.info('##############################################################################best:{}'.format(best))

    
if __name__ == '__main__':
    dict_plot = {'val':[]}
    name = ['val']
    ##################model_name#############################
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/local_data/dataset/polyp/detection/patients_complete/images/train/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/local_data/dataset/polyp/detection/patients_complete/images/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/nfs/P111yhchen/code/detection/seg_branch/runs/')

    parser.add_argument('--device', type=int,
                        default=0)
    
    parser.add_argument('--name', type=str,
                        default='baseline')

    parser.add_argument('--encoder', type=str,
                        default='pvt_v2_b2')

    parser.add_argument('--percentage', type=float,
                        default=1., help='percentage of training')
    
    opt = parser.parse_args()
    
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    opt.train_save = os.path.join(opt.train_save, opt.name)
    # ---- build models ----
    torch.cuda.set_device(opt.device)  # set your gpu device
    model = PVT_CASCADE(n_class=3, encoder=opt.encoder)
    model.cuda()
	
    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = opt.train_path
    gt_root = opt.train_path.replace('images', 'masks')

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size,
                              augmentation=opt.augmentation, percentage=opt.percentage)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, model_name = opt.name)
    
