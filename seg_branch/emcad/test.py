import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.networks import PVT_CASCADE
from utils.dataloader import get_loader, test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/nfs/P111yhchen/code/detection/seg_branch/runs/emcad_b0/best.pth')
    opt = parser.parse_args()
    model = PVT_CASCADE(n_class=3, encoder='pvt_v2_b0')
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    image_root = '/local_data/dataset/polyp/detection/patients_complete/images/val/'
    gt_root = '/local_data/dataset/polyp/detection/patients_complete/masks/val/'
    save_path = '/nfs/P111yhchen/code/detection/seg_branch/runs/emcad_b2/results/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    mean_acc = []      
    for ci in range(nc-1):
        acc = confusion_matrix[ci, ci] / np.sum(confusion_matrix[ci])
        mean_acc.append(acc)
    print(confusion_matrix)
    print(mean_acc)

