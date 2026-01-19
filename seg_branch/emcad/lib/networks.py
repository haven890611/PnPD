import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from scipy import ndimage

from lib.pvtv2 import pvt_v2_b2, pvt_v2_b0
from lib.decoders import CASCADE
#from lib.cnn_vit_backbone import Transformer, SegmentationHead

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class PVT_CASCADE(nn.Module):
    def __init__(self, n_class=1, encoder='pvt_v2_b2'):
        super(PVT_CASCADE, self).__init__()
        self.n_class = n_class
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        print(encoder)
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
            path = '/home/P111yhchen/medical_image_segmentation/pretrained/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        else:
            self.backbone = pvt_v2_b0()  # [64, 128, 320, 512]
            path = '/home/P111yhchen/medical_image_segmentation/pretrained/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = CASCADE(channels=channels)
        print('Model %s created, param count: %d' %
                     ('decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        #self.out_head1 = nn.Conv2d(32, n_class, 1)
        #self.out_head2 = nn.Conv2d(32, n_class, 1)
        #self.out_head3 = nn.Conv2d(32, n_class, 1)
        self.out_head4 = nn.Conv2d(channels[-1], n_class, 1)

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o, x4_oo = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
        #p1 = self.out_head1(x1_o)
        #p2 = self.out_head2(x2_o)
        #p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_oo)
        
        #p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        #p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        #p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')  
        #return x1_o, x2_o, x3_o, p4
        return x1_o, x2_o, x3_o, x4_oo, p4

        
if __name__ == '__main__':
    model = PVT_CASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())

