import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
    

        #self.ni = len(self.img_ids)
        #self.ims, self.mks = [None] * self.ni, [None] * self.ni
        #self.buffer = []
        #self.max_buffer_length = 1000
        
    def __len__(self):
        return len(self.img_ids)

    def loadtxt_safe_2d(self, path, dtype=float, delimiter=None, ncols=5):
        if not os.path.exists(path):
            return np.empty((0, ncols or 0), dtype=dtype)
            
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)  # ensure 2D if there is only one row
        if ncols is not None and arr.shape[1] != ncols:
            raise ValueError(f"Expected {ncols} columns, got {arr.shape[1]}.")
        return arr

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        #img, mask = self.ims[idx], self.mks[idx]
        #if img is None:
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        msk_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        lbl_path = img_path.replace('images', 'labels')[:-3] + 'txt'
        img = cv2.imread(img_path)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        lbl = self.loadtxt_safe_2d(lbl_path)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id, 'label': lbl}

