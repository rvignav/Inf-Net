# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import imageio
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
import glob
import sys
from PIL import Image

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int,
                        default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/NCP/NCP-1/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./Snapshots/save_weights/Inf-Net/Inf-Net-100.pth',
                        help='Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`')
    parser.add_argument('--save_path', type=str, default='./Results/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@gamil.com)\n----\n".format(opt), "#" * 20)

    model = Network()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
    model.load_state_dict(torch.load(
        opt.pth_path, map_location=torch.device('cpu')))
    # model.cuda()
    model.eval()

    subs = []
    folders = glob.glob(opt.data_path + '*')
    for folder in folders:
        s = glob.glob(folder + '/*')
        for item in s:
            subs.append(item + '/')

    count = 0
    for image_root in subs:
        test_loader = test_dataset(image_root, opt.testsize)
        os.makedirs(opt.save_path, exist_ok=True)

        path = opt.save_path + 'scans/Volume' + str(count+42)
        if not os.path.exists(path):
            os.makedirs(path)
            images = glob.glob(image_root + '/*')
            for i in range(len(images)):
                im = Image.open(images[i])
                im.save(path + '/' + str(i) + '.jpg')

        for i in range(test_loader.size):
            image, name = test_loader.load_data()
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

            res = lateral_map_2
            # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            if not os.path.exists(opt.save_path + 'masks/Volume' + str(count+42)):
                os.makedirs(opt.save_path + 'masks/Volume' + str(count+42))
            imageio.imwrite(opt.save_path + 'masks/Volume' + str(count+42) + '/' + name, res)
        
        count += 1
        
    print('Test Done!')


if __name__ == "__main__":
    inference()