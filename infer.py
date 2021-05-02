import argparse
import os
import numpy as np
from modeling.deeplab import *
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


if __name__=='__main__':

    device = 'cpu'
    num_classes = 4

    # Define network
    model = DeepLab(num_classes=num_classes,      # self.nclass
                    backbone='resnet',  # args.backbone
                    output_stride=16,   # args.out_stride
                    sync_bn=None,       # args.sync_bn
                    freeze_bn=False)    # args.freeze_bn

    model_path = './models/best.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    with torch.no_grad():
        img_path = '/home/hzw/Desktop/CRT/20.png'
        image = Image.open(img_path).convert('RGB')
        image1 = test_transform(image).to(device).unsqueeze(0)

        output = model(image1)
        output = torch.squeeze(output)
        # pred = output.data.cpu().numpy()
        # pr = np.argmax(pred, axis=1)
        pr = F.softmax(output.permute(1,2,0), dim = -1).cpu().numpy().argmax(axis=-1)


        colors = [(0, 0, 0), (128, 0, 0), (192, 0, 128), (128, 64, 12), (64, 0, 128), (192, 0, 128)]
        # ------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        # ------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image_mask = Image.fromarray(np.uint8(seg_img)).resize((image.size[0], image.size[1]), Image.NEAREST)
        image_mask.show()

        # ------------------------------------------------#
        #   将新图片和原图片混合
        # ------------------------------------------------#
        image_mix = Image.blend(image, image_mask, 0.4)
        image_mix.show()

        val_list = []
        c = [1, 2, 3]
        j_min = [5000, 5000, 5000]
        x_min = [5000, 5000, 5000]
        for i in range(pr.shape[0]):
            for j in range(pr.shape[1]):
                val = pr[i][j]

                if val != 0 and val not in val_list:
                    val_list.append(pr[i][j])

                if val in c:
                    if val == 1 and j < j_min[0]:
                        j_min[0] = j
                        x_min[0] = i
                    if val == 2 and j < j_min[1]:
                        j_min[1] = j
                        x_min[1] = i
                    if val == 3 and j < j_min[2]:
                        j_min[2] = j
                        x_min[2] = i

        label1 = j_min[0]
        label2 = j_min[1]

        image.show()
    print('---')

