import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from matplotlib import pyplot as plt

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_fsrcnn_x{}.'.format(args.scale)))

    plt.figure(figsize=(10, 10))
    first_layer = model.first_part
    last_layer = model.last_part
    feature_map1 = first_layer.__getitem__(0).weight.cpu().clone()
    feature_map2 = last_layer.__getitem__(0).weight.cpu().clone()
    print("number of the first layer: ", len(feature_map1))
    print("number of the last layer: ", len(feature_map2))
    for i in range(0, len(feature_map1)):
        map1 = feature_map1[i]
        plt.subplot(8, 7, i + 1)
        plt.axis('off')
        plt.imshow(map1[0, :, :].detach(), cmap='gray')
    plt.savefig('./data/visualization-nores/fist_layer.png')
    plt.figure(figsize=(10, 10))
    for i in range(0, len(feature_map2)):
        map2 = feature_map2[i]
        plt.subplot(8, 7, i + 1)
        plt.axis('off')
        plt.imshow(map2[0, :, :].detach(), cmap='gray')
    plt.savefig('./data/visualization-nores/last_layer.png')
