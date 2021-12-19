import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from src_files.models import create_model
import matplotlib

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')

parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--input-size', type=int, default=448)
parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--th', type=float, default=0.4)

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

# --model-path=D:/models/tresnet_l_git_448_90_0.pth

# --model-path=D:/models/tresnet_xl_git_640_91_4.pth
# --model-name=tresnet_xl
# --input-size=640

def main():
    print('Inference code on a single image')

    # parsing args
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    print('done')
    model.eval()
    state = torch.load(args.model_path, map_location='cpu')
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    detected_classes = classes_list[np_output > args.th]
    print('done\n')

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("detected classes: {}".format(detected_classes))

    plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
