import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def main():
    print('Inference code on a single image')

    # parsing args
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')


    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.image_size, args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()


    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    idx_th = scores > args.th
    detected_classes = detected_classes[idx_th]
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
