import os
import argparse
from matplotlib import cm
from PIL import Image
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.helper_functions.visualization import plot_activation
from src_files.models import create_model

import cv2
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch-size', default=56, type=int,
                    metavar='N', help='mini-batch size')

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> Image.Image:

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def main():
    args = parser.parse_args()
    args.model_name='tresnet_m'
    args.model_path = 'C:/git/github_miil/ML_Decoder/pretrained_models/tresnet_m_COCO_224_84_2.pth'
    args.image_size=224
    ##
    # args.model_name='tresnet_l'
    # args.model_path = 'C:/git/github_miil/ML_Decoder/pretrained_models/tresnet_l_COCO__448_90_0.pth'
    # args.image_size=448
    ##
    # args.model_name='tresnet_xl'
    # args.model_path = 'C:/git/github_miil/ML_Decoder/pretrained_models/tresnet_xl_COCO_640_91_4.pth'
    # args.image_size=640

    args.batch_size=1
    args.workers=0

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args,load_head=True).cuda()
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().eval()
    ########### eliminate BN for faster inference ###########
    print('done')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path_val = args.data
    # data_path_val = f'{args.data}/val2014'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    print("len(val_dataset)): ", len(val_dataset))



    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    iterator = iter(val_loader)
    for num_pics in range(5):
        x, target=iterator.next()
        target=target.max(dim=1)[0].cuda()
        model.eval()
        with torch.no_grad():
            output=torch.sigmoid(model(x.cuda()))

        # getting maps
        attn_output_weights=model.head.decoder.layers[0].attn_output_weights
        maps=attn_output_weights[0]
        tgts=target[0]
        indices=torch.nonzero(tgts, as_tuple=True)[0]
        spatial_res=int(args.image_size/32)
        relevant_maps=torch.index_select(maps, 0, indices).view(-1,spatial_res,spatial_res)


        for ind in range(indices.shape[0]): # looping over gt classes
            # get class name
            id_class=indices[ind].item()
            aaa=val_dataset.coco.dataset['categories']
            class_name=aaa[id_class]['name']

            # get inference score
            score = output[0][id_class].item()

            # get relevent map
            mask_np=relevant_maps[ind,:,:].cpu().numpy()

            #plot
            plot_activation(mask_np, x, args.image_size, score, class_name)
            plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
