import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.models import create_model

import cv2
import matplotlib

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

def main():
    args = parser.parse_args()
    args.model_name='tresnet_m'
    args.model_path = 'C:/git/github_miil/ML_Decoder/pretrained_models/tresnet_m_COCO_224_84_2.pth'
    args.image_size=224
    args.batch_size=1
    args.workers=0

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
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
        relevant_maps=torch.index_select(maps, 0, indices).view(-1,7,7)


        im=np.transpose(x[0].cpu().float().numpy(),(1, 2, 0))
        for ind in range(relevant_maps.shape[0]):

            # get class name
            id_class=indices[ind].item()
            aaa=val_dataset.coco.dataset['categories']
            class_name=aaa[id_class]['name']

            # get score
            score = output[0][id_class].item()

            mask_np=relevant_maps[ind,:,:].cpu().numpy()
            # mask_resize = cv2.resize(mask_np / mask_np.max(), (args.image_size, args.image_size))
            mask_resize = cv2.resize(mask_np , (args.image_size, args.image_size))
            mask_resize=mask_resize-mask_resize.min()
            mask_resize=mask_resize/mask_resize.max()
            mask_resize=mask_resize*mask_resize # just for sharper visualization
            mask_resize=np.expand_dims(mask_resize, axis=2)
            # plt.figure(ind)
            plt.subplot(1,3,1)
            plt.imshow(im)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(mask_resize)
            plt.axis('off')
            plt.title('score {:.2f}'.format(score))
            plt.subplot(1, 3, 3)
            plt.imshow(im*mask_resize)
            plt.axis('off')
            plt.title(class_name)
            plt.waitforbuttonpress()
        aaa=3



if __name__ == '__main__':
    main()
