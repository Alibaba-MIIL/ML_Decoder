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
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

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
    args.num_workers=0

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('done')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
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

    print("before next")
    iterator = iter(val_loader)
    x, target=iterator.next()
    print("after next")
    model.eval()
    output=torch.sigmoid(model(x.cuda()))
    aaa=3



if __name__ == '__main__':
    main()
