import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, \
    add_weight_decay, get_datasets_from_csv, update_wordvecs
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import pickle

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

def main():
    args = parser.parse_args()
    #NUS-WIDE defaults
    args.zsl = 1
    args.num_of_groups = 925
    args.use_ml_decoder = 1
    args.num_classes = 925

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('done')

    #NUS-WIDE Data loading
    json_path = os.path.join(args.data, 'benchmark_81_v0.json')
    wordvec_array = torch.load(os.path.join(args.data, 'wordvec_array.pth'))
    train_transform = transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ])
    val_transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ])
    train_dataset, val_dataset, train_cls_ids, test_cls_ids = \
        get_datasets_from_csv(args.data,
                              args.data,
                              train_transform, val_transform,
                              json_path)
    train_wordvecs = wordvec_array[..., train_cls_ids].float()
    test_wordvecs = wordvec_array[..., test_cls_ids].float()
    print('classes {}'.format(len(train_dataset.classes)))
    print('train_cls_ids {} test_cls_ids {} '.format(train_cls_ids.shape, test_cls_ids.shape))

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_zsl(model, train_loader, val_loader, args.lr, train_wordvecs, test_wordvecs)


def train_multi_label_zsl(model, train_loader, val_loader, lr, train_wordvecs=None,
                          test_wordvecs=None):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        update_wordvecs(model, train_wordvecs)
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()
        update_wordvecs(model, test_wordvecs=test_wordvecs)
        update_wordvecs(ema.module, test_wordvecs=test_wordvecs)

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
