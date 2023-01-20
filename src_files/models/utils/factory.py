import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args,load_head=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    ####################################################################################
    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if 'model' in state:
            key = 'model'
        else:
            key = 'state_dict'
        if not load_head:
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state[key], strict=True)

    return model
