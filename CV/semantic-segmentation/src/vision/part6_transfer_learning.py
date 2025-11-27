import logging
import os
import pdb
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.vision.part5_pspnet import PSPNet
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = get_logger()


def load_pretrained_model(args, device: torch.device):
    """Load Pytorch pre-trained PSPNet model from disk of type torch.nn.DataParallel.

    Note that `args.num_model_classes` will be size of logits output.

    Args:
        args:
        device:

    Returns:
        model
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        pretrained=False
    )

    # logger.info(model)
    if device.type == 'cuda':
        cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"=> loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")
    model = model.to(device)

    return model



def model_and_optimizer(args, model) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    This function is similar to get_model_and_optimizer in Part 3.

    Use the model trained on Camvid as the pretrained PSPNet model, change the
    output classes number to 2 (the number of classes for Kitti).
    Refer to Part 3 for optimizer initialization.

    Args:
        args: object containing specified hyperparameters
        model: pre-trained model on Camvid

    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    num_classes = 2  # Hardcode to 2 for KITTI transfer learning
    model.num_classes = num_classes
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.criterion = criterion
    
    in_dim = 2048
    if model.ppm is not None:
        reduction_dim = in_dim // len(model.ppm.features)
        fea_dim = in_dim + len(model.ppm.features) * reduction_dim
    else:
        fea_dim = in_dim
    
    model.cls = model._PSPNet__create_classifier(in_feats=fea_dim, out_feats=512, num_classes=num_classes)
    model.aux = model._PSPNet__create_classifier(in_feats=1024, out_feats=256, num_classes=num_classes)
    
    params_list = []
    params_list.append({"params": model.layer0.parameters(), "lr": args.base_lr})
    params_list.append({"params": model.layer1.parameters(), "lr": args.base_lr})
    params_list.append({"params": model.layer2.parameters(), "lr": args.base_lr})
    params_list.append({"params": model.layer3.parameters(), "lr": args.base_lr})
    params_list.append({"params": model.layer4.parameters(), "lr": args.base_lr})
    params_list.append({"params": model.cls.parameters(), "lr": args.base_lr * 10})
    params_list.append({"params": model.aux.parameters(), "lr": args.base_lr * 10})
    if model.ppm is not None:
        params_list.append({"params": model.ppm.parameters(), "lr": args.base_lr * 10})
    
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer
