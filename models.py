import torch
import torch.nn as nn
import timm

def get_convnext_base():
    model = timm.create_model('convnext_base', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model

def get_convnext_large():
    model = timm.create_model('convnext_large', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model

def get_mobileNetV3_large():
    model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model

def get_resnet50_32x4d():
    model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model

def get_swin_large():
    model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model

def get_deit3_large():
    model = timm.create_model('deit3_large_patch16_224', pretrained=True, num_classes=25)
    model = model.to('cuda:1')

    return model
