import torch
import torchvision

import numpy as np
import pandas as pd

import os.path
from tqdm import tqdm
from pathlib import Path

from imagenet_bg import ImageNetBG

IMAGENET_BG_PATH = "/datasets/ImageNet-BG/"
BATCH_SIZE = 512

def prepare__AlexNet():
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "AlexNet", model, preprocess

def prepare__VGG19_BN():
    model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "VGG19_BN", model, preprocess

def prepare__Inception_V3():
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "Inception_V3", model, preprocess

def prepare__ResNet152():
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "ResNet152", model, preprocess

###

def prepare__ShuffleNet_V2_X1_5():
    model = torchvision.models.shufflenet_v2_x1_5(weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "ShuffleNet_V2_X1_5", model, preprocess
    
def prepare__MobileNet_v3_L():
    model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "MobileNet_v3_L", model, preprocess

def prepare__RegNet_Y_800MF():
    model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "RegNet_Y_800MF", model, preprocess

###

def prepare__EfficientNet_B3():
    model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "EfficientNet_B3", model, preprocess
    
def prepare__ConvNeXt_L():
    model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "ConvNeXt_L", model, preprocess

def prepare__EfficientNet_V2_M():
    model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "EfficientNet_V2_M", model, preprocess

###

def prepare__ViT_L_16():
    model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "ViT_L_16", model, preprocess

def prepare__MaxVit_T():
    model = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "MaxVit_T", model, preprocess
    
def prepare__Swin_V2_B():
    model = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1)
    preprocess = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
    
    model.eval()
    model.cuda()

    return "Swin_V2_B", model, preprocess

def calculate_model_df(model, data_loader):
    all_out = []
    data_id = 0
    for i_batch, (inputs, targets, infos) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            logits = model(inputs.cuda())
            for i in range(len(inputs)):
                out = {}
                out["id"] = data_id
                out["label"] = targets[i].item()
                out["quality"] = infos[0][i]
                out["type"] = infos[1][i]
                out["background"] = infos[2][i]
                out["background_real_type"] = infos[3][i]
                out["logits"] = np.array(logits[i].detach().cpu())
                all_out.append(out)
                data_id += 1
    
    df = pd.DataFrame(all_out)
    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")

    return df

model_funs = [
    prepare__AlexNet,
    prepare__VGG19_BN,
    prepare__Inception_V3,
    prepare__ResNet152,
###
    prepare__ShuffleNet_V2_X1_5,
    prepare__MobileNet_v3_L,
    prepare__RegNet_Y_800MF,
###    
    prepare__EfficientNet_B3,
    prepare__ConvNeXt_L,
    prepare__EfficientNet_V2_M,
###   
    prepare__ViT_L_16,
    prepare__MaxVit_T,
    prepare__Swin_V2_B
]

Path("./model_results").mkdir(parents=True, exist_ok=True)    

for model_fun in model_funs:
    model_name, model, preprocess = model_fun()
    print(f"> {model_name}")
    if os.path.isfile(f"./model_results/{model_name}.pickle"):
        continue

    dataset = ImageNetBG(
        root_dir=IMAGENET_BG_PATH,
        labels_file='./imagenet_class_index.json',
        transform=preprocess
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    df = calculate_model_df(model, data_loader)
    df.to_pickle(f"./model_results/{model_name}.pickle")    