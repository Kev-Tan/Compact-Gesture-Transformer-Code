# ===== Standard Library =====
import os
import math
import json
import copy
import glob
import random
import functools
import argparse
from pathlib import Path
from urllib.request import urlopen

# ===== Third-party =====
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import timm
from timm import create_model

# ===== PyTorch =====
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import get_image_backend

# ===== Project / Local =====
import build_dataset

# ===== Env =====
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"



hyperparams = {
    # Network params
    'n_classes': 12,
    'pretrained': True,
    'n_head': 8,
    'dropout_backbone': 0.1,  # dropout2d
    'dropout_transformer': 0.5,  # dropout1d
    'dff': 1024,  # ff_size
    'n_module': 6,
    
    # Solver params (required by ModuleUtilizer)
    'solver': {
        'type': 'AdamW',  # or 'Adam', 'SGD', 'RMSProp'
        'base_lr': 0.0001,
        'weight_decay': 0.0001,
        'momentum': 0.9,  # only used for SGD
        'lr_policy': 'fixed',  # or 'step', 'multistep', 'exp', 'inv'
        'gamma': 0.1,
        'stepvalue': [50, 75]  # for multistep policy
    },
    
    # Checkpoint params (required by ModuleUtilizer)
    'checkpoints': {
        'save_policy': 'best',  # or 'all', 'early_stop'
        'save_name': 'gesture_model',
        'save_dir': './checkpoints/',
        'early_stop': 10  # patience for early stopping
    },
    
    # Other required params
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dataset': 'Briareo',
    'gpu': [0],  # GPU IDs for DataParallel
    'resume': None  # Path to checkpoint to resume from, or None
}

# Decide if we use timm_backbone or not
timm_backbone = True
model_name = "resnet18.a1_in1k"

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                  nn.Linear(dff, d_model)])

    def forward(self, queries, keys, values):
        att = self.attention(queries, keys, values)
        att = self.dropout(att)
        # att = self.layer_norm(queries + att)
        att = self.fc(att)
        att = self.dropout(att)
        return self.layer_norm(queries + att)

class EncoderSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, n_module=6):
        super(EncoderSelfAttention, self).__init__()
        self.encoder = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, n_head, dff, dropout_transformer)
                                      for _ in range(n_module)])
    def forward(self, x):
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).cuda()
        for l in self.encoder:
            in_encoder = l(in_encoder, in_encoder, in_encoder)
        return in_encoder


def build_timm_backbone(model_name=model_name, hyperparams=hyperparams):
    print('build_model args')
    print(f"Creating model: {model_name}")
        
    if 'vit' in model_name or 'deit' in model_name:
        model = timm.create_model(
            model_name,
            pretrained=hyperparams.get('pretrained', False),
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            num_classes=0,
            drop_rate=hyperparams.get('dropout2d', 0.0),
            drop_path_rate=hyperparams.get('drop_path', 0.0),
            drop_block_rate=None,
            img_size=hyperparams.get('input_size', 224)
        )
    else:
        try:
            model = timm.create_model(
                model_name,
                pretrained=hyperparams.get('pretrained', False),
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=0,
                drop_rate=hyperparams.get('dropout2d', 0.0),
                drop_path_rate=hyperparams.get('drop_path', 0.0),
                drop_block_rate=None
            )
        except:
            model = timm.create_model(
                model_name,
                pretrained=hyperparams.get('pretrained', False),
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=0,
                drop_rate=hyperparams.get('dropout2d', 0.0),
                drop_path_rate=hyperparams.get('drop_path', 0.0),
                drop_block_rate=None
            )

    return model

class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, backbone: nn.Module, in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone=0.1,
                 **kwargs):
        super(_GestureTransformer, self).__init__()
        
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.self_attention = EncoderSelfAttention(512, 64, 64, n_head= 8, dropout_transformer= 0.5, dff= 1024, n_module= 6)
        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        self.classifier = nn.Linear(512, out_planes)
        
        if(backbone=="timm"):
            self.backbone = build_timm_backbone() 
            print("Finish building backbone")
            
    def forward(self, x):
        shape = x.shape

        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])

        x = self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)

        x = self.self_attention(x)

        x = self.pool(x).squeeze(dim=1)
        x = self.classifier(x)
        return x
    
    
def build_model(backbone="timm", in_planes = 3, out_planes=12):
    model = _GestureTransformer(backbone = "timm", in_planes = in_planes, out_planes=out_planes)
    return model

def train_loop(device, train_loader, model, criterion, optimizer):
    # certain modules behave differently during train/test (dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # loss_epoch is similar to running_loss in the PyTorch tutorial
    loss_epoch = 0 
    correct = 0
    total = 0

    for idx, data in enumerate(tqdm(train_loader)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
                
        # ChatGPT generated, need clarification on what this means
        labels = labels.view(-1).long()
        
        # calculate outputs by running images through the network
        outputs = model(images)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        
        print(outputs.shape)  
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print("Correct Predictions:", (predicted == labels).sum().item())
        
        acc_iter = 100 * (predicted == labels).sum().item() / images.shape[0]
        print(f'{idx} / {len(train_loader)}, Loss: {loss}, Acc@1: {acc_iter}')

    acc = round(100 * correct / total, 2)
    return loss_epoch, acc

def test_loop(device, test_loader, model):
    # certain modules behave differently during train/test (dropout)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = round(100 * correct / total, 2)
    return acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--save_name_train', type=str, default='train_val')
    parser.add_argument('--save_name_test', type=str, default='test')
    parser.add_argument('--num_frames', type=int, default=40)
    parser.add_argument('--annotation_path', type=str, default="/home/mislab/Charlene/annotation_EgoGesture/egogestureall.json")
    args = parser.parse_args()
    
    train_loader = DataLoader(build_dataset.get_set(args, 'train'))
    test_loader = DataLoader(build_dataset.get_set(args, 'test'))
    val_loader = DataLoader(build_dataset.get_set(args, 'val'))
    
    model = build_model()
    criterion = nn.CrossEntropyLoss().to(hyperparams.get("device"))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                                          weight_decay=0.0001)
    
    device = hyperparams.get("device")

    for epoch in range(30):
        print(epoch+1)
        loss, train_acc = train_loop(device, train_loader, model, criterion, optimizer)
        test_acc = test_loop(device, test_loader, model)
        val_acc = test_loop(device, val_loader, model)
        
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        
        print(f"Epoch {epoch+1}: Loss={loss}, Train_Acc={train_acc}, Test_Acc={test_acc}, Val_Acc={val_acc}")

    writer.close()
    
    # sample = next(iter(train_loader))
    # print(sample[0])



if __name__ == '__main__':
    main()