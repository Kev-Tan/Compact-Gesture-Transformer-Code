import torch.nn as nn
import timm
from .timm_backbone import build_timm_backbone
from .attention import EncoderSelfAttention


class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, config, **kwargs):
        super(_GestureTransformer, self).__init__()
        print("Printing")
        
        self.in_planes = config.in_planes
        
        # Build backbone based on config
        # if config.backbone == "timm":
            # hyperparams = {
            #     "pretrained": config.pretrained,
            #     "dropout2d": config.dropout2d,
            #     "drop_path": config.drop_path,
            #     "input_size": config.input_size,
            # }
        self.backbone = build_timm_backbone(config)
        print("Finish building timm backbone")
        # else:
            # For other backbone types (resnet, vgg, etc.)
            # self.backbone = config.backbone(config.pretrained, config.in_planes, 
            #                                dropout=config.dropout_backbone)


        self.self_attention = EncoderSelfAttention(
            d_model = 512, 
            d_k = 64,
            d_v = 64,
            n_head = config.num_heads,
            n_module = config.n_module,
            dff = config.ff_size,
            dropout_transformer = config.attention_dropout,
            **kwargs
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        self.classifier = nn.Linear(512, config.out_planes)

    def forward(self, x):
        shape = x.shape

        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])
        x = self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)
        x = self.self_attention(x)
        x = self.pool(x).squeeze(dim=1)
        x = self.classifier(x)
        return x