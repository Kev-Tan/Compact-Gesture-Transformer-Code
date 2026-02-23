class TransformerConfig():
    def __init__(self, args,
                 model_name: str = "GestureTransformer",  # Architecture name
                 backbone: str = "resnet50",               # timm model name
                 in_planes: int = 3,
                 out_planes: int = 10,
                 pretrained: bool = True,
                 dropout_backbone: float = 0.1,
                 dropout2d: float = 0.1,
                 drop_path: float = 0.1,
                 input_size: int = 224,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 attention_dropout: float = 0.1,
                 ):
        
        self.model_name = args.model_name
        self.backbone = args.backbone
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pretrained = pretrained
        self.dropout_backbone = dropout_backbone
        self.dropout2d = dropout2d
        self.drop_path = drop_path
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout

# Usage:
# config = TransformerConfig()
# config = TransformerConfig(model_name="deit_base_patch16_224", out_planes=20)