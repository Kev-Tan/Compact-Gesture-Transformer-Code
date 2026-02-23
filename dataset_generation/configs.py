class TransformerConfig():
    def __init__(self, args,
                 model_name: str = "GestureTransformer",
                 backbone: str = "resnet18",          # matches JSON
                 in_planes: int = 3,
                 out_planes: int = 12,              # n_classes
                 pretrained: bool = True,
                 dropout_backbone: float = 0.1,
                 dropout2d: float = 0.1,            # from JSON
                 drop_path: float = 0.1,
                 input_size: int = 224,
                 num_heads: int = 8,                # n_head
                 hidden_dim: int = 512,
                 ff_size: int = 1024,               # feedforward size
                 attention_dropout: float = 0.5,    # dropout1d
                 n_module: int = 6                  # transformer layers
                 ):
        
        self.model_name = model_name
        self.backbone = backbone
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pretrained = pretrained
        self.dropout_backbone = dropout_backbone
        self.dropout2d = dropout2d
        self.drop_path = drop_path
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ff_size = ff_size
        self.attention_dropout = attention_dropout
        self.n_module = n_module