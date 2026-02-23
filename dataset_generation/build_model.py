from transformer import _GestureTransformer
import torch
import torch.nn as nn
from configs import TransformerConfig
import argparse

def build_model(args):
    if args.model_name == "transformer":
        model = Transformer(args)
        
def main():
    # What are the args i should put here?
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--backbone', type=str, required=True,
                        help='backbone')
    args = parser.parse_args()
    build_model(args)
    return

class Transformer(nn.Module):
    # We move the forward function here?
    def __init__(self, args):
        super(Transformer, self).__init__()
        cfg = TransformerConfig(args)
        model = _GestureTransformer(cfg)
        print(model)

if __name__ == "__main__":
    main()    