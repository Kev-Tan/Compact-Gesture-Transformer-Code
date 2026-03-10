import timm

def build_timm_backbone(config):
    print('Building timm backbone')
    print(f"Creating model with backbone: {config.backbone}")
        
    if 'vit' in config.model_name or 'deit' in config.model_name:
        model = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            num_classes=0,
            drop_rate=config.dropout2d,
            drop_path_rate=config.drop_path,
            drop_block_rate=None,
            img_size=config.input_size
        )
    else:
        try:
            model = timm.create_model(
                config.backbone,
                pretrained=config.pretrained,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=0,
                drop_rate=config.dropout2d,
                drop_path_rate=config.drop_path,
                drop_block_rate=None
            )
        except:
            model = timm.create_model(
                config.backbone,
                pretrained=config.pretrained,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=0,
                drop_rate=config.dropout2d,
                drop_path_rate=config.drop_path,
                drop_block_rate=None
            )

    return model