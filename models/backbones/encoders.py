import torch
import timm.models as tm


def encoders(encoder_cfg):
    _encoder_cfg = encoder_cfg.copy()
    etype = _encoder_cfg.pop('type')
    model = eval('tm.{}'.format(etype))(**_encoder_cfg)
    return model