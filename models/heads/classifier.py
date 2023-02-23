import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import encoders


class Classifier(nn.Module):
    """General classification network.

    Args:
        Args:
        encoder (dict): the config dict of encoder network.
        test_cfg (dict): the config dict of testing setting.
        train_cfg (dict): the config dict of training setting, including
            some hyperparameters of loss.

    """
    def __init__(self,
                 encoder,
                 test_cfg=None,
                 train_cfg=None):
        super(Classifier, self).__init__()
        assert isinstance(encoder, dict)
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.return_label = self.test_cfg.pop('return_label', True)
        self.return_feature = self.test_cfg.pop('return_feature', False)

        self.encoder = encoders(encoder)

        self.cls_loss = nn.CrossEntropyLoss()

    def _get_losses(self, feats, label):
        """calculate training losses"""
        loss_cls = self.cls_loss(feats, label[:, 0]).unsqueeze(0) * self.train_cfg['w_cls']
        loss = loss_cls
        return dict(loss_cls=loss_cls, loss=loss)

    def forward(self, img, label=None, domain=None):
        """forward"""
        feat = self.encoder(img) # [bz, num_classes], logits
        if self.training:
            return self._get_losses(feat, label)
        else:
            pred = F.softmax(feat, dim=1)[:, 0]
            output = [pred]
            if self.return_label:
                output.append(label)
            if self.return_feature:
                raise NotImplementedError
            return output

