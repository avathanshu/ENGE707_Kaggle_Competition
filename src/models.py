"""Model construction: a small CNN trained from scratch, and pretrained timm backbones."""
import timm
import torch.nn as nn

from .config import CLASSES

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class SimpleCNN(nn.Module):
    """Baseline: 4 blocks of (conv-BN-ReLU x2, max-pool), global average pooling, dropout, linear."""

    def __init__(self, num_classes=len(CLASSES), width=32, dropout=0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))

    def get_classifier(self):  # same method name as timm models (used for the head learning rate)
        return self.head


def create_model(cfg, pretrained=None, num_classes=len(CLASSES)):
    """Build the model named in cfg. pretrained=None means 'use cfg.pretrained'."""
    if cfg.model_name == "simple_cnn":
        return SimpleCNN(num_classes, width=cfg.cnn_width, dropout=cfg.drop_rate)
    kwargs = dict(drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path_rate)
    if "vit" in cfg.model_name:
        kwargs["img_size"] = cfg.img_size  # ViTs need the input size to size their position embeddings
    model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained if pretrained is None else pretrained,
                              num_classes=num_classes, **kwargs)
    head = model.get_classifier()
    if isinstance(head, nn.Linear):  # new 8-class head starts at uniform predictions (some timm inits give huge logits)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return model


def get_mean_std(model):
    """Normalisation constants the pretrained backbone was trained with (ImageNet for the CNN)."""
    if isinstance(model, SimpleCNN):
        return IMAGENET_MEAN, IMAGENET_STD
    data_cfg = timm.data.resolve_data_config({}, model=model)
    return data_cfg["mean"], data_cfg["std"]


def count_params(model):
    return sum(p.numel() for p in model.parameters())
