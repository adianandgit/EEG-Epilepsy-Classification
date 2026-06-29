import torch
import torch.nn as nn
import sys

VT_PROJECT_PATH = './UGP'
sys.path.append(VT_PROJECT_PATH)

try:
    from UGP.models.model import base_Model
    from UGP.config_files.Epilepsy_Configs import Config as BaseConfig
except ImportError:
    print(f"Import error: Unable to import from {VT_PROJECT_PATH}. Ensure the 'UGP' directory exists.")
    sys.exit(1)

class NewTransferModel(nn.Module):
    def __init__(self, num_new_classes):
        super(NewTransferModel, self).__init__()

        # Stem: Learnable adapter before pretrained backbone
        self.stem = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(178),
            nn.Conv1d(64, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

        # Backbone: Pretrained and frozen
        original_configs = BaseConfig()
        original_configs.input_channels = 1
        original_configs.features_len = 24
        self.backbone = base_Model(original_configs)
        self.backbone.logits = nn.Identity()

        # Head: New classifier
        feature_dim = original_configs.final_out_channels * original_configs.features_len
        self.feedforward_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_new_classes)
        )

    def freeze_backbone(self):
        """Freeze all parameters in the backbone."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass for end-to-end training."""
        x = self.stem(x)
        _, features = self.backbone(x)
        x_flat = features.reshape(features.shape[0], -1)
        return self.feedforward_net(x_flat)

    def extract_features(self, x):
        """Return feature vectors before classification head."""
        with torch.no_grad():
            x = self.stem(x)
            _, features = self.backbone(x)
            x_flat = features.reshape(features.shape[0], -1)
        return x_flat
