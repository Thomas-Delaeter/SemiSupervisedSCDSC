import torch.nn as nn

class AttentionPseudoClassifier(nn.Module):
    def __init__(self, in_channels, n_clusters):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, n_clusters)

    def forward(self, x):
        attn_map = self.attention(x)
        x = x * attn_map
        x = self.feature_extractor(x)
        return self.classifier(x)

# usage
# self.pseudo_classifier = AttentionPseudoClassifier(n_z, n_clusters)
