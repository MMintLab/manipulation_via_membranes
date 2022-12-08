import torch
import torch.nn as nn

from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet_loading_utils import get_pretrained_pointnet_classifier


class PointNetObjectEmbedding(nn.Module):
    def __init__(self, obj_embedding_size, freeze_pointnet=True):
        super().__init__()
        self.obj_embedding_size = obj_embedding_size
        self.pointnet_classifier = get_pretrained_pointnet_classifier(freeze=freeze_pointnet)
        self.embedding_fc = nn.Linear(256, self.obj_embedding_size)

    def forward(self, x):
        x = x.transpose(-2, -1)  # reshape to (B, K, N)
        x, _, _ = self.pointnet_classifier.base(x)
        # apply cassifier except last linear layer and dropout:
        for i, layer_i in enumerate(self.pointnet_classifier.classifier[:-2]):
            x = layer_i(x)
        out = self.embedding_fc(x)  # Returns a B x 40
        return out


# Debug:
if __name__ == '__main__':
    x = torch.ones((5, 10, 3))
    model = PointNetObjectEmbedding(10, freeze_pointnet=True)
    out = model(x)
    print(out.shape)