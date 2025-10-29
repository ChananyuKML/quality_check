import torch
import torce.nn as nn
import torchvision.models as models

class KeypointDetection(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_keypoints = num_keypoints
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = backbone.fc.in_features
        keypoint_head = nn.Sequential(
            nn.Linear(num_ftrs, self.num_keypoints * 2),
            nn.Sigmoid()
        )
        backbone.fc = keypoint_head
        self.backbone = backbone

    def forward(self, x):
        keypoints = self.backbone(x)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)
        return keypoints

