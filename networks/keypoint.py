import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import os
import cv2

class YOLOSegmentationDatasetV2(Dataset):
    """
    Fixed Dataset class that uses a default Albumentations pipeline
    if no transform is provided.
    """
    def __init__(self, image_dir, label_dir, resize_shape=(640, 640), transform=None):
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        # --- THIS IS THE FIX ---
        if transform:
            # If the user provides a custom Albumentations pipeline, use it
            self.transform = transform
        else:
            # If no transform is provided, create a default one
            self.transform = A.Compose([
                A.Resize(height=resize_shape[0], width=resize_shape[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        
        # 1. --- Load Image (Full Size) ---
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load as-is, don't resize yet
        image = cv2.imread(img_path)
        # Handle cases where image might not load
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            # Return a dummy tensor, or the next item
            return self.__getitem__((index + 1) % len(self)) 
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get ORIGINAL height and width
        original_h, original_w, _ = image.shape
        
        # 2. --- Load Labels and Create FULL-SIZE Mask ---
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Create an empty mask with the ORIGINAL image size
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                        
                    class_id = int(parts[0])
                    polygon_norm = np.array(parts[1:], dtype=float).reshape(-1, 2)
                    
                    # De-normalize points to the ORIGINAL image dimensions
                    polygon_denorm = polygon_norm.copy()
                    polygon_denorm[:, 0] *= original_w # Scale x by original Width
                    polygon_denorm[:, 1] *= original_h # Scale y by original Height
                    
                    points = polygon_denorm.astype(np.int32)
                    
                    # Draw on the full-size mask
                    cv2.fillPoly(mask, [points], color=(class_id + 1))

        # 3. --- Apply Transforms ---
        # Pass both the image and mask to albumentations
        # This part was already correct
        transformed = self.transform(image=image, mask=mask)
        
        image_tensor = transformed['image']
        mask_tensor = transformed['mask'] # This is now resized
        
        # Albumentations doesn't change mask type, so we set it to long
        return image_tensor, mask_tensor.long()
class YOLOSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, resize_shape=(640, 640)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.resize_shape = resize_shape # (Height, Width)
        
        # Get a sorted list of image filenames
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """Loads one sample (image and mask)."""
        
        # 1. --- Load Image ---
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read with OpenCV, which returns a (H, W, C) NumPy array
        image = cv2.imread(img_path)
        # Convert from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to the target shape
        # Note: cv2.resize expects (Width, Height)
        image_resized = cv2.resize(
            image, 
            (self.resize_shape[1], self.resize_shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 2. --- Load Labels and Create Mask ---
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Create an empty mask for this image (H, W)
        # We use the resized shape
        mask = np.zeros(self.resize_shape, dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    
                    if len(parts) < 3: # Must have class_id + at least one point
                        continue
                        
                    # The class ID from the file
                    class_id = int(parts[0])
                    
                    # Parse normalized (x, y) points
                    polygon_norm = np.array(parts[1:], dtype=float).reshape(-1, 2)
                    
                    # De-normalize points to the *resized* image dimensions
                    polygon_denorm = polygon_norm.copy()
                    polygon_denorm[:, 0] *= self.resize_shape[1] # Scale x by Width
                    polygon_denorm[:, 1] *= self.resize_shape[0] # Scale y by Height
                    
                    # Convert points to the integer format cv2.fillPoly expects
                    points = polygon_denorm.astype(np.int32)
                    
                    # --- Draw the polygon on the mask ---
                    # ❗️ IMPORTANT:
                    # We use (class_id + 1) as the pixel value.
                    # This ensures that class '0' becomes '1' on the mask,
                    # leaving '0' reserved for the background class.
                    cv2.fillPoly(mask, [points], color=(class_id + 1))

        # 3. --- Convert to PyTorch Tensors ---
        
        # Image: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        image_tensor = torch.from_numpy(
            image_resized.transpose(2, 0, 1)
        ).float() / 255.0
        
        # Mask: (H, W) -> (H, W)
        # Convert to a LongTensor, as required by loss functions (e.g., CrossEntropyLoss)
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor

class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.prediction_dims = 5 + num_classes
        out_channels = self.num_anchors * self.prediction_dims

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.prediction_layer = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = self.conv_block(x)
        predictions = self.prediction_layer(features)
        B, _, G, _ = predictions.shape
        predictions = predictions.view(B, self.num_anchors, self.prediction_dims, G, G)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()
        return predictions

class KeypointHeatmapHead(nn.Module):
    """
    This is a "decoder" or "head" that takes a feature map 
    (e.g., [B, 2048, 13, 13]) and upsamples it to produce heatmaps
    (e.g., [B, num_keypoints, 104, 104]).
    """
    def __init__(self, in_channels=3, num_keypoints=4):
        super().__init__()
        
        # This stack of "deconvolution" (ConvTranspose2d) layers 
        # will upsample the feature map.
        
        # We start at in_channels (e.g., 2048 from ResNet-50)
        # and end at num_keypoints.
        
        # Input: (B, 2048, 13, 13)
        self.deconv_stack = nn.Sequential(
            # (B, 2048, 13, 13) -> (B, 256, 26, 26)
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # (B, 256, 26, 26) -> (B, 256, 52, 52)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # (B, 256, 52, 52) -> (B, 256, 104, 104)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Final 1x1 conv to get one map per keypoint
            # (B, 256, 104, 104) -> (B, num_keypoints, 104, 104)
            nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0)
        )
        # NOTE: We do not use a Sigmoid here. The model outputs raw logits.
        # The loss function (like MSELoss) is applied directly to these.

    def forward(self, x):
        return self.deconv_stack(x)

class KeypointDetection(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # 1. Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Chop off the head (avgpool and fc)
        # We will use the backbone as a feature extractor
        self.stem = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 3. Add our new heatmap head
        # It takes the 2048 output channels from layer4
        self.head = KeypointHeatmapHead(
            in_channels=2048, 
            num_keypoints=num_keypoints
        )

    def forward(self, x):
        # Input shape (e.g., B, 3, 416, 416)
        
        # Pass through the ResNet backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) # (B, 2048, 13, 13)
        
        # Pass the rich feature map to our head
        heatmaps = self.head(features) # (B, num_keypoints, 104, 104)
        
        return heatmaps
