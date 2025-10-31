import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm  # For a nice progress bar
from networks.keypoint import *

# --- Configuration ---
IMAGE_DIR_PATH = "dataset/yolo/images"  # Path to your images
LABEL_DIR_PATH = "dataset/yolo/labels"  # Path to your .txt labels
BATCH_SIZE = 1
IMG_HEIGHT = 640
IMG_WIDTH = 640
NUM_KEYPOINTS = 4  # Example: 17 keypoints for COCO
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transforms = T.Compose([
    T.Resize((IMG_WIDTH, IMG_HEIGHT)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 1. Create the Dataset instance
seg_dataset = YOLOSegmentationDatasetV2(
    image_dir=IMAGE_DIR_PATH,
    label_dir=LABEL_DIR_PATH,
    resize_shape=(IMG_HEIGHT, IMG_WIDTH),
    transform=train_transforms
    )

# 2. Create the DataLoader instance
# Because we resize all images/masks to the same shape,
# we can use the default collate_fn.
train_loader = DataLoader(
    dataset=seg_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      # Shuffle for training
    num_workers=4,     # Use multiple CPU cores to load data
    pin_memory=True
)

model = KeypointDetection(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"--- Starting training on {DEVICE} ---")

for epoch in range(EPOCHS):
    
    # --- Training Phase ---
    model.train()  # Set model to training mode (enables batchnorm, dropout)
    
    running_loss = 0.0
    
    # Use tqdm for a progress bar over the training data
    for images, target_heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        
        # 1. Move data to the target device (e.g., GPU)
        images = images.to(DEVICE)
        target_heatmaps = target_heatmaps.to(DEVICE)
        
        # 2. Zero the gradients
        # This clears gradients from the previous batch
        optimizer.zero_grad()
        
        # 3. Forward Pass
        # Get the model's predictions (predicted heatmaps)
        predicted_heatmaps = model(images)
        
        # 4. Calculate Loss
        # Compare the model's output with the ground truth
        loss = loss_function(predicted_heatmaps, target_heatmaps)
        
        # 5. Backward Pass
        # Calculate the gradients of the loss w.r.t. model parameters
        loss.backward()
        
        # 6. Optimizer Step
        # Update the model's weights using the calculated gradients
        optimizer.step()
        
        # Accumulate the loss for logging
        running_loss += loss.item()

    # --- End of Epoch ---
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.6f}")
    
    # -----------------------------------------------------------------
    # (Optional) Validation Phase
    # You would add a similar loop here, but with model.eval()
    # and no optimizer.step() or loss.backward()
    # -----------------------------------------------------------------

print("--- Training finished ---")

# (Optional) Save your trained model
# torch.save(model.state_dict(), "keypoint_model.pth")






# --- Example: Loop through one batch ---
try:
    # Get one batch
    images, masks = next(iter(data_loader))

    print(f"--- Batch Loaded ---")
    print(f"Images batch shape: {images.shape}")
    print(f"Masks batch shape:  {masks.shape}")
    print(f"Masks data type:    {masks.dtype}")
    print(f"Unique mask values: {torch.unique(masks)}")

except StopIteration:
    print("DataLoader is empty or paths are incorrect.")
