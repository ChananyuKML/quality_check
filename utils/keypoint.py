import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os, time

class KeypointResNet(nn.Module):
    def __init__(self, num_keypoints=4):
        super(KeypointResNet, self).__init__()
        
        # 1. Load a pre-trained ResNet-50 model
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 2. Get the number of features from the original fully-connected layer
        # For ResNet-50, this is 2048
        num_ftrs = self.backbone.fc.in_features
        
        # 3. "Cut off" the original head by replacing it with an Identity layer
        # This gives us the raw feature vector from the backbone
        self.backbone.fc = nn.Identity()
        
        # 4. Define our new "regression head"
        # It takes the features (e.g., 2048) and outputs N * 2 coordinates
        self.keypoint_head = nn.Sequential(
            nn.Linear(num_ftrs, num_keypoints * 2),
            nn.Sigmoid()  # Squashes output to be between 0 and 1
        )
        
        self.num_keypoints = num_keypoints

    def forward(self, x):
        # 1. Get features from the backbone
        features = self.backbone(x)
        
        # 2. Get coordinates from the new head
        # Output shape will be (batch_size, num_keypoints * 2)
        keypoints = self.keypoint_head(features)
        
        # 3. (Optional but recommended) Reshape for easier use
        # Reshape to (batch_size, num_keypoints, 2)
        keypoints = keypoints.view(keypoints.size(0), self.num_keypoints, 2)
        
        return keypoints
    
class YoloKeypointDataset(Dataset):
    def __init__(self, img_dir, label_dir, num_keypoints=4, 
                 has_visibility=False, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            label_dir (str): Directory with all the .txt label files.
            num_keypoints (int): The number of keypoints (e.g., 4).
            has_visibility (bool): True if YOLO format includes visibility 
                                   (e.g., x, y, vis).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.num_keypoints = num_keypoints
        self.has_visibility = has_visibility
        
        # Get all image filenames (e.g., 'product_001.jpg')
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. Get label path
        # (e.g., 'product_001.jpg' -> 'product_001.txt')
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 3. Load image
        image = Image.open(img_path).convert("RGB")
        
        # 4. Load and parse label
        keypoints_flat = []
        try:
            with open(label_path, 'r') as f:
                # We assume one line (one object) per file
                line = f.readline()
                parts = line.strip().split()
                
                # The YOLO format is:
                # <class> <x> <y> <w> <h> <kpt1_x> <kpt1_y> ...
                # We skip the first 5 elements (class + bounding box)
                kpt_data = parts[5:]
                
                if self.has_visibility:
                    # Data is [x1, y1, v1, x2, y2, v2, ...]
                    # We step by 3 and take the first two (x, y)
                    for i in range(self.num_keypoints):
                        kpt_index = i * 3
                        keypoints_flat.append(float(kpt_data[kpt_index]))     # x
                        keypoints_flat.append(float(kpt_data[kpt_index + 1])) # y
                else:
                    # Data is [x1, y1, x2, y2, ...]
                    # We just take all of them (assuming 2 * num_keypoints)
                    keypoints_flat = [float(p) for p in kpt_data]

        except Exception as e:
            print(f"Error parsing label file {label_path}: {e}")
            # Return dummy data if file is bad
            keypoints_flat = [0.0] * (self.num_keypoints * 2)
            
        keypoints_target = torch.tensor(keypoints_flat, dtype=torch.float32)
        
        # 5. Apply image transforms
        if self.transform:
            image = self.transform(image)
            
        return image, keypoints_target

def train_model():
    # --- !! UPDATE THESE PARAMETERS !! ---
    IMG_DIR = "data/images"        # Directory with your images
    LABEL_DIR = "data/labels"      # Directory with your YOLO .txt labels
    NUM_KEYPOINTS = 4              # Number of points (e.g., 4 corners)
    HAS_VISIBILITY = False         # Set True if your labels are (x, y, vis)
    
    # --- Hyperparameters ---
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    print("Starting training script...")
    
    # --- 1. Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Define Transforms ---
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Create Dataset and DataLoader ---
    try:
        train_dataset = YoloKeypointDataset(img_dir=IMG_DIR,
                                            label_dir=LABEL_DIR,
                                            num_keypoints=NUM_KEYPOINTS,
                                            has_visibility=HAS_VISIBILITY,
                                            transform=data_transform)
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=4) # num_workers speeds up loading
    except FileNotFoundError:
        print(f"Error: Could not find data directories.")
        print(f"Looked for images in: {os.path.abspath(IMG_DIR)}")
        print(f"Looked for labels in: {os.path.abspath(LABEL_DIR)}")
        return
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # --- 4. Initialize Model, Loss, Optimizer ---
    model = KeypointResNet(num_keypoints=NUM_KEYPOINTS).to(device)
    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model, Dataset, and Optimizer initialized. Starting training...")

    # --- 5. The Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for i, (images, keypoints_target) in enumerate(train_loader):
            # Move data to the device
            images = images.to(device)
            keypoints_target = keypoints_target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # Model output shape: (batch_size, N_keypoints, 2)
            keypoints_pred = model(images)
            
            # Flatten predictions to (batch_size, N_keypoints * 2)
            # This matches the target shape
            keypoints_pred_flat = keypoints_pred.view(keypoints_pred.size(0), -1)
            
            # --- Calculate Loss ---
            loss = criterion(keypoints_pred_flat, keypoints_target)
            
            # --- Backward Pass + Optimize ---
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # --- End of Epoch ---
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {epoch_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Optional: Save a checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = f"keypoint_resnet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    print("Training finished.")
    final_model_path = "keypoint_resnet_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

# --- Run the training ---
if __name__ == "__main__":
    train_model()