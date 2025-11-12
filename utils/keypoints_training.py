from utils/networks import *
import torch
import torch.nn as nn
import torchvision import datasets, transform
from torch.utils.data import DataLoader, Dataset, random_split
import argparse

transform = transforms.Compose([
    transforms.resize((224, 224)),
    transforms.ToTensor()
])

class ImageKeypointDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
                file_path =  os.path.join(root_dir)

        # print(os.listdir(root_dir))
        # walk through dataset/classX/{real,fake}
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            # print(class_path)
            if not os.path.isdir(class_path):
                continue

            for label_name in ["0_real", "1_fake"]:
                label_path = os.path.join(class_path, label_name)
                # print(label_path)
                if not os.path.isdir(label_path):
                    continue
                
                for file_name in os.listdir(label_path):
                    file_path = os.path.join(label_path, file_name)
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.data.append((file_path, self.label_map[label_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(len(self.data))
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


parser = argparse.ArgumentParser(description="Add an option for training process")
parser.add_argument("--dataset", type=str, help="Dataset used for training")
parser.add_argument("--key_num", type=str, help="Number of keypoints")
args = parser.parse_args()

def train(train_loader, num_points, epochs, device):
    model = KeypointsDetection(num_points)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
		train_loss = 0
		correct = 0
		iter = 1
        
        for image in train_loader:
            optimizer.zero_grad()
            pred_keypoints = model(batch_images)
            loss = criterion(pred_keypoints, true_keypoints)
            loss.backward()
            optimizer.step()
            print(f"Trained from {epoch}/{epoches}, Loss: {loss.item():.6f}")


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    predicted_keypoints = model(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {predicted_keypoints.shape}")
    print("---")
    
    
    # --- 3. Example Training Step ---
    print("Running a dummy training step...")
    # For a real project, this would come from your DataLoader
    
    # A dummy batch of 16 images
    batch_images = torch.randn(16, 3, 224, 224).to(device)
    
    # A dummy batch of 16 "ground truth" labels.
    # **IMPORTANT**: These MUST be normalized to the [0, 1] range
    # to match the model's Sigmoid output.
    # (e.g., x_pixel / image_width, y_pixel / image_height)
    true_keypoints = torch.rand(16, N_POINTS, 2).to(device)
    
    
    # --- Define Loss and Optimizer ---
    
    # Mean Squared Error (MSE) is a good choice for coordinate regression
    criterion = nn.MSELoss() 
    # Adam is a standard, robust optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # --- The Training Loop (one step) ---
    
    # 1. Zero the gradients
    optimizer.zero_grad()
    
    # 2. Get model predictions
    pred_keypoints = model(batch_images)
    
    # 3. Calculate the loss
    loss = criterion(pred_keypoints, true_keypoints)
    
    # 4. Backpropagate the loss
    loss.backward()
    
    # 5. Update the model weights
    optimizer.step()
    
    print(f"Dummy training step complete.")
    print(f"Loss: {loss.item():.6f}")
    
