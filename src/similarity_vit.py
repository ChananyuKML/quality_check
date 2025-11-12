import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import sys
import argparse # Import argparse

class FeatureExtractor(nn.Module):
    """
    A class to extract features using a pre-trained Vision Transformer (ViT-B/16).
    
    This model removes the final classification head ('heads')
    to output the high-level feature vector (the [class] token embedding).
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load a pre-trained ViT-B/16 model
        # weights=models.ViT_B_16_Weights.DEFAULT loads the best available weights
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # Replace the final classification head with an Identity layer
        # The 'heads' module is the classifier.
        # This makes the model output the 768-dim [class] token embedding
        # which comes from the model's encoder.
        base_model.heads = nn.Identity()
        
        self.features = base_model
        
    def forward(self, x):
        """
        Forward pass to get features.
        """
        # Get features from the ViT model
        # The output will be (batch_size, 768)
        x = self.features(x)
        # The output is already flat (B, 768), so no need for torch.flatten
        return x

def preprocess_image(image_path):
    """
    Loads an image from a given path and applies the necessary
    transformations for the ViT model.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    # Get the transforms associated with the ViT weights
    # This ensures we use the exact same preprocessing the model was trained with
    weights = models.ViT_B_16_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # The default transforms for ViT are:
    # 1. Resize to 256x256
    # 2. Center crop to 224x224
    # 3. Convert to a PyTorch tensor
    # 4. Normalize using ImageNet mean and standard deviation
    
    try:
        # Open the image using PIL
        image = Image.open(image_path).convert('RGB')
        # Apply the transformations
        image_tensor = transform(image)
        # Add a batch dimension (B, C, H, W) -> (1, C, H, W)
        return image_tensor.unsqueeze(0)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None

def get_image_features(image_path, model):
    """
    Gets the feature vector for a single image.
    
    Args:
        image_path (str): The file path to the image.
        model (nn.Module): The feature extractor model.
        
    Returns:
        torch.Tensor: The extracted feature vector, or None if an error occurs.
    """
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    if image_tensor is None:
        return None
    
    # Set the model to evaluation mode (e.g., disable dropout)
    model.eval()
    
    # Get features
    with torch.no_grad(): # Disable gradient calculation for inference
        features = model(image_tensor)
        
    return features

def calculate_similarity(features1, features2):
    """
    Calculates the cosine similarity between two feature vectors.
    
    Args:
        features1 (torch.Tensor): The first feature vector (shape [1, N]).
        features2 (torch.Tensor): The second feature vector (shape [1, N]).
        
    Returns:
        float: The cosine similarity score (between -1 and 1).
    """
    # Use torch.nn.functional.cosine_similarity
    # dim=1 to compute similarity across the feature dimension
    similarity_score = F.cosine_similarity(features1, features2, dim=1)
    
    # .item() extracts the scalar value from the tensor
    return similarity_score.item()

# --- Main execution ---
if __name__ == "__main__":
    
    # --- Set up argparse ---
    parser = argparse.ArgumentParser(description="Compare the visual similarity of three images using a Vision Transformer (ViT).")
    
    # Add three required positional arguments for the image paths
    parser.add_argument("image1", help="Path to the first image file.")
    parser.add_argument("image2", help="Path to the second image file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
        
    print(f"Comparing '{args.image1}', '{args.image2}'...")
    print("Using Vision Transformer (ViT-B/16) as feature extractor.")

    # Initialize the feature extractor model
    feature_extractor = FeatureExtractor()
    
    # Get features for all three images
    print("Extracting features...")
    features1 = get_image_features(args.image1, feature_extractor)
    features2 = get_image_features(args.image2, feature_extractor)
    
    # Check if all features were extracted successfully
    if features1 is not None and features2 is not None:
        # Calculate and print the similarity for all pairs
        print("\n--- Similarity Scores ---")
        
        sim_1_2 = calculate_similarity(features1, features2)
        print(f"  (Image 1 vs Image 2): {sim_1_2:.4f}")
    
        
    else:
        print("\nCould not calculate similarity due to errors processing one or more images.")