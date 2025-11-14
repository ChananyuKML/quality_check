import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

def get_feature_extractor():
    """
    Loads a pre-trained ResNet-50 model and modifies it to be a 
    feature extractor.
    """
    # Use the new 'weights' parameter for pre-trained models
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    
    # Remove the final classification layer (model.fc)
    # We'll use the output of the 'avgpool' layer as our feature vector
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    # Set the model to evaluation mode (e.g., disables dropout)
    feature_extractor.eval()
    
    return feature_extractor, weights.transforms()

def get_image_embedding(image_path, model, preprocess):
    """
    Loads and preprocesses an image, then passes it through the
    feature extractor to get its embedding.
    """
    try:
        # Open the image and convert to RGB
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    # Apply the transformations
    img_t = preprocess(img)
    
    # Add a batch dimension (models expect batches)
    # Shape becomes [1, 3, 224, 224]
    batch_t = torch.unsqueeze(img_t, 0)
    
    # Get the embedding
    with torch.no_grad():
        # Pass the image through the model
        embedding = model(batch_t)
    
    # The output from ResNet-50's avgpool is [1, 2048, 1, 1]
    # We flatten it to a 1D vector of size 2048
    embedding_flat = torch.flatten(embedding)
    
    # L2 normalize the embedding (good practice for cosine similarity)
    embedding_normalized = F.normalize(embedding_flat, p=2, dim=0)
    
    return embedding_normalized

# --- Main execution ---

# 1. Initialize the model and preprocessing pipeline
extractor, preprocess_pipeline = get_feature_extractor()
print("✅ Model loaded and ready.")

# 2. Define your images
#    (Replace these with paths to your own images)
img1_path = "test_output/ref_front.jpg"
img2_path = "test_output/frontal_view_output.jpg"
img3_path = "test_output/pc.jpg" # A different image

# 3. Get embeddings
print(f"Processing {img1_path}...")
emb1 = get_image_embedding(img1_path, extractor, preprocess_pipeline)

print(f"Processing {img2_path}...")
emb2 = get_image_embedding(img2_path, extractor, preprocess_pipeline)

print(f"Processing {img3_path}...")
emb3 = get_image_embedding(img3_path, extractor, preprocess_pipeline)

# 4. Calculate and print similarity
if emb1 is not None and emb2 is not None and emb3 is not None:
    # Compare image 1 and 2 (should be similar if they are)
    sim_1_2 = F.cosine_similarity(emb1, emb2, dim=0)
    
    # Compare image 1 and 3 (should be less similar)
    sim_1_3 = F.cosine_similarity(emb1, emb3, dim=0)

    print("\n--- Results ---")
    print(f"Shape of one embedding: {emb1.shape}")
    print(f"Similarity between '{img1_path}' and '{img2_path}': {sim_1_2.item():.4f}")
    print(f"Similarity between '{img1_path}' and '{img3_path}': {sim_1_3.item():.4f}")

else:
    print("\nSkipping similarity calculation due to file errors.")
    print("Please create 'image1.jpg', 'image2.jpg', and 'image3.jpg' to run.")
