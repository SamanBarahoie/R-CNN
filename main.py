import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    Load a pre-trained Mask R-CNN model with ResNet-50 FPN backbone.
    Returns:
        model: Pre-trained Mask R-CNN model in evaluation mode.
    """
    # Load the pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(weights="COCO_V1")
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Move model to the appropriate device
    return model

def preprocess_image(image_path):
    """
    Preprocess the input image for Mask R-CNN.
    Args:
        image_path (str): Path to the input image.
    Returns:
        image_tensor: Preprocessed image tensor ready for model input.
    """
    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H, W]
    return image_tensor.to(device), image

def perform_inference(model, image_tensor):
    """
    Perform instance segmentation on the input image tensor.
    Args:
        model: Pre-trained Mask R-CNN model.
        image_tensor: Preprocessed image tensor.
    Returns:
        predictions: Dictionary containing masks, boxes, labels, and scores.
    """
    with torch.no_grad():
        predictions = model(image_tensor)[0]  # Get predictions for the first (and only) image
    return predictions

def visualize_results(image, predictions, score_threshold=0.5):
    """
    Visualize the segmentation results including masks, bounding boxes, and labels.
    Args:
        image: Original PIL image.
        predictions: Model predictions containing masks, boxes, labels, and scores.
        score_threshold: Minimum confidence score for displaying predictions.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)

    # Get predictions
    masks = predictions["masks"].cpu().numpy()  # Shape: [N, 1, H, W]
    boxes = predictions["boxes"].cpu().numpy()  # Shape: [N, 4]
    labels = predictions["labels"].cpu().numpy()  # Shape: [N]
    scores = predictions["scores"].cpu().numpy()  # Shape: [N]

    # Plot masks and bounding boxes for predictions above threshold
    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            # Plot mask
            mask = masks[i, 0]  # Shape: [H, W]
            plt.imshow(mask, alpha=0.5, cmap="jet")

            # Plot bounding box
            box = boxes[i]
            plt.gca().add_patch(
                plt.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    fill=False, color="red", linewidth=2
                )
            )

            # Add label and score
            plt.text(
                box[0], box[1] - 5, f"Class {labels[i]}: {scores[i]:.2f}",
                color="white", fontsize=10, bbox=dict(facecolor="red", alpha=0.8)
            )

    plt.axis("off")
    plt.show()

def main(image_path):
    """
    Main function to perform instance segmentation on an image.
    Args:
        image_path (str): Path to the input image.
    """
    # Load model
    model = load_model()

    # Preprocess image
    image_tensor, image = preprocess_image(image_path)

    # Perform inference
    predictions = perform_inference(model, image_tensor)

    # Visualize results
    visualize_results(image, predictions, score_threshold=0.5)

if __name__ == "__main__":
    # Example usage (replace with your image path)
    image_path = "car.jpg"
    main(image_path)