"""
Single Image Prediction Module
Handles prediction for individual traffic sign images with enhanced preprocessing
"""

import json
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
from tensorflow import keras

from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE
)


def detect_and_crop_sign(image: Image.Image, margin: float = 0.1) -> Image.Image:
    """
    Detect and crop the main traffic sign from image using color-based detection
    
    Args:
        image: PIL Image
        margin: Extra margin around detected sign (0.1 = 10%)
        
    Returns:
        Cropped PIL Image containing the main sign
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_height, original_width = img_cv.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for traffic signs (red, blue, yellow)
    # Red signs (two ranges because red wraps around in HSV)
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Blue signs
    blue_lower = np.array([100, 70, 50])
    blue_upper = np.array([130, 255, 255])
    
    # Yellow signs
    yellow_lower = np.array([15, 70, 50])
    yellow_upper = np.array([35, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Combine all masks
    mask = mask_red1 | mask_red2 | mask_blue | mask_yellow
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No sign detected, return center crop
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        return image.crop((left, top, left + size, top + size))
    
    # Find largest contour (main sign)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(original_width - x, w + 2 * margin_x)
    h = min(original_height - y, h + 2 * margin_y)
    
    # Make it square (traffic signs are typically square/circular)
    size = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    
    x = max(0, center_x - size // 2)
    y = max(0, center_y - size // 2)
    
    # Ensure we don't go out of bounds
    if x + size > original_width:
        x = original_width - size
    if y + size > original_height:
        y = original_height - size
    
    # Crop from original PIL image
    cropped = image.crop((x, y, x + size, y + size))
    
    return cropped


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better recognition
    
    Args:
        image: PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Enhance color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
    
    return image


def load_and_preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int] = IMAGE_SIZE,
    auto_crop: bool = True,
    enhance: bool = True
) -> Tuple[np.ndarray, Optional[Image.Image]]:
    """
    Load and preprocess a single image for prediction with advanced preprocessing
    
    Args:
        image_path: Path to image file
        target_size: Target size to resize image (height, width)
        auto_crop: Automatically detect and crop main sign
        enhance: Apply image enhancement
        
    Returns:
        Tuple of (preprocessed image array, cropped PIL image for visualization)
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Store original for visualization
    cropped_img = None
    
    # Auto-crop to detect main sign
    if auto_crop:
        try:
            img = detect_and_crop_sign(img)
            cropped_img = img.copy()
        except Exception as e:
            print(f"⚠️ Auto-crop failed, using full image: {e}")
    
    # Enhance image quality
    if enhance:
        try:
            img = enhance_image(img)
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e}")
    
    # Resize image using high-quality LANCZOS resampling
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize to 0-1 range
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, cropped_img


def predict_image(
    image_path: str,
    model: keras.Model = None,
    class_names: dict = None,
    top_k: int = 3,
    auto_crop: bool = True,
    enhance: bool = True,
    use_tta: bool = False
) -> dict:
    """
    Predict traffic sign class for a single image with enhanced preprocessing
    
    Args:
        image_path: Path to image file
        model: Trained Keras model (loads if None)
        class_names: Dictionary of class names (loads if None)
        top_k: Number of top predictions to return
        auto_crop: Automatically detect and crop main sign
        enhance: Apply image enhancement
        use_tta: Use test-time augmentation (slower but more accurate)
        
    Returns:
        Dictionary containing prediction results
    """
    # Load model if not provided
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
    
    # Load class names if not provided
    if class_names is None:
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names not found at: {CLASS_NAMES_PATH}")
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
    
    # Preprocess image
    img_array, cropped_img = load_and_preprocess_image(
        image_path, 
        auto_crop=auto_crop,
        enhance=enhance
    )
    
    # Make prediction (with optional test-time augmentation)
    if use_tta:
        predictions = predict_with_tta(img_array, model)
    else:
        predictions = model.predict(img_array, verbose=0)
    
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_confidences = predictions[0][top_indices]
    
    # Get predicted class (highest probability)
    predicted_class_idx = top_indices[0]
    predicted_class_name = class_names.get(str(predicted_class_idx), f"Class {predicted_class_idx}")
    confidence = float(top_confidences[0])
    
    # Get top k predictions
    top_predictions = []
    for idx, conf in zip(top_indices, top_confidences):
        class_name = class_names.get(str(idx), f"Class {idx}")
        top_predictions.append({
            'class_idx': int(idx),
            'class_name': class_name,
            'confidence': float(conf)
        })
    
    # Return results with cropped image
    return {
        'predicted_class_idx': int(predicted_class_idx),
        'predicted_class_name': predicted_class_name,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'cropped_image': cropped_img  # For visualization
    }


def predict_with_tta(img_array: np.ndarray, model: keras.Model, n_augment: int = 5) -> np.ndarray:
    """
    Predict with test-time augmentation for improved accuracy
    
    Args:
        img_array: Preprocessed image array
        model: Keras model
        n_augment: Number of augmentations to average
        
    Returns:
        Averaged predictions
    """
    from scipy import ndimage
    
    predictions = []
    
    # Original prediction
    pred = model.predict(img_array, verbose=0)
    predictions.append(pred)
    
    # Small rotations using scipy
    for angle in [-5, 5]:
        img_rotated = rotate_image_scipy(img_array, angle)
        pred = model.predict(img_rotated, verbose=0)
        predictions.append(pred)
    
    # Slight brightness adjustments
    for factor in [0.9, 1.1]:
        img_bright = img_array * factor
        img_bright = np.clip(img_bright, 0, 1)
        pred = model.predict(img_bright, verbose=0)
        predictions.append(pred)
    
    # Average all predictions
    avg_pred = np.mean(predictions[:n_augment], axis=0)
    
    return avg_pred


def rotate_image_scipy(img_array: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image array by given angle using scipy
    
    Args:
        img_array: Image array with batch dimension [1, H, W, C]
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image array
    """
    from scipy import ndimage
    
    # Remove batch dimension
    img = img_array[0]
    
    # Rotate each channel separately
    rotated = np.zeros_like(img)
    for i in range(img.shape[2]):
        rotated[:, :, i] = ndimage.rotate(img[:, :, i], angle, reshape=False, order=1)
    
    # Clip values to valid range
    rotated = np.clip(rotated, 0, 1)
    
    # Add batch dimension back
    return np.expand_dims(rotated, axis=0)


def predict_and_display(image_path: str, auto_crop: bool = True, enhance: bool = True):
    """
    Predict and display results for a single image with enhanced preprocessing
    
    Args:
        image_path: Path to image file
        auto_crop: Automatically detect and crop main sign
        enhance: Apply image enhancement
    """
    print(f"\n{'='*60}")
    print(f"🔮 TRAFFIC SIGN PREDICTION (Enhanced)")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Auto-crop: {'✓' if auto_crop else '✗'}")
    print(f"Enhancement: {'✓' if enhance else '✗'}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    # Make prediction
    try:
        result = predict_image(
            image_path,
            auto_crop=auto_crop,
            enhance=enhance
        )
        
        print(f"\n{'='*60}")
        print(f"🚦 PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"\n🎯 Predicted Sign: {result['predicted_class_name']}")
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        
        # Confidence interpretation
        if result['confidence'] > 0.9:
            print(f"   └─ Very High Confidence ✓")
        elif result['confidence'] > 0.7:
            print(f"   └─ High Confidence ✓")
        elif result['confidence'] > 0.5:
            print(f"   └─ Moderate Confidence ⚠️")
        else:
            print(f"   └─ Low Confidence ❌")
        
        print(f"\n🔝 Top 3 Predictions:")
        print(f"{'Rank':<6} {'Confidence':<12} {'Class'}")
        print("-" * 60)
        for i, pred in enumerate(result['top_predictions'], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{emoji} {i:<3} {pred['confidence']*100:>6.2f}%      {pred['class_name']}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Predict image from command line argument
        image_path = sys.argv[1]
        
        # Optional flags
        auto_crop = "--no-crop" not in sys.argv
        enhance = "--no-enhance" not in sys.argv
        
        predict_and_display(image_path, auto_crop=auto_crop, enhance=enhance)
    else:
        print("Usage: python -m src.predict <image_path> [--no-crop] [--no-enhance]")
        print("\nExamples:")
        print("  python -m src.predict data/test/0/gtsrb_0_00000.png")
        print("  python -m src.predict my_sign.jpg --no-crop")
        print("  python -m src.predict my_sign.jpg --no-enhance")
        print("\nOptions:")
        print("  --no-crop      Disable automatic sign detection and cropping")
        print("  --no-enhance   Disable image enhancement")
