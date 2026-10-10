"""
Configuration file for Traffic Sign Recognition System
Centralized configuration for paths, hyperparameters, and settings
"""

from pathlib import Path

# ========================================
# PROJECT PATHS
# ========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
FEEDBACK_DIR = DATA_DIR / "feedback"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
SAMPLE_PREDICTIONS_DIR = OUTPUT_DIR / "sample_predictions"

# Model file paths
MODEL_PATH = MODEL_DIR / "traffic_sign_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"

# Output file paths
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

# ========================================
# DATASET CONFIGURATION
# ========================================
# Supported image formats
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# ========================================
# MODEL HYPERPARAMETERS
# ========================================
# Image preprocessing
IMAGE_SIZE = (64, 64)  # Height x Width
IMAGE_CHANNELS = 3  # RGB
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# Optimizer settings
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'

# Model architecture
DROPOUT_RATE = 0.4
L2_REGULARIZATION = 0.0001

# ========================================
# CALLBACKS CONFIGURATION
# ========================================
# Early stopping
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MONITOR = 'val_loss'

# ReduceLROnPlateau
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-7

# ========================================
# DATA AUGMENTATION
# ========================================
AUGMENTATION_ROTATION = 0.15  # 15 degrees
AUGMENTATION_ZOOM = 0.15  # 15%
AUGMENTATION_TRANSLATION = 0.1  # 10%

# ========================================
# REPRODUCIBILITY
# ========================================
RANDOM_SEED = 42

# ========================================
# DISPLAY SETTINGS
# ========================================
VERBOSE = 1  # Training verbosity level

# ========================================
# CLASS NAMES MAPPING (GTSRB Standard)
# ========================================
# These are the standard GTSRB traffic sign names
GTSRB_CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# ========================================
# ENSURE DIRECTORIES EXIST
# ========================================
def create_directories():
    """Create necessary directories if they don't exist"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("✓ All necessary directories created/verified")

if __name__ == "__main__":
    create_directories()
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📁 Model Directory: {MODEL_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
