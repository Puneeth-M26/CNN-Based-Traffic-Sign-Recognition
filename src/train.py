"""
Model Training Module
Handles the complete training pipeline including callbacks and history saving
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
from datetime import datetime

from src.config import (
    TRAIN_DIR, MODEL_PATH, CLASS_NAMES_PATH, TRAINING_HISTORY_PATH,
    PLOTS_DIR, EPOCHS, RANDOM_SEED, VERBOSE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MONITOR,
    REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_LR,
    GTSRB_CLASS_NAMES, create_directories
)
from src.preprocessing import create_train_val_datasets
from src.model import create_model, display_model_summary


def create_callbacks(model_path: Path) -> list:
    """
    Create training callbacks
    
    Args:
        model_path: Path to save the best model
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1
        )
    ]
    
    return callbacks


def plot_training_history(history: keras.callbacks.History, output_dir: Path):
    """
    Plot and save training history graphs
    
    Args:
        history: Keras training history object
        output_dir: Directory to save plots
    """
    history_dict = history.history
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history_dict['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_path}")
    
    plt.close()


def save_training_history(history: keras.callbacks.History, output_path: Path):
    """
    Save training history to JSON file
    
    Args:
        history: Keras training history object
        output_path: Path to save JSON file
    """
    history_dict = history.history
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {}
    for key, value in history_dict.items():
        if isinstance(value, np.ndarray):
            serializable_history[key] = value.tolist()
        elif isinstance(value, list):
            serializable_history[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        else:
            serializable_history[key] = value
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"✓ Training history saved to: {output_path}")


def save_class_names_mapping(class_names: list, output_path: Path):
    """
    Save class names with descriptive labels to JSON
    
    Args:
        class_names: List of class folder names (as strings)
        output_path: Path to save JSON file
    """
    # Create mapping with descriptive names
    class_mapping = {}
    for idx, class_name in enumerate(class_names):
        class_num = int(class_name)
        class_mapping[str(idx)] = GTSRB_CLASS_NAMES.get(class_num, f"Class {class_num}")
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Class names saved to: {output_path}")


def train_model():
    """
    Complete training pipeline
    """
    print("\n" + "="*60)
    print("🚦 TRAFFIC SIGN RECOGNITION - MODEL TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seed for reproducibility
    keras.utils.set_random_seed(RANDOM_SEED)
    
    # Create necessary directories
    create_directories()
    
    # Load datasets
    train_ds, val_ds, class_names = create_train_val_datasets(str(TRAIN_DIR))
    num_classes = len(class_names)
    
    # Save class names
    save_class_names_mapping(class_names, CLASS_NAMES_PATH)
    
    # Create model
    model = create_model(num_classes)
    
    # Display model summary
    display_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks(MODEL_PATH)
    
    # Train model
    print(f"\n{'='*60}")
    print(f"🎯 STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=VERBOSE
    )
    
    # Training complete
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    
    # Get final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n📊 Final Training Metrics:")
    print(f"   Training Accuracy: {final_train_acc:.4f}")
    print(f"   Training Loss: {final_train_loss:.4f}")
    print(f"   Validation Accuracy: {final_val_acc:.4f}")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    
    # Find best epoch
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Save training history
    save_training_history(history, TRAINING_HISTORY_PATH)
    
    # Plot training history
    plot_training_history(history, PLOTS_DIR)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📁 SAVED FILES")
    print(f"{'='*60}")
    print(f"✓ Model: {MODEL_PATH}")
    print(f"✓ Class names: {CLASS_NAMES_PATH}")
    print(f"✓ Training history: {TRAINING_HISTORY_PATH}")
    print(f"✓ Training plots: {PLOTS_DIR / 'training_history.png'}")
    print(f"{'='*60}")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Next steps:")
    print(f"   1. Run evaluation: python -m src.evaluate")
    print(f"   2. Launch Streamlit app: streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_model()
