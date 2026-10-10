"""
Model Evaluation Module
Evaluates trained model on test dataset and generates metrics including ROC curves
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from pathlib import Path
from itertools import cycle

from src.config import (
    TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH,
    OUTPUT_DIR, GTSRB_CLASS_NAMES
)
from src.preprocessing import create_test_dataset


def load_trained_model(model_path: Path) -> keras.Model:
    """
    Load trained model from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded Keras model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    
    return model


def load_class_names(class_names_path: Path) -> dict:
    """
    Load class names from JSON file
    
    Args:
        class_names_path: Path to class names JSON file
        
    Returns:
        Dictionary mapping class indices to names
    """
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names not found at: {class_names_path}")
    
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    return class_names


def evaluate_model_on_test_set(model: keras.Model, test_ds) -> tuple:
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained Keras model
        test_ds: Test dataset
        
    Returns:
        Tuple of (test_loss, test_accuracy, y_true, y_pred, y_pred_probs)
    """
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING MODEL ON TEST SET")
    print(f"{'='*60}")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions
    print(f"\n🔮 Generating predictions...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    
    print(f"✓ Generated {len(y_pred)} predictions")
    
    return test_loss, test_accuracy, y_true, y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred, class_names: dict, output_path: Path):
    """
    Generate and save confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Dictionary of class names
        output_path: Path to save confusion matrix
    """
    print(f"\n📊 Generating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine figure size based on number of classes
    num_classes = len(class_names)
    figsize = (max(20, num_classes * 0.5), max(16, num_classes * 0.4))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create labels with shortened names
    labels = []
    for i in range(num_classes):
        label = class_names.get(str(i), f"Class {i}")
        # Shorten long labels
        if len(label) > 20:
            label = label[:17] + "..."
        labels.append(f"{i}: {label}")
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,  # Don't show numbers (too many)
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Traffic Sign Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    
    plt.close()


def generate_classification_report(y_true, y_pred, class_names: dict, output_path: Path):
    """
    Generate and save classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Dictionary of class names
        output_path: Path to save report
    """
    print(f"\n📋 Generating classification report...")
    
    # Get class names list in order
    target_names = [class_names.get(str(i), f"Class {i}") for i in range(len(class_names))]
    
    # Generate report
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    
    # Print report
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(report)
    
    # Save report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRAFFIC SIGN CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Classification report saved to: {output_path}")


def print_per_class_accuracy(y_true, y_pred, class_names: dict, top_n: int = 10):
    """
    Print per-class accuracy statistics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Dictionary of class names
        top_n: Number of best/worst classes to display
    """
    print(f"\n{'='*60}")
    print(f"PER-CLASS ACCURACY")
    print(f"{'='*60}")
    
    num_classes = len(class_names)
    class_accuracies = []
    
    for i in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) == 0:
            continue
        
        # Calculate accuracy for this class
        class_preds = y_pred[class_indices]
        class_acc = np.mean(class_preds == i)
        
        class_name = class_names.get(str(i), f"Class {i}")
        class_accuracies.append((i, class_name, class_acc, len(class_indices)))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[2], reverse=True)
    
    # Print best classes
    print(f"\n🏆 Top {top_n} Best Performing Classes:")
    print(f"{'Class':<6} {'Accuracy':<10} {'Samples':<8} {'Name'}")
    print("-" * 80)
    for i, name, acc, count in class_accuracies[:top_n]:
        print(f"{i:<6} {acc*100:>6.2f}%    {count:<8} {name}")
    
    # Print worst classes
    print(f"\n⚠️  Top {top_n} Worst Performing Classes:")
    print(f"{'Class':<6} {'Accuracy':<10} {'Samples':<8} {'Name'}")
    print("-" * 80)
    for i, name, acc, count in class_accuracies[-top_n:]:
        print(f"{i:<6} {acc*100:>6.2f}%    {count:<8} {name}")


def plot_roc_curve_multiclass(y_true, y_pred_probs, class_names: dict, output_path: Path):
    """
    Generate and save ROC curves for multi-class classification
    
    Args:
        y_true: True labels (integers)
        y_pred_probs: Predicted probabilities (n_samples, n_classes)
        class_names: Dictionary of class names
        output_path: Path to save ROC curve plot
    """
    print(f"\n📈 Generating ROC curves...")
    
    n_classes = len(class_names)
    
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink',
        linestyle=':',
        linewidth=4
    )
    
    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
        color='navy',
        linestyle=':',
        linewidth=4
    )
    
    # Plot ROC curves for selected classes (to avoid clutter)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive'])
    
    # Plot top 10 classes by AUC
    sorted_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)
    for i, color in zip(sorted_classes[:10], colors):
        class_name = class_names.get(str(i), f"Class {i}")
        if len(class_name) > 30:
            class_name = class_name[:27] + "..."
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            alpha=0.5,
            label=f'Class {i}: {class_name} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multi-Class Traffic Sign Classification', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to: {output_path}")
    
    # Print AUC summary
    print(f"\n📊 ROC AUC Summary:")
    print(f"   Micro-average AUC: {roc_auc['micro']:.4f}")
    print(f"   Macro-average AUC: {roc_auc['macro']:.4f}")
    print(f"   Mean per-class AUC: {np.mean([roc_auc[i] for i in range(n_classes)]):.4f}")
    
    plt.close()
    
    return roc_auc


def plot_roc_curve_summary(y_true, y_pred_probs, class_names: dict, output_path: Path):
    """
    Generate simplified ROC curve showing only average curves
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        class_names: Dictionary of class names
        output_path: Path to save plot
    """
    print(f"\n📈 Generating ROC curve summary...")
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink',
        linestyle='-',
        linewidth=3
    )
    
    # Plot macro-average
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
        color='navy',
        linestyle='-',
        linewidth=3
    )
    
    # Plot all individual curves in light gray
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color='gray', alpha=0.1, lw=1)
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve Summary - Traffic Sign Classification\n(43 classes shown in gray)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve summary saved to: {output_path}")
    
    plt.close()



def evaluate():
    """
    Complete evaluation pipeline
    """
    print("\n" + "="*60)
    print("🚦 TRAFFIC SIGN RECOGNITION - MODEL EVALUATION")
    print("="*60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print(f"Please train the model first by running: python -m src.train")
        return
    
    # Load model
    model = load_trained_model(MODEL_PATH)
    
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    
    # Load test dataset
    test_ds, test_class_names = create_test_dataset(str(TEST_DIR))
    
    # Evaluate model (now returns probabilities too)
    test_loss, test_accuracy, y_true, y_pred, y_pred_probs = evaluate_model_on_test_set(model, test_ds)
    
    # Generate confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)
    
    # Generate classification report
    generate_classification_report(y_true, y_pred, class_names, CLASSIFICATION_REPORT_PATH)
    
    # Generate ROC curves (detailed version)
    roc_detailed_path = OUTPUT_DIR / "roc_curve_detailed.png"
    plot_roc_curve_multiclass(y_true, y_pred_probs, class_names, roc_detailed_path)
    
    # Generate ROC curve summary (simplified version)
    roc_summary_path = OUTPUT_DIR / "roc_curve_summary.png"
    plot_roc_curve_summary(y_true, y_pred_probs, class_names, roc_summary_path)
    
    # Print per-class accuracy
    print_per_class_accuracy(y_true, y_pred, class_names, top_n=10)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📁 GENERATED FILES")
    print(f"{'='*60}")
    print(f"✓ Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print(f"✓ Classification Report: {CLASSIFICATION_REPORT_PATH}")
    print(f"✓ ROC Curve (Detailed): {roc_detailed_path}")
    print(f"✓ ROC Curve (Summary): {roc_summary_path}")
    print(f"{'='*60}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"\n💡 Next step:")
    print(f"   Launch Streamlit app: streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluate()
