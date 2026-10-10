"""
Image Preprocessing and Data Augmentation Module
Handles image loading, preprocessing, and augmentation for training
"""

import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from src.config import (
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    AUGMENTATION_ROTATION, AUGMENTATION_ZOOM, 
    AUGMENTATION_TRANSLATION, VALIDATION_SPLIT
)


def create_data_augmentation() -> keras.Sequential:
    """
    Create data augmentation pipeline for training
    
    Returns:
        Sequential model with augmentation layers
    """
    augmentation = keras.Sequential([
        keras.layers.RandomRotation(
            AUGMENTATION_ROTATION,
            seed=RANDOM_SEED
        ),
        keras.layers.RandomZoom(
            AUGMENTATION_ZOOM,
            seed=RANDOM_SEED
        ),
        keras.layers.RandomTranslation(
            height_factor=AUGMENTATION_TRANSLATION,
            width_factor=AUGMENTATION_TRANSLATION,
            seed=RANDOM_SEED
        ),
        keras.layers.RandomContrast(
            factor=0.2,
            seed=RANDOM_SEED
        ),
    ], name='data_augmentation')
    
    return augmentation


def load_dataset_from_directory(
    directory: str,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    shuffle: bool = True,
    seed: int = RANDOM_SEED
) -> tf.data.Dataset:
    """
    Load dataset from directory using TensorFlow
    
    Args:
        directory: Path to dataset directory
        batch_size: Batch size for loading
        image_size: Target image size (height, width)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducibility
        
    Returns:
        TensorFlow Dataset
    """
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',  # Integer labels for sparse_categorical_crossentropy
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        interpolation='bilinear'
    )
    
    return dataset


def prepare_dataset(
    dataset: tf.data.Dataset,
    augment: bool = False,
    cache: bool = True,
    shuffle_buffer: int = 1000
) -> tf.data.Dataset:
    """
    Prepare dataset with normalization, caching, and prefetching
    
    Args:
        dataset: Input TensorFlow dataset
        augment: Whether to apply data augmentation
        cache: Whether to cache the dataset
        shuffle_buffer: Buffer size for shuffling
        
    Returns:
        Prepared TensorFlow Dataset
    """
    # Normalization layer (0-255 to 0-1)
    normalization = keras.layers.Rescaling(1./255)
    
    # Apply normalization
    dataset = dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation for training
    if augment:
        augmentation = create_data_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Cache for performance
    if cache:
        dataset = dataset.cache()
    
    # Shuffle for training
    if augment:  # Only shuffle training data
        dataset = dataset.shuffle(shuffle_buffer)
    
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def create_train_val_datasets(
    train_directory: str,
    validation_split: float = VALIDATION_SPLIT,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = RANDOM_SEED
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """
    Create training and validation datasets with split
    
    Args:
        train_directory: Path to training data directory
        validation_split: Fraction of data to use for validation
        batch_size: Batch size
        image_size: Target image size
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, class_names)
    """
    print(f"\n{'='*60}")
    print(f"📦 LOADING DATASETS")
    print(f"{'='*60}")
    
    # Load training dataset
    train_ds = keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training',
        interpolation='bilinear'
    )
    
    # Load validation dataset
    val_ds = keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='validation',
        interpolation='bilinear'
    )
    
    # Get class names
    class_names = train_ds.class_names
    
    print(f"✓ Training batches: {len(train_ds)}")
    print(f"✓ Validation batches: {len(val_ds)}")
    print(f"✓ Number of classes: {len(class_names)}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Image size: {image_size}")
    
    # Prepare datasets
    print(f"\n🔧 Preparing datasets with augmentation and optimization...")
    train_ds = prepare_dataset(train_ds, augment=True)
    val_ds = prepare_dataset(val_ds, augment=False)
    
    print(f"✓ Datasets prepared successfully!")
    
    return train_ds, val_ds, class_names


def create_test_dataset(
    test_directory: str,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[tf.data.Dataset, list]:
    """
    Create test dataset
    
    Args:
        test_directory: Path to test data directory
        batch_size: Batch size
        image_size: Target image size
        
    Returns:
        Tuple of (test_dataset, class_names)
    """
    print(f"\n📦 Loading test dataset...")
    
    test_ds = keras.utils.image_dataset_from_directory(
        test_directory,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,  # Don't shuffle test data
        interpolation='bilinear'
    )
    
    class_names = test_ds.class_names
    
    print(f"✓ Test batches: {len(test_ds)}")
    print(f"✓ Number of classes: {len(class_names)}")
    
    # Prepare test dataset (no augmentation)
    test_ds = prepare_dataset(test_ds, augment=False)
    
    return test_ds, class_names


if __name__ == "__main__":
    from src.config import TRAIN_DIR, TEST_DIR
    
    print("Testing preprocessing pipeline...")
    
    # Test training/validation split
    train_ds, val_ds, class_names = create_train_val_datasets(str(TRAIN_DIR))
    print(f"\nClass names: {class_names[:5]}... (showing first 5)")
    
    # Test data augmentation
    augmentation = create_data_augmentation()
    print(f"\nData augmentation layers: {len(augmentation.layers)}")
