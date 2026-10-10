"""
CNN Model Architecture Module
Defines and builds the Convolutional Neural Network for traffic sign classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple

from src.config import (
    INPUT_SHAPE, DROPOUT_RATE, L2_REGULARIZATION,
    LEARNING_RATE, OPTIMIZER
)


def build_cnn_model(num_classes: int, input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> keras.Model:
    """
    Build CNN model for traffic sign classification
    
    Architecture:
        - Input Layer
        - Conv Block 1: 32 filters
        - Conv Block 2: 64 filters
        - Conv Block 3: 128 filters
        - Conv Block 4: 256 filters
        - Global Average Pooling
        - Dense Layer with Dropout
        - Output Layer
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    
    # L2 regularizer
    l2_reg = regularizers.l2(L2_REGULARIZATION)
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Activation('relu', name='relu1_1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Activation('relu', name='relu1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Activation('relu', name='relu2_1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Activation('relu', name='relu2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Activation('relu', name='relu3_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.Activation('relu', name='relu3_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # Block 4: 256 filters
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Activation('relu', name='relu4_1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2_reg, name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.Activation('relu', name='relu4_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(0.3, name='dropout4')(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers
    x = layers.Dense(256, kernel_regularizer=l2_reg, name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Activation('relu', name='relu_dense1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_dense')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='TrafficSignCNN')
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = LEARNING_RATE,
    optimizer_name: str = OPTIMIZER
) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer ('adam', 'sgd', etc.)
        
    Returns:
        Compiled Keras model
    """
    
    # Choose optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_model(num_classes: int) -> keras.Model:
    """
    Create and compile complete CNN model
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model ready for training
    """
    print(f"\n{'='*60}")
    print(f"🏗️  BUILDING CNN MODEL")
    print(f"{'='*60}")
    
    # Build model
    model = build_cnn_model(num_classes)
    
    # Compile model
    model = compile_model(model)
    
    print(f"✓ Model built successfully!")
    print(f"✓ Total classes: {num_classes}")
    print(f"✓ Input shape: {INPUT_SHAPE}")
    print(f"✓ Optimizer: {OPTIMIZER}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    
    return model


def display_model_summary(model: keras.Model):
    """
    Display model architecture summary
    
    Args:
        model: Keras model
    """
    print(f"\n{'='*60}")
    print(f"📋 MODEL ARCHITECTURE")
    print(f"{'='*60}")
    
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print(f"📊 MODEL PARAMETERS")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Create model with 43 classes (GTSRB standard)
    test_model = create_model(num_classes=43)
    
    # Display summary
    display_model_summary(test_model)
    
    print("✓ Model creation test complete!")
