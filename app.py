"""
Traffic Sign Recognition - Streamlit Web Application
Interactive web interface for traffic sign classification
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import pandas as pd

from src.config import MODEL_PATH, CLASS_NAMES_PATH, FEEDBACK_DIR
from src.predict import predict_image

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# HELPER FUNCTIONS
# ========================================
@st.cache_resource
def load_model_and_classes():
    """Load model and class names (cached)"""
    try:
        from tensorflow import keras
        
        if not MODEL_PATH.exists():
            return None, None, "Model file not found. Please train the model first."
        
        if not CLASS_NAMES_PATH.exists():
            return None, None, "Class names file not found. Please train the model first."
        
        model = keras.models.load_model(MODEL_PATH)
        
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        
        return model, class_names, None
    
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def save_feedback(predicted_class: str, confidence: float, correct_class: str = None, was_correct: bool = None):
    """Save user feedback to CSV"""
    try:
        feedback_file = FEEDBACK_DIR / "feedback.csv"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create feedback entry
        feedback_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.4f}",
            'was_correct': str(was_correct) if was_correct is not None else "",
            'correct_class': correct_class if correct_class else ""
        }
        
        # Append to CSV
        df = pd.DataFrame([feedback_data])
        
        if feedback_file.exists():
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_file, mode='w', header=True, index=False)
        
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


# ========================================
# SIDEBAR
# ========================================
def render_sidebar():
    """Render sidebar content"""
    st.sidebar.title("ℹ️ Project Information")
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Model Details")
    st.sidebar.markdown("""
    - **Model**: Convolutional Neural Network (CNN)
    - **Framework**: TensorFlow / Keras
    - **Task**: Multi-class Classification
    - **Input Size**: 64 × 64 × 3 (RGB)
    """)
    
    # Load class info
    model, class_names, error = load_model_and_classes()
    if class_names:
        st.sidebar.markdown(f"- **Classes**: {len(class_names)}")
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🏗️ Architecture")
    st.sidebar.markdown("""
    The CNN architecture consists of:
    - **4 Convolutional Blocks** with BatchNormalization
    - **MaxPooling** layers for spatial reduction
    - **Dropout** layers for regularization
    - **Global Average Pooling**
    - **Dense layers** with softmax activation
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("✨ Enhanced Features")
    st.sidebar.markdown("""
    **NEW Preprocessing:**
    - 🎯 Auto sign detection & cropping
    - 🔍 Color-based isolation (HSV)
    - ✨ Image enhancement (sharpness, contrast)
    - 🖼️ High-quality LANCZOS resampling
    - 🔄 Test-Time Augmentation (optional)
    
    **Handles:**
    - Multiple signs in frame
    - Angled/tilted signs
    - Complex backgrounds
    - Poor lighting conditions
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📚 About")
    st.sidebar.markdown("""
    This system uses deep learning to classify traffic signs from images.
    
    **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)
    
    **Features**:
    - Real-time prediction
    - Top-3 confidence scores
    - User feedback system
    - Real-world image support
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with ❤️ using TensorFlow & Streamlit")


# ========================================
# MAIN CONTENT
# ========================================
def render_home():
    """Render home page"""
    st.markdown('<div class="main-header">🚦 Traffic Sign Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning-Based Traffic Sign Classification</div>', unsafe_allow_html=True)
    
    # Enhanced features banner
    st.markdown("""
    <div class="info-banner">
        ✨ <strong>Enhanced with AI Preprocessing</strong> - Now handles real-world images with multiple signs, angles, and complex backgrounds!
    </div>
    """, unsafe_allow_html=True)
    
    # Check if model is available
    model, class_names, error = load_model_and_classes()
    
    if error:
        st.markdown(f'<div class="warning-box">⚠️ <strong>Warning:</strong> {error}</div>', unsafe_allow_html=True)
        st.info("📝 To train the model, run: `python -m src.train`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.markdown("---")
    st.subheader("📤 Upload Traffic Sign Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a traffic sign image for classification"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Traffic Sign", use_container_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
            st.caption(f"Format: {image.format if image.format else 'Unknown'}")
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                auto_crop = st.checkbox("Auto-detect and crop sign", value=True, 
                                       help="Automatically detect and crop the main traffic sign")
                enhance = st.checkbox("Enhance image quality", value=True,
                                     help="Apply sharpness, contrast, and color enhancement")
                use_tta = st.checkbox("Use Test-Time Augmentation (slower)", value=False,
                                     help="Average predictions over multiple augmentations for better accuracy")
            
            # Predict button
            if st.button("🔮 Predict Traffic Sign", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = Path("temp_upload.png")
                        image.save(temp_path)
                        
                        # Make prediction with enhanced preprocessing
                        result = predict_image(
                            str(temp_path), 
                            model, 
                            class_names,
                            auto_crop=auto_crop,
                            enhance=enhance,
                            use_tta=use_tta
                        )
                        
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                        
                        # Store results in session state
                        st.session_state['prediction_result'] = result
                        st.session_state['uploaded_image'] = image
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        import traceback
                        error_details = traceback.format_exc()
                        with st.expander("🔍 Error Details"):
                            st.code(error_details)
        
        # Display prediction results if available
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            stored_image = st.session_state.get('uploaded_image', image)
            
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            # Main prediction
            st.markdown(f"### 🚦 Predicted Sign")
            st.markdown(f"## **{result['predicted_class_name']}**")
            
            # Confidence score with interpretation
            confidence = result['confidence'] * 100
            st.markdown(f"### 🎯 Confidence Score: **{confidence:.2f}%**")
            
            # Confidence interpretation with color
            if confidence > 90:
                st.success("✓ Very High Confidence")
            elif confidence > 70:
                st.info("✓ High Confidence")
            elif confidence > 50:
                st.warning("⚠️ Moderate Confidence")
            else:
                st.error("❌ Low Confidence")
            
            # Progress bar for confidence
            st.progress(result['confidence'])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 3 predictions
            st.markdown("---")
            st.subheader("📊 Top 3 Predictions")
            
            for i, pred in enumerate(result['top_predictions'], 1):
                conf_percent = pred['confidence'] * 100
                
                with st.container():
                    col_rank, col_name, col_conf = st.columns([0.5, 3, 1.5])
                    
                    with col_rank:
                        if i == 1:
                            st.markdown("🥇")
                        elif i == 2:
                            st.markdown("🥈")
                        else:
                            st.markdown("🥉")
                    
                    with col_name:
                        st.markdown(f"**{pred['class_name']}**")
                    
                    with col_conf:
                        st.markdown(f"`{conf_percent:.2f}%`")
                    
                    st.progress(pred['confidence'])
            
            # Feedback section
            st.markdown("---")
            
            # Action buttons
            col_feedback, col_new = st.columns(2)
            
            with col_feedback:
                st.subheader("💬 Feedback")
            
            with col_new:
                if st.button("🔄 New Prediction", type="secondary", use_container_width=True):
                    # Clear session state
                    if 'prediction_result' in st.session_state:
                        del st.session_state['prediction_result']
                    if 'uploaded_image' in st.session_state:
                        del st.session_state['uploaded_image']
                    if 'show_feedback_form' in st.session_state:
                        del st.session_state['show_feedback_form']
                    st.rerun()
            
            st.markdown("Help us improve by providing feedback on this prediction:")
            
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("✅ Correct Prediction", use_container_width=True):
                    if save_feedback(
                        predicted_class=result['predicted_class_name'],
                        confidence=result['confidence'],
                        was_correct=True
                    ):
                        st.success("Thank you for your feedback!")
            
            with col_no:
                if st.button("❌ Incorrect Prediction", use_container_width=True):
                    st.session_state['show_feedback_form'] = True
            
            # Feedback form for incorrect predictions
            if st.session_state.get('show_feedback_form', False):
                with st.form("feedback_form"):
                    st.write("Please tell us the correct traffic sign:")
                    
                    # Get all class names for dropdown
                    class_options = [class_names[str(i)] for i in range(len(class_names))]
                    correct_class = st.selectbox("Correct Traffic Sign:", class_options)
                    
                    submitted = st.form_submit_button("Submit Feedback")
                    
                    if submitted:
                        if save_feedback(
                            predicted_class=result['predicted_class_name'],
                            confidence=result['confidence'],
                            was_correct=False,
                            correct_class=correct_class
                        ):
                            st.success("Thank you for your feedback! This will help improve the model.")
                            st.session_state['show_feedback_form'] = False


def render_model_info():
    """Render model information page"""
    st.title("📊 Model Information")
    
    model, class_names, error = load_model_and_classes()
    
    if error:
        st.error(error)
        return
    
    st.markdown("---")
    
    # Model statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Classes", len(class_names))
    
    with col2:
        st.metric("Input Size", "64 × 64 × 3")
    
    with col3:
        total_params = model.count_params()
        st.metric("Total Parameters", f"{total_params:,}")
    
    st.markdown("---")
    
    # Architecture details
    st.subheader("🏗️ Model Architecture")
    
    st.markdown("""
    ### Convolutional Neural Network (CNN)
    
    The model uses a deep CNN architecture optimized for traffic sign classification:
    
    #### Layer Structure:
    1. **Input Layer**: 64×64×3 RGB images
    2. **Conv Block 1**: 32 filters → BatchNorm → ReLU → MaxPool → Dropout
    3. **Conv Block 2**: 64 filters → BatchNorm → ReLU → MaxPool → Dropout
    4. **Conv Block 3**: 128 filters → BatchNorm → ReLU → MaxPool → Dropout
    5. **Conv Block 4**: 256 filters → BatchNorm → ReLU → MaxPool → Dropout
    6. **Global Average Pooling**: Reduces spatial dimensions
    7. **Dense Layer**: 256 neurons with BatchNorm and Dropout
    8. **Output Layer**: Softmax activation for 43 classes
    
    #### Key Features:
    - **Batch Normalization**: Stabilizes and accelerates training
    - **Dropout**: Prevents overfitting (rates: 0.25-0.4)
    - **L2 Regularization**: Improves generalization
    - **Data Augmentation**: Rotation, zoom, translation, contrast
    
    #### Training Configuration:
    - **Optimizer**: Adam (lr=0.001)
    - **Loss Function**: Sparse Categorical Crossentropy
    - **Callbacks**: Early Stopping, ReduceLROnPlateau, ModelCheckpoint
    """)
    
    st.markdown("---")
    
    # Class names
    st.subheader("🏷️ Traffic Sign Classes")
    
    st.markdown(f"The model can classify {len(class_names)} different traffic sign types:")
    
    # Display classes in a table
    class_data = []
    for idx in range(len(class_names)):
        class_data.append({
            'Class ID': idx,
            'Traffic Sign Name': class_names[str(idx)]
        })
    
    df = pd.DataFrame(class_data)
    st.dataframe(df, use_container_width=True, height=400)


def render_about():
    """Render about page"""
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Deep Learning-Based Traffic Sign Recognition and Classification
    
    ### 🎯 Project Overview
    This project implements an end-to-end deep learning solution for traffic sign recognition and classification.
    The system can identify and classify 43 different types of traffic signs with high accuracy.
    
    ### 🚀 Objectives
    - Develop a robust CNN model for traffic sign classification
    - Create an interactive web application for real-time predictions
    - Achieve high accuracy on the GTSRB benchmark dataset
    - Provide user-friendly interface for practical use
    
    ### 🛠️ Technology Stack
    - **Deep Learning**: TensorFlow / Keras
    - **Data Processing**: NumPy, Pandas, Pillow
    - **Visualization**: Matplotlib, Seaborn
    - **Web Application**: Streamlit
    - **ML Tools**: Scikit-learn
    
    ### 📊 Dataset
    The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which contains:
    - 43 different traffic sign classes
    - Thousands of images per class
    - Various lighting and weather conditions
    - Different image sizes and perspectives
    
    ### 🏗️ Project Pipeline
    ```
    Dataset → Preprocessing → Data Augmentation → CNN Model →
    Training → Validation → Testing → Evaluation → Web Application
    ```
    
    ### 🎓 Key Features
    - **High Accuracy**: Achieved through deep CNN architecture
    - **Real-time Prediction**: Fast inference on uploaded images
    - **Top-3 Predictions**: Shows confidence scores for top predictions
    - **User Feedback**: Collect feedback to improve model
    - **Professional UI**: Clean and intuitive Streamlit interface
    
    ### 📈 Model Performance
    The model is evaluated using:
    - Accuracy
    - Precision, Recall, F1-Score
    - Confusion Matrix
    - Per-class accuracy analysis
    
    ### 🔮 Future Enhancements
    - Real-time video stream classification
    - Mobile application deployment
    - Transfer learning with pre-trained models
    - Object detection for multiple signs
    - GPS integration for location-aware warnings
    - Voice alerts for drivers
    - Integration with driver assistance systems
    
    ### 👥 Use Cases
    - **Autonomous Vehicles**: Traffic sign recognition for self-driving cars
    - **Driver Assistance**: Real-time warnings and alerts
    - **Traffic Management**: Automated sign inventory and monitoring
    - **Education**: Learning tool for traffic regulations
    - **Research**: Benchmark for computer vision algorithms
    
    ### 📝 How to Use
    1. Navigate to the **Home** page
    2. Upload a traffic sign image (JPG or PNG)
    3. Click the **Predict** button
    4. View the predicted sign and confidence score
    5. Provide feedback to help improve the model
    
    ### 🙏 Acknowledgments
    - GTSRB dataset creators
    - TensorFlow and Keras teams
    - Streamlit development team
    - Open source community
    
    ---
    
    **Built with ❤️ for traffic safety and AI education**
    """)


# ========================================
# MAIN APP
# ========================================
def main():
    """Main application"""
    # Render sidebar
    render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Model Info", "ℹ️ About"])
    
    with tab1:
        render_home()
    
    with tab2:
        render_model_info()
    
    with tab3:
        render_about()


if __name__ == "__main__":
    main()
