🌿 AI Crop Disease Detection Application

An end-to-end deep learning application for identifying 22 types of crop diseases and pests from leaf images. The system combines a fine-tuned Convolutional Neural Network (CNN) for visual classification with Google’s Gemini AI to deliver clear, actionable agricultural insights.

📌 Project Overview

This project addresses the critical challenge of early crop disease diagnosis. By leveraging transfer learning with state-of-the-art deep learning architectures, the system achieves high accuracy in classifying diseases across Cashew, Cassava, Maize, and Tomato crops.

The solution is deployed as a user-friendly Streamlit web application that not only detects diseases but also provides treatment recommendations and preventive measures using Generative AI.

✨ Features

Multi-Class Classification
Accurately identifies 22 disease and healthy classes across multiple crop types.

AI-Powered Insights
Integrates Google Gemini 1.5 Flash to generate real-time disease summaries, treatment options, and prevention guidance.

Automatic Model Retrieval
Automatically downloads trained model weights from a secure external source on first run, bypassing GitHub file size limitations.

Robust Data Pipeline
Built on a reliable training pipeline that includes data cleaning, augmentation, and validation to ensure consistent model performance.

Interactive User Interface
Clean, responsive, and easy-to-use interface developed with Streamlit.

🛠️ Technologies Used

Frontend: Streamlit

Deep Learning: TensorFlow / Keras

Model Architectures: ResNet50 (Fine-tuned), DenseNet121

Generative AI: Google Gemini API

Image Processing: Pillow (PIL), NumPy

Training Environment: Google Colab (GPU)

🔬 Model Development

The model was developed using a systematic and research-driven approach:

Data Cleaning
Implemented a robust pipeline using TensorFlow decoding to remove corrupted images from a 1.35GB Kaggle dataset.

Architecture Search
Evaluated multiple architectures including ANN, Custom CNN, VGG16, ResNet50, EfficientNetB3, and DenseNet121.

Optimization
Applied transfer learning with a two-stage fine-tuning process, along with data augmentation, L2 regularization, and dropout to reduce overfitting.

Performance
Achieved a test accuracy of 85.62% on an unseen test set, demonstrating strong generalization.

💡 Future Improvements

Object Detection
Integrate YOLOv8 to localize infected regions on leaf images.

Mobile Deployment
Convert the model to TensorFlow Lite (TFLite) for offline Android and iOS applications.

Multilingual Support
Use Gemini to translate disease insights and recommendations into local languages for wider accessibility.
