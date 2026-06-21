Parkinson's Disease Prediction using Deep Learning
This project aims to predict Parkinson's Disease using deep learning models trained on both
hand-drawn images (spiral and wave) and audio-based features. A user-friendly Streamlit
web app has also been developed for real-time prediction. Project Structure

├── Training/
│ ├── spiral/
│ │ ├── healthy/
│ │ └── parkinsons/
│ ├── wave/
│ │ ├── healthy/
│ │ └── parkinsons/
├── Testing/
│ ├── healthy/
│ └── parkinsons/
├── Validation/
│ ├── healthy/
│ └── parkinsons/
├── models/
│ ├── vgg16_model.h5
│ ├── resnet50_model.h5
│ ├── densenet121_model.h5
│ └── cnn_model.h5
├── app.py
├── audio_model.py
├── requirements.txt
└── README.md

Features
--> Image-based classification using:
o VGG16
o ResNet50
o DenseNet121
o Custom CNN

Dataset Details
Image Dataset -- Train Set: Spiral and wave drawings (healthy/parkinsons) -- Test Set: Separate spiral and wave images (healthy/parkinsons) -- Validation Set: For model evaluation
Audio Dataset -- Extracted acoustic features from patient voice recordings. -- Used to train models separately and select the best performer. Models and Accuracy
Model Dataset Type Accuracy (approx)
VGG16 Image ~92%
ResNet50 Image ~90%
DenseNet121 Image ~93%
CNN (Custom) Image ~88%
Best Audio Model Audio ~95%
How to Run
1. Install Requirements
pip install -r requirements.txt
2. Run Streamlit App
streamlit run GR18_AD_24-25_Streamlit_validator_parkisons.exe.py
3. Predict Parkinson's Disease
--> Upload spiral or wave image
--> The app will return whether the person is healthy or has Parkinson's
Model Training
Image Models
Trained using:  Data augmentation
--> Cross-validation
--> Categorical labels: 0 (Healthy), 1 (Parkinson's)
Audio Model  Features extracted: MFCC, Chroma, Zero-Crossing Rate, etc.  Used classification models like SVM, Random Forest, etc.  Selected based on performance metrics
