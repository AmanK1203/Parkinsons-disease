# 🧠 Parkinson's Disease Prediction Using Deep Learning

## 📌 Project Overview

This project focuses on predicting Parkinson's Disease using Deep Learning and Machine Learning techniques.

The system analyzes two different types of data:

- 🖼️ Hand-drawn spiral and wave images
- 🎙️ Audio-based acoustic features extracted from patient voice recordings

For image-based prediction, multiple deep learning architectures were trained and evaluated, including VGG16, ResNet50, DenseNet121, and a Custom CNN.

An interactive web application was also developed using Streamlit to provide a user-friendly interface for making predictions.

---

## 🎯 Project Objective

The primary objective of this project is to develop an automated machine learning system that can assist in identifying patterns associated with Parkinson's Disease from:

1. Hand-drawn spiral and wave images
2. Voice/audio-based features

The project compares different models and evaluates their performance to identify suitable models for prediction.

> **Note:** This project is intended for educational and research purposes and should not be considered a medical diagnostic system.

---

# 🚀 Key Features

### 🖼️ Image-Based Parkinson's Detection

The system classifies hand-drawn images into:

- Healthy
- Parkinson's

The image-based models include:

- VGG16
- ResNet50
- DenseNet121
- Custom CNN

### 🎙️ Audio-Based Prediction

The project also analyzes acoustic features extracted from patient voice recordings.

Features include:

- MFCC
- Chroma
- Zero-Crossing Rate
- Other extracted acoustic features

Classification models such as:

- SVM
- Random Forest
- Other classification approaches

were evaluated based on their performance.

### 🌐 Streamlit Application

A Streamlit-based application provides a user-friendly interface for real-time prediction.

Users can upload a spiral or wave drawing and receive a model prediction.

---

# 🏗️ Project Architecture

```text
                    Parkinson's Disease Prediction
                              │
              ┌───────────────┴───────────────┐
              │                               │
        Image-Based Data                 Audio-Based Data
              │                               │
       Spiral / Wave Images             Voice Recordings
              │                               │
              ▼                               ▼
       Image Preprocessing             Feature Extraction
              │                               │
              ▼                               ├── MFCC
        Data Augmentation               ├── Chroma
              │                         └── Zero-Crossing Rate
              ▼                               │
       Deep Learning Models                    ▼
              │                         ML Classification
      ┌───────┼────────┐                       │
      │       │        │                       │
    VGG16  ResNet50 DenseNet121                │
      │       │        │                       │
      └───────┼────────┘                       │
              │                                │
              ▼                                ▼
          Prediction                       Prediction
              │                                │
              └──────────────┬─────────────────┘
                             ▼
                      Streamlit Application
