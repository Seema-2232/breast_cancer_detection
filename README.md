🧬 Breast Cancer Detection using Deep Learning (CNN)
📌 Overview

This project implements a Convolutional Neural Network (CNN) model to classify breast cancer histopathology images as Malignant or Benign.
The system performs end-to-end processing including image preprocessing, model training, evaluation, and deployment using Streamlit.

🎯 Problem Statement

Early detection of breast cancer significantly improves survival rates.
This project aims to build an automated deep learning system that predicts tumor type from histopathology images.

📂 Dataset

Dataset: IDC Histopathology Images

Image type: RGB microscopic tissue images

Classes:

0 → Benign

1 → Malignant

(Dataset not uploaded due to size constraints.)

🧠 Model Architecture

Convolutional Layers (Conv2D)

MaxPooling Layers

Batch Normalization

Dropout (Regularization)

Fully Connected Dense Layers

Output Layer (Sigmoid activation)

Loss Function: Binary Crossentropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Scikit-learn

Streamlit

📊 Model Performance

Training Accuracy: ~XX%

Validation Accuracy: ~XX%

Confusion Matrix included

ROC-AUC Score: ~XX

(Replace with your actual results)

📁 Project Structure
Breast-Cancer-Detection/
│
├── dataset/                # Not uploaded (large size)
├── models/
│   └── breast_cancer_model.h5
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
🚀 How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Streamlit App
streamlit run app.py
🖥 Application Features

Upload histopathology image

Real-time cancer prediction

Probability score output

Risk interpretation

Clean UI interface
📌 Future Improvements

Transfer Learning (ResNet50 / EfficientNet)

Model Explainability (Grad-CAM)

Deployment on Streamlit Cloud

Clinical validation with larger datasets

👩‍💻 Author

Seema Karki
B.Tech AIML | Deep Learning Enthusiast
Focused on real-world AI applications in healthcare.
