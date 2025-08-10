🐟 Multiclass Fish Image Classification
📌 Overview
This project implements multiclass fish species classification using deep learning in TensorFlow/Keras.
We train both CNN models from scratch and transfer learning architectures, evaluate their performance, and deploy the best model via a Streamlit web application for real-time predictions.

🎯 Key Skills & Technologies
Python

TensorFlow / Keras (Deep Learning)

CNN & Transfer Learning (EfficientNet, VGG16, ResNet50, MobileNet, InceptionV3)

Data Augmentation & Preprocessing

Model Evaluation & Visualization

Streamlit (Deployment)

📂 Project Structure


── data/                               # Dataset (train, val, test folders)
── fish_classification.ipynb           # Model training & evaluation
── streamlit_fish_app.py                # Streamlit web app for predictions
── efficientnet_fish_classifier_final.h5# Best trained model
── requirements.txt                     # Python dependencies
── README.md                            # Project documentation
🧩 Problem Statement
Given an input image of a fish, predict its species from multiple predefined categories.
The goal is to:

Compare multiple architectures to determine the best model.

Deploy the model in a user-friendly prediction interface.

💼 Business Use Cases
Fisheries Management: Automate fish species recognition.

Marine Research: Speed up marine biodiversity analysis.

Food Industry: Automate sorting in seafood processing plants.

🛠 Methodology
1️⃣ Data Preprocessing
Images loaded using ImageDataGenerator.

Normalized to range [0, 1].

Applied augmentation: rotation, zoom, flipping.

2️⃣ Model Training & Fine-Tuning
CNN from scratch.

Transfer learning with:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0 (best performing).

Fine-tuned layers for improved accuracy.

3️⃣ Model Evaluation
Metrics: Accuracy, Precision, Recall, F1-score.

Confusion matrix visualization.

Tracked training/validation accuracy & loss curves.

4️⃣ Deployment (Streamlit)
Loads efficientnet_fish_classifier_final.h5.

Accepts .jpg, .jpeg, .png uploads.

Predicts fish species & confidence score.

Displays per-class probability scores.

📊 Classes in Model
The model predicts the following categories:


1. fish sea_food trout
2. fish sea_food striped_red_mullet
3. fish sea_food shrimp
4. fish sea_food sea_bass
5. fish sea_food red_sea_bream
6. fish sea_food red_mullet
7. fish sea_food hourse_mackerel
8. fish sea_food gilt_head_bream
9. fish sea_food black_sea_sprat
10. animal fish
11. animal fish bass
📦 Installation & Usage
🔹 Clone the Repository

git clone https://github.com/yourusername/fish-image-classification.git
cd fish-image-classification
🔹 Install Dependencies

pip install -r requirements.txt
🔹 Run the Streamlit App

streamlit run streamlit_fish_app.py

📁 Dataset
Images organized by species in separate folders.

Place dataset under data/train, data/val, data/test.

Loaded with ImageDataGenerator for efficient preprocessing.

🚀 Deliverables
✅ Trained EfficientNetB0 model (.h5).

✅ Streamlit prediction app.

✅ Training notebook with experiments & evaluation.

✅ Performance comparison report.

