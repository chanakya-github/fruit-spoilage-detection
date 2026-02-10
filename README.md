
# 🍎 Food Spoilage Detection using Machine Learning & Deep Learning

## 📌 Project Overview

Food spoilage is a major concern for food safety and quality control. Spoiled food often shows **visible changes** such as discoloration, texture variation, and surface degradation due to microbial activity.

This project demonstrates an **image-based food spoilage detection system** using:

* **Traditional Machine Learning** (for a lightweight, interpretable, deployable solution)
* **Deep Learning (CNN)** (for higher accuracy and complex pattern learning)

The system classifies food images into freshness stages using visual cues extracted from images.
The project is designed as an **academic proof-of-concept** and a **learning-focused end-to-end ML system**.

---

## 🧠 Project Approaches

### 1️⃣ Traditional Machine Learning Approach (Primary & Deployed)

This is the **main implementation** of the project and is fully deployed as a web application.

#### 🔹 Pipeline Steps

* Image loading and resizing (100 × 100)
* Feature extraction using **mean RGB color values**
* Model training and evaluation
* Model selection based on performance
* Model deployment using **Streamlit**

#### 🔹 Models Evaluated

* Logistic Regression
* Support Vector Machine (RBF kernel)
* **Random Forest (selected)**

📌 **Why Random Forest?**
Random Forest achieved the best performance and can model **non-linear relationships** between color features and spoilage patterns while remaining interpretable and efficient.

#### 🔹 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score
* 5-fold Cross-Validation

The trained Random Forest model is saved and reused for real-time predictions in the web app.

---

### 2️⃣ Deep Learning Approach (CNN – Experimental Module)

A **Convolutional Neural Network (CNN)** was implemented separately to explore whether deep learning can improve performance over traditional ML.

📌 **Important Note**

* The CNN was trained and tested in **Google Colab** due to higher computational requirements
* This module is included for **academic comparison and learning purposes**
* It is **not integrated into the deployed web app**

#### 🔹 Key Highlights

* Learns features automatically from images
* Captures complex visual patterns (texture, spots, surface changes)
* Achieved approximately **96% accuracy**
* Demonstrates why CNNs are better suited for image-heavy problems

📁 The CNN notebook is available in the `deep_learning/` folder.

---

## 🖥️ Web Application (Streamlit)

A simple and user-friendly **Streamlit web app** demonstrates how the ML model can be used in practice.

### 🔹 Features

* Upload a food image
* Automatic image quality check (lighting validation)
* Image preprocessing
* Freshness prediction using ML model
* Confidence score display
* Freshness stages:

  * Fresh
  * Semi-spoiled (probability-based)
  * Spoiled

📌 Although the model is binary, **probability thresholds** are used to introduce an intermediate freshness stage.

---

## 🧪 Dataset Information

* Image dataset sourced from **Kaggle**
* Classes:

  * Fresh
  * Rotten
* Dataset is **not included** in this repository due to size and licensing constraints
* Folder-based structure used for supervised learning

The dataset used for CNN experiments is similar in nature and serves as an approximation of smartphone-captured food images.

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* scikit-learn
* TensorFlow / Keras
* Streamlit
* Google Colab

---

## 📊 Results Summary

| Approach                       | Accuracy |
| ------------------------------ | -------- |
| Traditional ML (Random Forest) | ~92%     |
| Deep Learning (CNN)            | ~96%     |

✔ CNN achieved higher accuracy
✔ ML approach remains faster, interpretable, and easier to deploy

---

## ⚠️ Limitations

* Image-based detection only
* Cannot directly detect microorganisms or chemical properties
* Performance depends on lighting and image quality
* Limited food categories
* Intended as a **supporting tool**, not a replacement for laboratory testing

---

## 🚀 Future Improvements

* Multi-class freshness detection (Fresh / Semi-spoiled / Spoiled)
* Integration with microbiological or pH data
* Mobile application development
* Real-time camera-based detection
* Cloud deployment for scalability

---

## 🎓 Academic & Internship Context

This project showcases:

* End-to-end ML system development
* Model comparison and selection
* Practical deployment using Streamlit
* Deep learning experimentation and evaluation
* Engineering decision-making under computational constraints

It reflects a **progressive learning approach** from traditional ML to deep learning.

---

## 👤 Author

**Chanakya**
Computer Science Student
Aspiring Machine Learning Engineer

GitHub: [https://github.com/chanakya-github](https://github.com/chanakya-github)

---

