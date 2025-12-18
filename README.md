# 👗 Fashion Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Model-ResNet50-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Overview
This project implements a **Fashion Product Recommendation System** using a Convolutional Neural Network (CNN). It utilizes the pre-trained **ResNet50** model to extract features from a dataset of 44,000 fashion product images and uses the **Nearest Neighbors** algorithm to recommend similar products based on Euclidean distance.

The project includes an interactive web application built with **Streamlit**, allowing users to upload an image and receive visually similar fashion recommendations instantly.

### 🚀 Key Features
* **Deep Learning Feature Extraction:** Uses ResNet50 (trained on ImageNet) to generate embeddings for fashion items.
* **Similarity Search:** Utilizes k-Nearest Neighbors (k-NN) to find the closest matches in the vector space.
* **Interactive UI:** A user-friendly Streamlit web interface for easy image uploading and recommendation visualization.

---

## 📂 Dataset
The dataset consists of **44,000 fashion product images** sourced from Kaggle.
* **Source:** [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
* **Preprocessing:** All images are resized to `224x224` pixels to match the input requirements of ResNet50.

> **Note:** Due to the large size of the dataset, images are not hosted in this repository. You must download them locally to run the project.

---

## 🛠️ Tech Stack & Prerequisites
Ensure you have **Python 3.8+** installed. The project relies on the following libraries:

* **Deep Learning:** `tensorflow`, `keras`
* **Computer Vision:** `pillow`, `opencv-python`
* **Machine Learning:** `scikit-learn`, `numpy`, `pandas`
* **Web Framework:** `streamlit`

### Install Dependencies
Create a `requirements.txt` file and run:
```bash
pip install -r requirements.txt

```

*Recommended `requirements.txt` content:*

```text
streamlit
tensorflow
scikit-learn
numpy
pillow
tqdm

```

---

## 🧠 Model Details

### 1. Feature Extractor (ResNet50)

* **Architecture:** ResNet50 (pre-trained on ImageNet).
* **Input Shape:** `(224, 224, 3)`
* **Modifications:** The top classification layer is removed (`include_top=False`), and a `GlobalMaxPool2D` layer is added to reduce feature dimensions to a 1D vector.
* **Output:** A high-dimensional feature vector representing the visual style of the image.

### 2. Similarity Search (Nearest Neighbors)

* **Algorithm:** Brute-force Search
* **Metric:** Euclidean Distance
* **Neighbors:** Returns top 5-6 most similar images.

---

## ⚙️ How to Run Locally

Since this project requires the full image dataset, it is designed for **local execution**.

### Step 1: Clone the Repository

```bash
git clone [https://github.com/your-username/fashion-recommendation-system.git](https://github.com/your-username/fashion-recommendation-system.git)
cd fashion-recommendation-system

```

### Step 2: Setup Data

1. Download the **Fashion Product Images Dataset** from Kaggle.
2. Extract the images into a folder named `images/` inside the project directory.
3. Ensure you have the pre-computed feature files: `Images_features.pkl` and `filenames.pkl`. (If not, run the training script to generate them).

### Step 3: Run the App

Execute the Streamlit application:

```bash
streamlit run app.py

```

### Step 4: Use the App

1. A web interface will open in your browser (usually at `http://localhost:8501`).
2. Upload an image of a shirt, shoe, or accessory.
3. The system will display the **Top 5 Recommended Products** from the database.

---

## 📸 Workflow

1. **User Upload:** User uploads an image via the Streamlit UI.
2. **Preprocessing:** Image is resized and normalized.
3. **Feature Extraction:** ResNet50 converts the image into a numeric vector.
4. **Retrieval:** k-NN finds the nearest vectors in the database.
5. **Display:** The corresponding images for those vectors are shown to the user.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🔗 References

* Dataset: [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
* Libraries: TensorFlow, Scikit-learn, Streamlit

