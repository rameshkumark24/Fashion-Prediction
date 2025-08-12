Fashion Product Recommendation System

Overview

This project implements a Fashion Product Recommendation System using a Convolutional Neural Network (CNN) based on the pre-trained ResNet50 model. The system extracts features from a dataset of 44,000 fashion product images and uses the Nearest Neighbors algorithm to recommend similar products based on Euclidean distance. The project also includes a web application for deploying the recommendation system.

The workflow involves:


Extracting features from images using ResNet50.


Using Nearest Neighbors to find similar images.


Deploying the model as a web app for user interaction.

Dataset

The dataset used in this project consists of 44,000 fashion product images sourced from Kaggle.

Link: Fashion Product Images Dataset

The images are resized to 224x224 pixels to match the input requirements of ResNet50.

Prerequisites

Ensure you have the following installed:

Python 3.8+

Required Python libraries (listed below)

Model Details

ResNet50: A pre-trained CNN model from TensorFlow, used for feature extraction.

Weights: imagenet

Top layer removed (include_top=False)

Input shape: (224, 224, 3)

Additional layer: GlobalMaxPool2D to reduce feature dimensions.

Nearest Neighbors: Uses sklearn.neighbors.NearestNeighbors with:

n_neighbors=6

Algorithm: brute

Metric: euclidean

How It Works

Feature Extraction:

Images are resized to 224x224 and preprocessed using preprocess_input.

ResNet50 extracts feature vectors, which are flattened and normalized.

Recommendation:

Normalized features are fed into the Nearest Neighbors algorithm to find the top 6 similar images based on Euclidean distance.

Web App:

Users upload an image, and the system recommends visually similar fashion products.

Deployment

The web app is built using a framework like Flask or Streamlit (modify based on your implementation).

To deploy:

Ensure all dependencies are installed.

Run app.py and access the app via a web browser.

For production, consider deploying on platforms like Heroku, AWS, or Vercel.

References

Dataset: Kaggle Fashion Product Images

Code Inspiration: Google Drive Link

Libraries: TensorFlow, scikit-learn, NumPy, Pillow

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License

This project is licensed under the MIT License. See the LICENSE file for details.



A web browser for running the web application
