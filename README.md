# 😷 Face Mask Detection using CNN

This project is a deep learning-based face mask detection system that uses a Convolutional Neural Network (CNN) to classify images into two categories:

* People wearing a face mask 😷
* People not wearing a face mask 😐

---

## 📌 Project Overview

The purpose of this project is to detect the presence or absence of face masks in images using computer vision techniques. The model is trained on thousands of labeled images and can be used for public safety monitoring or access control systems.

---

## 📥 Dataset

* **Source**: [Kaggle - Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* **Categories**:

  * `with_mask`: 3,725 images
  * `without_mask`: 3,828 images

---

## 🧪 Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* Google Colab (or Jupyter Notebook)

---

## ⚙️ Installation

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn kaggle
```

---

## 🚀 How to Run the Project

1. **Download Dataset from Kaggle**:

   * Set up your Kaggle API key (`kaggle.json`) and place it in the working directory.
   * Run the following commands to download and extract the dataset:

     ```bash
     kaggle datasets download -d omkargurav/face-mask-dataset
     unzip face-mask-dataset.zip
     ```

2. **Run the Notebook**:

   * Open `FaceMaskPrediction.ipynb` in Jupyter or Google Colab.
   * Follow the steps in the notebook to:

     * Load and preprocess the image data.
     * Create training and testing splits.
     * Normalize image pixel values.
     * Train a CNN model.
     * Evaluate performance and visualize predictions.

3. **Training Configuration**:

   * 80% training / 20% testing split
   * Input images resized to 128x128 pixels
   * Labels: `1` for "with mask", `0` for "without mask"

---

## 🧠 Model Architecture

The model is built using Keras Sequential API with the following layers:

* `Conv2D`: 32 filters, 3x3 kernel, ReLU activation
* `MaxPooling2D`: 2x2 pool size
* `Flatten`
* `Dense`: 128 units, ReLU
* `Dropout`: 0.5
* `Dense`: 64 units, ReLU
* `Dropout`: 0.5
* `Dense`: 2 units, Sigmoid activation (for binary classification)

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='sigmoid')
])
```

---

## 📊 Results

* The model achieves high accuracy in distinguishing between masked and unmasked faces.
* Exact results may vary depending on training epochs and dataset shuffling.

---

## 📷 Sample Predictions

Add your own predictions and visualization like:

```python
img = mpimg.imread('data/with_mask/sample.jpg')
plt.imshow(img)
```

---

## 🔮 Future Improvements

* Real-time face mask detection with webcam
* Deploy as a web app using Streamlit or Flask
* Enhance accuracy with data augmentation
* Use Transfer Learning (e.g., MobileNet, ResNet)
