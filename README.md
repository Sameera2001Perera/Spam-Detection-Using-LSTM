# 📧 Spam Detection using LSTM

## 📝 Project Overview

This project is a **Spam Detection System** built using **Natural Language Processing (NLP) and Deep Learning**. It classifies emails as **Spam or Not Spam** using an **LSTM-based neural network** trained on a balanced dataset.

## ⚙️ Features

- **Preprocessing:** Removes punctuation, stopwords, and subject lines.
- **Tokenization & Padding:** Converts text to sequences for deep learning.
- **LSTM-based Model:** Uses word embeddings and LSTM layers for classification.
- **Binary Classification:** Uses **sigmoid activation** for spam detection.
- **Callbacks:** Implements **Early Stopping & ReduceLROnPlateau** for better training.

---

## 📂 Dataset

- The dataset used is [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset), which contains email text and a binary label (`spam: 1 for Spam, 0 for Not Spam`).
- **Balancing Strategy:** Downsampling the majority class (ham emails) to avoid class imbalance.

---

## 🏗️ Model Architecture

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

- **Embedding Layer:** Converts words into dense vector representations.
- **LSTM Layer:** Captures sequential dependencies in text.
- **Dense Layer (ReLU):** Extracts non-linear features.
- **Output Layer (Sigmoid):** Outputs probability for spam classification.

---

## 📊 Model Training

### **Train the Model**

```python
history = model.fit(
    train_padded, train_y,
    validation_data=(test_padded, test_y),
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)
```

### **Model Evaluation**

Loss: 0.09856212884187698
Accuracy: 0.9799270033836365
