# 📊 Sentiment Analysis on Social Media Text

## 🚀 Overview

This project implements a machine learning pipeline to perform **sentiment analysis on Twitter data**, classifying text into **positive** and **negative** sentiments.

It uses Natural Language Processing (NLP) techniques along with multiple machine learning models to analyze large-scale social media text efficiently.

---

## 🧠 Models Used

* Logistic Regression ✅ (Best Performance ~83%)
* Linear Support Vector Classifier (SVC)
* Bernoulli Naive Bayes

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas, NumPy
* Matplotlib, Seaborn
* WordCloud

---

## 📂 Project Structure

```
sentiment-analysis/
│
├── src/
│   ├── preprocess.py     # Text preprocessing pipeline
│   ├── train.py          # Model training and vectorization
│   ├── evaluate.py       # Model evaluation & confusion matrix
│   └── predict.py        # Prediction pipeline
│
├── models/               # Saved models (ignored in Git)
├── data/                 # Dataset (not included)
│
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md
```

---

## 📊 Dataset

* **Sentiment140 Dataset**
* Contains **1.6 million tweets**
* Labels:

  * `0 → Negative`
  * `1 → Positive`

⚠️ Dataset is not included due to size.
You can download it from Kaggle.

---

## ⚙️ Features

### 🔹 Text Preprocessing

* Lowercasing
* URL replacement
* Emoji conversion
* Username removal
* Noise cleaning (regex)
* Lemmatization

### 🔹 Feature Engineering

* TF-IDF Vectorization
* n-grams (1,2)
* Max features: 500,000

### 🔹 Model Evaluation

* Classification Report
* Confusion Matrix Visualization

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 83%      |
| Linear SVC          | 82%      |
| BernoulliNB         | 80%      |

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add dataset

Place dataset inside:

```
data/training.csv
```

### 4️⃣ Run the project

```bash
python main.py
```

---

## 🔍 Example Output

```
                              text sentiment
0                   I hate twitter  Negative
1       May the Force be with you.  Positive
2  Mr. Stark, I don't feel so good  Negative
```

---

## 💾 Model Saving

Models and vectorizer are saved using **Pickle**:

* `vectoriser.pickle`
* `lr_model.pickle`
* `bnb_model.pickle`

---

## 📌 Future Improvements

* Deep Learning Models (LSTM / BERT)
* Real-time Twitter API integration
* Web App using Streamlit
* Deployment (Docker / Cloud)

---

## 👨‍💻 Author

**Nandalal**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
