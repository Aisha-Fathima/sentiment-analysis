
# Hotel Review Sentiment Analysis

A machine learning project that performs sentiment analysis on hotel reviews to determine if a customer is satisfied or dissatisfied with the services provided.

---

### Table of Contents

* [Overview](#overview)
* [Why Sentiment Analysis](#why-sentiment-analysis)
* [Model Architecture](#model-architecture)
* [Text Preprocessing](#text-preprocessing)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Results](#results)
* [License](#license)

---

### Overview

Sentiment analysis is a technique used to extract subjective information from text data. In this project, we analyze hotel reviews to understand customer sentiments — whether they are positive (satisfied) or negative (dissatisfied).

---

### Why Sentiment Analysis

Hotels receive hundreds or thousands of reviews. Manually reading and interpreting all reviews is inefficient and time-consuming. A machine learning model automates this task, enabling faster and more scalable insights into customer satisfaction.

---

### Model Architecture

This project uses:

* **CountVectorizer** to convert text data into numerical vectors.
* **Support Vector Machine (SVM)** for binary classification of sentiments.

#### Why SVM?

SVM is a robust classifier that tries to find the best hyperplane to separate classes (positive vs. negative). It works well for both linear and non-linear data.

#### Types of SVM

* **Linear SVM**: Used when data can be separated with a straight line.
* **Non-linear SVM**: Used when the data requires a more complex separation using kernel functions.

---

### Text Preprocessing

Before training the model, the text data is cleaned and prepared using the following steps:

* Convert text to lowercase
* Remove special characters and punctuation
* Remove stopwords
* Tokenization

The **NLTK** library is used for natural language processing tasks.

---

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* re (regular expressions)

---

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/hotel-review-sentiment-analysis.git
cd hotel-review-sentiment-analysis
pip install -r requirements.txt
```

Or install packages manually:

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install nltk
pip install re
```

---

### How to Run

1. Prepare your dataset (CSV format) with columns for review text and sentiment labels.
2. Run the main Python script:

```bash
python sentiment_analysis.py
```

The model will:

* Preprocess the review text
* Convert text to vectors using CountVectorizer
* Train an SVM classifier
* Predict sentiment on test data
* Display accuracy and evaluation metrics

---

### Results

After training, the model outputs:

* Accuracy score
* Confusion matrix
* Precision, Recall, and F1-score

These metrics help evaluate the model's performance on unseen data.



