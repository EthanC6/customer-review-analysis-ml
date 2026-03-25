# Customer Review Analysis using Machine Learning

This project applies Natural Language Processing (NLP) and machine learning techniques to analyze customer reviews and build predictive models. The goal is to extract insights from textual data and support decision-making in an e-commerce setting.

---

## 📌 Project Objectives

This project includes two main tasks:

### 🔹 Task 1: Recommendation Prediction
- Predict whether a customer recommends a product (binary classification)
- Target variable: `Recommended IND` (0 or 1)
- Evaluation metric: **Weighted F1-score**

### 🔹 Task 2: Star Rating Prediction
- Predict the exact star rating given in a review (1–5)
- Target variable: `Rating`
- Evaluation metric: **Accuracy**

---

## 🧠 Methodology

### Data Processing
- Combined `Title` and `Review Text` into a unified text feature
- Performed text cleaning:
  - Lowercasing
  - Removing punctuation and extra spaces
- Handled missing values:
  - Text → filled with empty strings
  - Numeric → filled with median values

---

### Feature Engineering
- Used **TF-IDF Vectorization** to convert text into numerical features
- Included:
  - Unigrams, bigrams, and trigrams
  - Stopword removal
  - Sublinear term frequency scaling

---

### Models Used

#### Task 1 (Recommendation Prediction)
- Naive Bayes (baseline)
- Logistic Regression
- **Balanced Logistic Regression (best model)**

#### Task 2 (Rating Prediction)
- Naive Bayes
- Logistic Regression (**best model**)
- Tuned Logistic Regression

---

## 📊 Results

### ✅ Task 1: Recommendation Prediction
- **Best Model:** Balanced Logistic Regression  
- **Weighted F1-score:** **0.90+**  
- Strong performance despite class imbalance

### ✅ Task 2: Star Rating Prediction
- **Best Model:** Logistic Regression  
- **Accuracy:** **~0.65**  
- Successfully exceeds the full-score threshold (≥ 0.60)

---

## 🔍 Key Insights

- Class imbalance significantly affects model performance
- Positive reviews (recommended / 5-star) are easier to predict
- Lower ratings (1–2 stars) are harder due to:
  - Fewer samples
  - Less consistent language
- Most prediction errors occur between neighboring ratings (e.g., 4 → 5)

---

## ⚠️ Limitations

- Models rely only on text (no structured features like product category)
- Difficulty in distinguishing subtle sentiment differences
- Multi-class rating prediction does not account for ordinal relationships

---

## 💡 Future Improvements

- Incorporate structured features (e.g., product category, user demographics)
- Explore deep learning models (e.g., BERT)
- Use ordinal regression for rating prediction
- Apply resampling techniques to handle class imbalance

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Google Colab

---

## 📁 Repository Structure

```text
.
├── notebook.ipynb
├── task1_predictions.csv
├── task2_predictions.csv
└── README.md
