# 🏦 Customer Churn Prediction
### Deep Learning model to predict bank customer churn using an Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-ANN-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

---

## 📌 Problem Statement

Customer churn — when a customer leaves a bank — is a costly problem for financial institutions. Acquiring a new customer is significantly more expensive than retaining an existing one. This project builds a binary classification model to **predict whether a customer will churn**, allowing the bank to proactively intervene.

---

## 📊 Dataset

- **Source:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) — Kaggle
- **Size:** 10,000 rows × 14 columns
- **Target variable:** `Exited` (1 = churned, 0 = retained)

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is active |
| EstimatedSalary | Estimated annual salary |

---

## 🧠 Model Architecture

A fully connected **Artificial Neural Network (ANN)** built with TensorFlow/Keras:

```
Input Layer     →  11 features
Hidden Layer 1  →  11 neurons, ReLU activation
Hidden Layer 2  →  11 neurons, ReLU activation
Output Layer    →  1 neuron, Sigmoid activation
```

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Callbacks:** EarlyStopping (patience=10, restore_best_weights=True)

---

## ⚙️ Workflow

```
Raw Data
   │
   ├── Drop irrelevant columns (RowNumber, CustomerId, Surname)
   ├── One-Hot Encoding (Geography, Gender) with drop_first=True
   ├── Train/Test Split (80/20, random_state=1)
   ├── Feature Scaling (StandardScaler)
   │
   └── ANN Training (100 max epochs → stopped at epoch 35, best at epoch 25)
          │
          └── Evaluation → Accuracy, Confusion Matrix, Classification Report
```

---

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | **85.95%** |
| Precision (Churn) | 77% |
| Recall (Churn) | 46% |
| F1-Score (Churn) | 0.57 |

**Confusion Matrix:**
```
              Predicted: No    Predicted: Yes
Actual: No       1529              56
Actual: Yes       225             190
```

> 🔍 The model performs well overall, but recall for churners is 46% — a known limitation due to class imbalance (80/20 split in target). Future improvements include SMOTE oversampling or class weighting.

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the notebook**
```bash
jupyter notebook Customer_churn_prediction.ipynb
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── Customer_churn_prediction.ipynb   # Main notebook
├── Churn_Modelling.csv               # Dataset
├── requirements.txt                  # Dependencie
└── README.md                         # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Handle class imbalance with SMOTE or `class_weight`
- [ ] Experiment with deeper/wider network architectures + Dropout
- [ ] Add ROC-AUC curve and score
- [ ] Hyperparameter tuning with Keras Tuner
- [ ] Deploy model as a REST API using Flask or FastAPI

---

*If you found this project useful, consider giving it a ⭐ on GitHub!*
