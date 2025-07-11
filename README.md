# Machine-Learning-Based-Employee-Promotion-Prediction

## 📌 Overview

A machine learning-based system designed to assist HR departments in making fair, efficient, and data-driven promotion decisions. The application uses multiple classification algorithms to predict employee promotion eligibility, aiming to reduce bias and improve decision accuracy.

---

## 🚀 Features

* Predicts promotion eligibility using historical HR datasets
* Implements multiple ML models: Gradient Boosting, Random Forest, SVM, KNN, and Logistic Regression
* Web interface built with Django and Flask
* Handles class imbalance using SMOTE
* Visualization dashboards for model insights

---

## 🧰 Tech Stack

* **Languages:** Python 3.8+
* **Frameworks:** Django, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Database:** SQLite3
* **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

---

## 🤖 ML Models Implemented

* ✅ Gradient Boosting (Best performance)
* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression
* K-Nearest Neighbors (KNN)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/sukanya581/Machine-Learning-Based-Employee-Promotion-Prediction.git
cd Machine-Learning-Based-Employee-Promotion-Prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install required packages
pip install -r requirements.txt

# Run the Django server
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

---

## How to Use

1. Register or log in as an admin/user
2. Upload HR data in `.csv` format
3. Train the model using the uploaded dataset
4. Enter new employee details to get a promotion prediction
5. View model accuracy and visualization charts

---

## 📈 Accuracy & Results

* **Gradient Boosting**:

  * ✅ Training Accuracy: \~99%
  * ✅ Test Accuracy: \~88%
* ROC Curve, Confusion Matrix, and Feature Importance graphs available

---

## 📁 Folder Structure

```
├── templates/           # HTML templates
├── static/              # CSS and JS files
├── views.py             # Django views
├── models.py            # Database models
├── forms.py             # Form handling
├── train.csv            # HR training dataset
├── ml_model.pkl         # Trained ML model
├── manage.py            # Django entry point
```

---

## 🙌 Author

**Sukanya Katta**
📧 [LinkedIn](https://www.linkedin.com/in/sukanya-katta/) | 🌐 [GitHub](https://github.com/sukanyakatta96)

---

## 📄 License

This project is for educational and demonstration purposes.

---
