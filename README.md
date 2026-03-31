# Random Forest Stress Analysis

> "A Multi-Class Stress Classification System Powered by **Random Forest** and Hyperparameter Optimization."

This project implements a robust machine learning pipeline to predict human stress levels using the **Random Forest Classifier**. By leveraging **Grid Search Cross-Validation**, the model achieves optimized performance in classifying stress states based on physiological and behavioral features.

##  Project Overview

Understanding and predicting stress is vital for mental well-being. This project goes beyond simple modeling by implementing a comprehensive data science workflow:
1.  **Data Stratification:** Splitting data into Training (70%), Validation (10%), and Testing (20%) sets to ensure rigorous evaluation.
2.  **Algorithm Selection:** Utilizing the **Random Forest** ensemble method for its robustness against overfitting and ability to handle non-linear data.
3.  **Hyperparameter Tuning:** Employing `GridSearchCV` to systematically find the optimal combination of `n_estimators`, `max_depth`, and leaf constraints.
4.  **Advanced Evaluation:** Generating Confusion Matrices, Feature Importance rankings, and Multi-class ROC curves.

## ️ Technical Stack

- **Core Algorithm:** Random Forest Classifier (Ensemble Learning)
- **Optimization:** Grid Search (Hyperparameter Tuning)
- **Data Processing:** `pandas`, `numpy`, `sklearn.model_selection`
- **Visualization:** `matplotlib`, `seaborn` (Heatmaps, ROC Curves, Bar Charts)
- **Metrics:** Accuracy, Confusion Matrix, Classification Report, ROC-AUC

##  How to Run

### 1. Prerequisites
Ensure you have Python 3 and the following libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
