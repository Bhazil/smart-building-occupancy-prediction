# Smart Building Occupancy Prediction

## 📌 Project Overview
This project implements an end-to-end **data engineering and machine learning pipeline**
to predict building occupancy using environmental sensor data.

The pipeline follows industry best practices:
- Modular data ingestion
- Data cleaning
- Feature engineering
- Machine learning modeling
- Model evaluation
- Workflow orchestration using **Apache Airflow**

---

## 🎯 Research Questions Addressed

### RQ1: How effective are classical machine learning models in predicting occupancy?
- Implemented **Logistic Regression** and **Random Forest**
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Generated:
  - Evaluation table
  - Model comparison figure

---

### RQ2: How does multi-sensor data fusion affect prediction performance?
- Environmental sensor data is combined during feature engineering
- Multiple sensor inputs contribute to improved prediction capability

---

### RQ3: How does feature engineering impact model performance?
- Raw sensor data is transformed into meaningful features
- Models are trained **only on engineered features**
- Performance comparison demonstrates the impact of feature engineering

---

### RQ4: How can model evaluation improve transparency and trust?
- Clear performance metrics allow transparent model comparison
- Random Forest supports future feature-importance analysis

---

### RQ5: What are the practical implications for smart buildings?
- Occupancy prediction enables:
  - Energy optimization
  - HVAC automation
  - Intelligent building management
  - Cost and energy savings

---

## 🗂 Project Structure


