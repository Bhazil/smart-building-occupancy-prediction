# Smart Building Occupancy Prediction

##  Project Overview
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

##  Research Questions Addressed

### RQ1: How effective are classical ML models in predicting occupancy?
- Implemented **Logistic Regression** and **Random Forest**
- Evaluated using Accuracy, Precision, Recall, and F1-score
- Generated an evaluation table and model comparison figure

### RQ2: How does multi-sensor data fusion affect prediction performance?
- Environmental sensor data is combined during feature engineering
- Multiple sensor inputs improve prediction capability

### RQ3: How does feature engineering impact model performance?
- Raw sensor data is transformed into meaningful features
- Models are trained **only on engineered features**
- Performance comparison highlights feature engineering impact

### RQ4: How can model evaluation improve transparency and trust?
- Clear performance metrics enable transparent comparison
- Random Forest supports future feature-importance analysis

### RQ5: What are the practical implications for smart buildings?
- Energy optimization
- HVAC automation
- Intelligent building management
- Cost and energy savings

---

## 🗂 Project Structure

```text
smart-building-occupancy-prediction/
├── dags/
│   └── project_pipeline_dag.py
├── src/
│   ├── data_ingestion/
│   ├── data_cleaning/
│   ├── feature_engineering/
│   ├── modeling/
│   └── evaluation/
├── data/
├── tables/
├── figures/
├── README.md
└── .gitignore

