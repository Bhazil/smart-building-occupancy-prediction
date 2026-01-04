# Smart Building Occupancy Prediction

## Project Overview
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

## Research Questions Addressed

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

## Project Structure

```bash
smart-building-occupancy-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── occupancy_sensor_data.csv     # Original sensor dataset
│   │
│   ├── processed/
│   │   └── cleaned_data.csv               # Cleaned & validated data
│   │
│   └── features/
│       └── engineered_features.csv        # Feature-engineered dataset
│
├── src/
│   ├── ingestion/
│   │   └── load_data.py                   # Load raw sensor data
│   │
│   ├── preprocessing/
│   │   └── clean_data.py                  # Handle missing values & data types
│   │
│   ├── feature_engineering/
│   │   └── build_features.py              # Sensor fusion & feature creation
│   │
│   ├── modeling/
│   │   ├── train_logistic_regression.py   # Classical ML model
│   │   ├── train_random_forest.py          # Tree-based ML model
│   │   └── predict.py                     # Occupancy prediction
│   │
│   └── evaluation/
│       └── evaluate_models.py             # Accuracy, Precision, Recall, F1-score
│
├── dags/
│   └── smart_building_occupancy_pipeline.py
│                                          # Apache Airflow DAG (end-to-end)
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb    # EDA & sensor behavior analysis
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison.png            # RQ1 comparison figure
│   │   └── feature_importance.png          # RQ3 visualization
│   │
│   └── tables/
│       └── evaluation_metrics.csv          # RQ4 evaluation table
│
└── logs/
    └── pipeline_execution.log              # Airflow & script logs

```

## Workflow 

The project workflow is implemented as an end-to-end automated pipeline using **Apache Airflow**.  
Each stage of the data engineering process is executed as an independent Python script and orchestrated through a DAG.

---

### Data Ingestion
**Script:** `src/ingestion/load_data.py`

- Loads raw environmental sensor data from CSV files  
- Validates input schema and file structure  
- Stores the original dataset in the `data/raw/` directory  

**Airflow Task:** `data_ingestion`

---

### Data Preprocessing
**Script:** `src/preprocessing/clean_data.py`

- Handles missing values  
- Corrects data types  
- Removes noisy and inconsistent sensor readings  
- Saves cleaned data to `data/processed/`

**Airflow Task:** `data_preprocessing`

---

### Feature Engineering
**Script:** `src/feature_engineering/build_features.py`

- Performs multi-sensor data fusion  
- Creates meaningful engineered features  
- Prepares data for model training  
- Stores features in `data/features/`

**Airflow Task:** `feature_engineering`

---

### Model Training
**Scripts:**
- `src/modeling/train_logistic_regression.py`
- `src/modeling/train_random_forest.py`

- Trains classical ML models  
- Saves trained models to the `models/` directory  

**Airflow Task:** `model_training`

---

### Model Evaluation
**Script:** `src/evaluation/evaluate_models.py`

- Computes Accuracy, Precision, Recall, and F1-score  
- Generates evaluation tables and figures  
- Stores results in `reports/`

**Airflow Task:** `model_evaluation`

---

### Pipeline Execution Order

```text
Data Ingestion
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation

```

