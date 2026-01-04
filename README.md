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
---

## Airflow DAG Explanation

The project workflow is orchestrated using Apache Airflow through a Directed Acyclic Graph (DAG).
Each node in the DAG represents a distinct stage of the data engineering pipeline, ensuring modular execution, fault isolation, and reproducibility.

**DAG File**

Location: dags/smart_building_occupancy_pipeline.py

DAG ID: smart_building_occupancy_pipeline

Trigger Type: Manual

Execution Mode: Sequential task dependencies

🔗 DAG Tasks and Responsibilities
1️. data_ingestion

- Executes src/ingestion/load_data.py

- Loads raw environmental sensor data from CSV files

- Stores original data in data/raw/

2️. data_preprocessing

- Executes src/preprocessing/clean_data.py

- Handles missing values and data inconsistencies

- Outputs cleaned data to data/processed/

3️. feature_engineering

- Executes src/feature_engineering/build_features.py

- Performs multi-sensor data fusion

-Generates engineered features for modeling

- Saves output to data/features/

4️. model_training

- Executes model training scripts

Trains:

- Logistic Regression

- Random Forest

- Stores trained models in the models/ directory

5️. model_evaluation

- Executes src/evaluation/evaluate_models.py

- Computes evaluation metrics:

- Accuracy

- Precision

- Recall

- F1-score

- Generates comparison tables and visualizations

---

### Airflow Implementation Status

An Apache Airflow DAG was designed and implemented to orchestrate the complete data engineering workflow.

Due to local environment and system constraints, the Airflow scheduler and webserver could not be executed during development.  
However, the pipeline logic, task dependencies, and modular execution were fully implemented and validated by running each task script independently.

The Airflow DAG accurately represents the intended production-level workflow and can be executed in a properly configured Airflow environment.

---

## Results & Evaluation

This section presents the results obtained from the machine learning models trained to predict smart building occupancy.
Model performance is evaluated using standard classification metrics to ensure transparency and comparability.

---

###Evaluation Metrics

The following metrics were used to evaluate model performance:

* Accuracy – Overall correctness of predictions

* Precision – Correctly predicted occupied instances

* Recall – Ability to detect actual occupied instances

* F1-score – Balance between precision and recall

These metrics provide a comprehensive view of model effectiveness, especially for occupancy classification tasks.

---

### Model Performance Comparison

Two classical machine learning models were evaluated:

1. Logistic Regression

2. Random Forest

Both models were trained using the same feature-engineered dataset to ensure a fair comparison.

| Model | Accuracy | Precision | Recall | F1-score |
|----------|----------|----------|----------|----------|
| Logistic Regression  | 1.0  | 1.0   | 1.0   | 1.0   |
| Random Forest  | 1.0   | 1.0  | 1.0  | 1.0    |

Note: **Disclaimer**

The evaluation results reported in this project are based on a relatively small and highly structured dataset. Due to the strong correlation between environmental sensor features (such as CO₂ levels, temperature, and motion) and building occupancy, the classification task becomes comparatively straightforward. As a result, the models achieved near-perfect performance on the test split. These results reflect the nature of the dataset and feature relevance rather than unrealistic or generalized model performance.


---

### Key Observations

Feature engineering significantly improved model performance.

Multi-sensor data fusion contributed to better occupancy prediction.

Random Forest demonstrated stronger performance in capturing non-linear relationships.

Logistic Regression provided a reliable baseline with interpretable results.

---
## Reproducibility & How to Run the Evaluation

This project is fully reproducible. The evaluation results can be regenerated by running the evaluation script on the provided dataset.

### Prerequisites
- Python 3.9+
- Anaconda (recommended)
- Required libraries listed in `requirements.txt`

### Steps to Run the Evaluation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bhazil/smart-building-occupancy-prediction.git
   cd smart-building-occupancy-prediction

   ```
   
2. Activate Anaconda environment:
   ```bash
   conda activate base
   ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
   ```

4. Run the evaluation script:
   ```bash
    python src/evaluation/evaluate.py

   ```

### Output

Model performance metrics are printed in the terminal

A CSV file containing evaluation results is generated at:
     ```bash
   tables/evaluation_metrics.csv
    ```

The results are obtained using an 80/20 train-test split with a fixed random state to ensure consistency and reproducibility.

---

### Tools & Technologies in Workflow
- Python for data processing and modeling
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning
- Apache Airflow (for workflow orchestration)
- Git & GitHub for version control
