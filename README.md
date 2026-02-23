# IoT Micro Gas Turbine: Predictive Analytics Using Deep Learning

**Author:** Rakesh R. Bhatija  
**Course:** AAI-530 – Data Analytics and Internet of Things  
**University:** University of San Diego  
**Instructor:** Anamika Singh, M.S  
**Date:** February 23, 2026  

---

## Project Overview

This project designs a theoretical end-to-end Internet of Things (IoT) architecture for a micro gas turbine system and applies machine learning techniques to generate predictive insights from real-world sensor data.

Micro gas turbines are widely used in distributed energy systems due to their compact size, fuel flexibility, and operational efficiency. Continuous monitoring using IoT sensors enables predictive analytics to improve efficiency, reduce downtime, and enhance reliability.

This project implements two complementary machine learning models:

- **LSTM Time-Series Forecasting Model**: Predicts future electrical power output based on historical patterns.
- **Deep Neural Network (DNN) Regression Model**: Learns the nonlinear relationship between turbine input voltage and electrical power output.

An interactive Tableau dashboard visualizes system performance and model predictions.

---

## Dataset Description

The dataset consists of real IoT sensor measurements collected from a micro gas turbine system.

**Key variables:**

- Electrical Power Output (time-series variable)  
- Input Voltage  
- Timestamp (consistent time intervals)  

The dataset is suitable for time-series modeling due to consistent temporal sampling and minimal anomalies.

---

## IoT System Architecture

The proposed IoT system follows a **three-layer architecture**:

### 1. Perception Layer (Data Acquisition & Edge Processing)

- Voltage sensors measure turbine input voltage.  
- Power sensors measure electrical power generation.  
- Optional environmental sensors (temperature, pressure).  
- Edge device (e.g., Raspberry Pi or industrial gateway):
  - Noise filtering  
  - Data normalization  
  - Aggregation  
- Communication via MQTT or HTTP over Wi-Fi/Ethernet.

### 2. Cloud Layer (Data Storage & Machine Learning)

- Data ingestion via MQTT broker.  
- Stored in a time-series database.  
- Preprocessing pipeline:
  - Cleaning  
  - Feature engineering  
  - Scaling (for DNN model)  
- Deployment of two ML models:
  - LSTM for time-series forecasting  
  - DNN for nonlinear regression  
- Optional analytics and alert engine.

### 3. Application Layer (Visualization)

- Interactive Tableau dashboard.  
- Real-time monitoring.  
- Historical trend analysis.  
- Actual vs predicted comparisons.  
- Model performance evaluation.

---

## Data Preprocessing

- Removed duplicate records.  
- Handled missing values (row-wise deletion).  
- Converted relevant columns to numeric format.  
- Sorted data chronologically.  
- Applied `StandardScaler` (for DNN model).

**LSTM Sequence Preparation:**

- Sliding window approach.  
- Window size: 20 time steps.  
- Input: Previous 20 power values.  
- Output: Next electrical power value.

Example:


## Model 1: LSTM Time-Series Forecasting

**Objective:** Predict future electrical power output using historical power values.

**Architecture:**

- Input: Sequence of 20 time steps  
- Layers:
  - LSTM layer  
  - Dense output layer  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

**Key Features:**

- Captures temporal dependencies  
- Learns trends and short-term fluctuations  
- Demonstrates stable convergence with minimal overfitting

---

## Model 2: Deep Neural Network (DNN) Regression

**Objective:** Model the nonlinear relationship between input voltage and electrical power output.

**Architecture:**

- Input: Voltage values  
- Multiple fully connected Dense layers  
- ReLU activation  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

**Performance (Test Set):**

- MSE: 13110.007  
- MAE: 92.152  

**Notes:**

- Does not model temporal dependencies.  
- Provides instantaneous prediction based on voltage input.  
- Complements the LSTM model.

---

## Model Comparison

| Model | Purpose | Strength |
|-------|---------|----------|
| LSTM | Time-series forecasting | Captures temporal dependencies |
| DNN | Nonlinear regression | Captures voltage-power relationship |

Together, the models provide:

- Future forecasting capability  
- Operational behavior understanding

---

## Tableau Dashboard Components

The Tableau dashboard includes:


**Dashboard Design Principles:**

- Consistent color scheme  
- Clear labeling  
- Logical layout  
- Focus on actionable insights

---

## Repository Structure

---
To be updated later

## How to Run the Project

1. Clone the repository  
2. Install dependencies:  
   - Python 3.x  
   - TensorFlow / Keras  
   - NumPy  
   - Pandas  
   - Scikit-learn  
   - Matplotlib  
3. Run each cell of : 
   - LSTM model notebook  
   - DNN model notebook  


---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory. Neural Computation*.  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
- OASIS. (2014). *MQTT Version 3.1.1 Standard*. https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/

---

## License

This project was developed for academic purposes as part of AAI-530 at the University of San Diego.

