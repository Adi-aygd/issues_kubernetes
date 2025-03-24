
---

# 🚀 Kubernetes Pod Failure Prediction using Machine Learning

## 📌 Project Overview
This project focuses on detecting and predicting Kubernetes pod failures using machine learning. Our AI model analyzes pod metrics such as restart count, CPU usage, memory usage, and historical namespace risks to forecast potential pod failures in real-time.

✅ **Problem Addressed:**  
Pod failures like `CrashLoopBackOff` or `Failed` states can cause downtime in Kubernetes clusters. Our model proactively predicts such failures, allowing teams to take preventive actions.

---

## 📊 Dataset Description
The dataset consists of Kubernetes pod metrics including:
- `timestamp`
- `namespace`
- `status`
- `restart_count`
- `cpu_usage`
- `memory_usage`

### 🛠 Features Engineered:
- **Hour of Day**
- **Day of Week**
- **Resource Stress Score**
- **Namespace Historical Risk**
- **Failure Risk Flag**

Failure conditions were simulated with:
- High restart counts (>5)
- Pod status as `CrashLoopBackOff` or `Failed`
- Resource usage spikes

---

## 🔮 Model Used
We implemented a **Random Forest Classifier** due to its robustness with tabular data and ability to handle non-linear relationships.

### 📈 Evaluation Metrics:
| Metric              | Score |
|---------------------|-------|
| Precision (Fail)    | 0.89  |
| Recall (Fail)       | 0.78  |
| F1-Score (Fail)     | 0.83  |
| Overall Accuracy    | 82%   |

**Confusion Matrix:**
```
[[77 11]  → True Negative / False Positive
 [25 87]] → False Negative / True Positive
```

---

## 🔍 Sample Prediction Output
```
Pod Failure Prediction:
Will Fail: False
Failure Probability: 38.00%
```
👉 The model outputs whether the pod is likely to fail and the probability percentage.

---

## 💻 Project Structure
```
├── kubernetes_pod_failures.csv       # Dataset
├── test.py                           # Main training and prediction script
├── pod_failure_model.joblib          # Saved ML model
├── pod_failure_scaler.joblib         # Saved scaler
├── README.md                         # Project overview
```

---

## 🚀 How to Run
1. Clone the repo:
```
git clone <repo-link>
cd <repo-folder>
```
2. Install the dependencies:
```
pip install -r requirements.txt
```
3. Run the model:
```
python test.py
```
4. Update `sample_pod` dictionary for real-time predictions.

---

## 🧠 Future Improvements
✅ Real-time data collection from a Kubernetes cluster  
✅ Deployment as a monitoring microservice  
✅ Integration with Prometheus/Grafana for visualization  

---

## 👨‍💻 Team Credits
- **AI/ML Developer:** Aditya Gupta, Priyanshu Thakur, Ritesh Mahara 
- **Hackathon Project:** Kubernetes Pod Failure Prediction Model
- **Tools & Libraries:** Python, Pandas, Scikit-learn, Joblib

---
