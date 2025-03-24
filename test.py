import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import datetime

class KubernetesPodFailurePredictor:
    def __init__(self, data_path):
        """
        Initialize the pod failure prediction model
        
        :param data_path: Path to the Kubernetes pod data CSV
        """
        self.data = pd.read_csv("/Users/aditya/projects_all/test/kubernetes_pod_failures.csv")
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the Kubernetes pod data
        Adds engineered features and prepares data for modeling
        """
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Feature engineering
        self.data['hour_of_day'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        
        # Create failure risk indicator
        self.data['failure_risk'] = np.where(
            (self.data['status'].isin(['CrashLoopBackOff', 'Failed'])) | 
            (self.data['restart_count'] > 5),
            1, 0
        )
        
        # Advanced anomaly detection feature
        def calculate_resource_stress(row):
            """
            Calculate a custom resource stress metric
            """
            cpu_threshold = 1.5  # High CPU usage
            memory_threshold = 400  # High memory usage
            
            stress_score = 0
            if row['cpu_usage'] > cpu_threshold:
                stress_score += 1
            if row['memory_usage'] > memory_threshold:
                stress_score += 1
            
            return stress_score
        
        self.data['resource_stress'] = self.data.apply(calculate_resource_stress, axis=1)
        
        # Create namespace risk mapping
        namespace_risk = self.data.groupby('namespace')['failure_risk'].mean().to_dict()
        self.data['namespace_historical_risk'] = self.data['namespace'].map(namespace_risk)
        
    def prepare_features(self):
        """
        Prepare features for machine learning model
        
        :return: Features and target variable
        """
        features = [
            'restart_count', 
            'cpu_usage', 
            'memory_usage', 
            'hour_of_day', 
            'day_of_week', 
            'resource_stress',
            'namespace_historical_risk'
        ]
        
        X = self.data[features]
        y = self.data['failure_risk']
        
        return X, y
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train a Random Forest Classifier for pod failure prediction
        
        :param test_size: Proportion of data to use for testing
        :param random_state: Random seed for reproducibility
        :return: Trained model and evaluation metrics
        """
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state, 
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluation
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model and scaler
        joblib.dump(model, 'pod_failure_model.joblib')
        joblib.dump(scaler, 'pod_failure_scaler.joblib')
        
        return model
    
    def predict_pod_failures(self, pod_data):
        """
        Predict failure probability for new pod data
        
        :param pod_data: Dictionary of pod metrics
        :return: Failure prediction and probability
        """
        # Load saved model and scaler
        model = joblib.load('pod_failure_model.joblib')
        scaler = joblib.load('pod_failure_scaler.joblib')
        
        # Prepare input data
        input_df = pd.DataFrame([pod_data])
        
        # Add missing engineered features
        current_time = datetime.datetime.now()
        input_df['hour_of_day'] = current_time.hour
        input_df['day_of_week'] = current_time.weekday()
        
        # Compute `resource_stress` feature
        def calculate_resource_stress(row):
            cpu_threshold = 1.5  
            memory_threshold = 400  
            
            stress_score = 0
            if row['cpu_usage'] > cpu_threshold:
                stress_score += 1
            if row['memory_usage'] > memory_threshold:
                stress_score += 1
                
            return stress_score
        
        input_df['resource_stress'] = input_df.apply(calculate_resource_stress, axis=1)

        # Compute `namespace_historical_risk`
        if pod_data['namespace'] in self.data['namespace'].unique():
            namespace_risk = self.data.groupby('namespace')['failure_risk'].mean().to_dict()
            input_df['namespace_historical_risk'] = namespace_risk.get(pod_data['namespace'], 0)
        else:
            input_df['namespace_historical_risk'] = 0  # Default risk for unseen namespaces

        # Prepare features for prediction
        features = [
            'restart_count', 
            'cpu_usage', 
            'memory_usage', 
            'hour_of_day', 
            'day_of_week',
            'resource_stress',
            'namespace_historical_risk'
        ]

        X_input = input_df[features]
        X_input_scaled = scaler.transform(X_input)
        
        prediction = model.predict(X_input_scaled)
        probability = model.predict_proba(X_input_scaled)
        
        return {
            'failure_prediction': bool(prediction[0]),
            'failure_probability': probability[0][1]
        }

def main():
    predictor = KubernetesPodFailurePredictor('kubernetes_pod_data.csv')
    predictor.train_model()
    
    sample_pod = {
        'restart_count': 3,
        'cpu_usage': 1.5,
        'memory_usage': 350,
        'namespace': 'default'
    }
    
    result = predictor.predict_pod_failures(sample_pod)
    print("\nPod Failure Prediction:")
    print(f"Will Fail: {result['failure_prediction']}")
    print(f"Failure Probability: {result['failure_probability']:.2%}")

if __name__ == "__main__":
    main()
