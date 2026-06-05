# modelling.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_ready_data(data_dir):
    """Memuat data secara relatif karena MLProject dijalankan di folder yang sama."""
    print(f"Memuat data dari folder: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_ready.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_ready.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_ready.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test_ready.csv")).squeeze()
    
    return X_train, X_test, y_train.astype(int), y_test.astype(int)

if __name__ == "__main__":
    # Jalur folder data disesuaikan dengan struktur di dalam folder MLProject
    DATA_DIR = "student_preprocessing"
    X_train, X_test, y_train, y_test = load_ready_data(DATA_DIR)
    
    # Set backend local tracking untuk GitHub Actions
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("Student_Stress_CI")
    
    # Menggunakan autolog sesuai kriteria basic modelling
    mlflow.sklearn.autolog(log_models=True)
    
    print("=== CI Re-training: Memulai Pelatihan Model ===")
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(random_state=42, max_depth=5, n_estimators=50)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Re-training Selesai! Akurasi Model Baru: {acc:.4f}")